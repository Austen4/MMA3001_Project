"""
Final models: training, saving and applying them to new environmental readings.

Two final models are provided, both estimating the **building headcount** (time-weighted
mean number of people over each hour, summed over the four monitored rooms in
``config.BUILDING_ROOMS``) from one environmental sensor:

``ridge_env_time``
    Ridge regression on environmental features plus smooth time-of-day terms.
``poly2``
    Degree-2 polynomial ridge regression on environmental features only.

Inputs
------
A table of sensor readings, one row per reading (any interval up to the bin width):

========== ======= ================== ==========================================
column     unit    accepted range     notes
========== ======= ================== ==========================================
timestamp  —       ISO 8601           or ``unix_time`` (s since epoch, UTC);
                                      timestamps without an offset are taken as UTC
co2        ppm     300 – 5000         **required**
temp       °C      −20 – 60           optional
rh         %RH     0 – 100            optional
voc        ppb     0 – 10 000         optional
pm25       µg/m³   0 – 1000           optional
pressure   mb      800 – 1100         optional (not used by the models)
========== ======= ================== ==========================================

The sensor export's own column names (e.g. ``Carbon dioxide (ppm)``) are also accepted.

Handling of invalid, missing and unsupported input
--------------------------------------------------
* Missing file, no timestamp column or no ``co2`` column, or no parseable timestamps:
  :class:`InputError` (nothing is predicted).
* Unparseable timestamps, duplicate timestamps, non-numeric values and values outside the
  accepted range: the value (or row) is discarded and counted in ``attrs["issues"]``.
* An hour with no CO2 reading, or whose previous hour has none (so the CO2 rate of change
  is unknown): no prediction, ``status = "no_prediction"``.
* Any other missing feature: imputed with the training median, ``status = "warning"``.
* Features outside the range seen in training (extrapolation): predicted, but
  ``status = "warning"``. Polynomial models in particular can be unreliable there.
* The models were trained on sensor 6012002000869 (building 60, G.38). Readings from a
  sensor elsewhere are outside the validated scope; the tool cannot detect this.
"""
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import config, dataset, evaluation, models

FINAL_MODELS = {
    "ridge_env_time": ("Ridge (environment + time)", lambda: models.RidgeRegression(include_time=True)),
    "poly2": ("Polynomial ridge (degree 2)", lambda: models.PolynomialRidgeRegression(degrees=2)),
}
# Model inputs whose training range is recorded and checked for extrapolation.
RANGE_FEATURES = ["co2", "dco2_dt", "temp", "rh", "voc", "pm25"]
REQUIRED_FEATURES = ["co2", "dco2_dt"]


class InputError(ValueError):
    """Prediction input that cannot be used at all."""


# ---------------------------------------------------------------- input

def read_readings(path):
    """Read and validate a CSV of environmental readings.

    Parameters
    ----------
    path : str or Path
        CSV file (UTF-8, optional BOM) in the format described in the module docstring.

    Returns
    -------
    pandas.DataFrame
        See :func:`normalise_readings`.

    Raises
    ------
    InputError
        If the file does not exist or cannot be used.
    """
    path = Path(path)
    if not path.is_file():
        raise InputError(f"input file not found: {path}")
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except (pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError) as e:
        raise InputError(f"could not read {path.name} as CSV: {e}") from e
    return normalise_readings(df)


def normalise_readings(df):
    """Standardise column names, parse timestamps and remove invalid values.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw readings table.

    Returns
    -------
    pandas.DataFrame
        ``unix_time`` (int, s UTC) and one column per variable in
        ``config.PREDICTION_INPUTS`` (float, NaN where missing or invalid), sorted by time
        with duplicate timestamps removed. ``attrs["issues"]`` maps each problem found
        to the number of values or rows affected.

    Raises
    ------
    InputError
        If there is no timestamp column, no required column, or no valid timestamp.
    """
    df = df.rename(columns={long: short for short, (long, _, _) in config.PREDICTION_INPUTS.items()})
    issues = {}
    if "unix_time" in df.columns:
        t = pd.to_numeric(df["unix_time"], errors="coerce")
    elif "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce", format="mixed")
        t = (ts - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)
    else:
        raise InputError("input needs a 'timestamp' (ISO 8601) or 'unix_time' column")
    missing = [c for c in config.REQUIRED_PREDICTION_INPUTS if c not in df.columns]
    if missing:
        raise InputError(f"required column(s) missing: {missing}")
    if t.isna().all():
        raise InputError("no valid timestamps in input")
    if t.isna().any():
        issues["rows with unparseable timestamps (dropped)"] = int(t.isna().sum())

    out = pd.DataFrame({"unix_time": t})
    for short, (_, unit, (lo, hi)) in config.PREDICTION_INPUTS.items():
        if short not in df.columns:
            out[short] = np.nan
            if short not in ("pressure",):
                issues[f"column '{short}' absent (imputed)"] = len(df)
            continue
        v = pd.to_numeric(df[short], errors="coerce")
        bad_type = v.isna() & df[short].notna()
        bad_range = v.notna() & ((v < lo) | (v > hi))
        if bad_type.any():
            issues[f"non-numeric {short} values (removed)"] = int(bad_type.sum())
        if bad_range.any():
            issues[f"{short} outside {lo:g}-{hi:g} {unit} (removed)"] = int(bad_range.sum())
        out[short] = v.mask(bad_range).astype(float)

    out = out.dropna(subset=["unix_time"])
    out["unix_time"] = out["unix_time"].astype(np.int64)
    dup = out["unix_time"].duplicated()
    if dup.any():
        issues["duplicate timestamps (later copies dropped)"] = int(dup.sum())
    out = out[~dup].sort_values("unix_time").reset_index(drop=True)
    out.attrs["issues"] = issues
    return out


def prepare_features(readings, bin_minutes=config.FINAL_BIN_MINUTES):
    """Bin validated readings and derive the model features.

    Parameters
    ----------
    readings : pandas.DataFrame
        Output of :func:`normalise_readings`.
    bin_minutes : int
        Bin width the models were trained with.

    Returns
    -------
    pandas.DataFrame
        One row per bin with readings, with the columns produced by
        :func:`building_occupancy.dataset.add_features`.
    """
    long = readings.rename(columns={s: lng for s, (lng, _, _) in config.PREDICTION_INPUTS.items()})
    long["sensorid"] = 0
    return dataset.add_features(dataset.bin_environment(long, 0, bin_minutes), bin_minutes)


# ---------------------------------------------------------------- training and persistence

def train_final_models(df, sensor=config.FINAL_SENSOR, bin_minutes=config.FINAL_BIN_MINUTES):
    """Validate, then train, both final models.

    Each model is first fitted on the earliest ``1 - config.TEST_FRACTION`` of ``df`` and
    scored on the rest (the reported performance); the delivered model is then refitted
    on all of ``df`` with the same procedure.

    Parameters
    ----------
    df : pandas.DataFrame
        Modelling dataset from :func:`building_occupancy.dataset.build_dataset`.
    sensor, bin_minutes
        Recorded in the model card.

    Returns
    -------
    dict
        ``{"models": {key: fitted model}, "card": dict}``; the card documents data,
        settings, hold-out performance and training ranges.
    """
    train, test = evaluation.chronological_split(df)
    entries, fitted = {}, {}
    for key, (name, make) in FINAL_MODELS.items():
        validated = make().fit(train, train[config.TARGET].to_numpy())
        metrics = evaluation.regression_metrics(test[config.TARGET], validated.predict(test))
        final = make().fit(df, df[config.TARGET].to_numpy())
        fitted[key] = final
        entries[key] = {
            "name": name,
            "alpha": final.alpha_,
            "degree": getattr(final, "degree_", 1),
            "n_coefficients": int(len(final.coef_)),
            "holdout_metrics": {k: round(v, 4) for k, v in metrics.items()},
        }
    card = {
        "target": f"{config.TARGET}: mean people per {bin_minutes}-min bin, summed over rooms "
                  f"{list(config.BUILDING_ROOMS.values())}",
        "sensor": sensor,
        "sensor_location": config.ENV_SENSORS.get(sensor, "unknown"),
        "bin_minutes": bin_minutes,
        "training_period_utc": [f"{df['time_utc'].min()}", f"{df['time_utc'].max()}"],
        "n_training_rows": len(df),
        "holdout": {"from_utc": f"{test['time_utc'].iloc[0]}", "n_rows": len(test),
                    "note": "metrics from models trained on the earlier rows only"},
        "models": entries,
        "input_ranges_in_training": {f: [float(df[f].min()), float(df[f].max())] for f in RANGE_FEATURES},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
    }
    return {"models": fitted, "card": card}


def save_bundle(bundle, model_file=config.FINAL_MODEL_FILE, card_file=config.FINAL_MODEL_CARD):
    """Write the fitted models (joblib) and the human-readable model card (JSON)."""
    Path(model_file).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_file)
    Path(card_file).write_text(json.dumps(bundle["card"], indent=2, ensure_ascii=False), encoding="utf-8")


def load_bundle(model_file=config.FINAL_MODEL_FILE):
    """Load a bundle written by :func:`save_bundle`.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist (run ``scripts/06_train_final_models.py``).
    """
    model_file = Path(model_file)
    if not model_file.is_file():
        raise FileNotFoundError(f"{model_file} not found - run scripts/06_train_final_models.py first")
    bundle = joblib.load(model_file)
    if set(bundle) != {"models", "card"}:
        raise ValueError(f"{model_file} is not a final-model bundle")
    return bundle


# ---------------------------------------------------------------- prediction

def predict(readings, bundle):
    """Predict building headcount for each hour of validated readings.

    Parameters
    ----------
    readings : pandas.DataFrame
        Output of :func:`normalise_readings` / :func:`read_readings`.
    bundle : dict
        Output of :func:`load_bundle`.

    Returns
    -------
    pandas.DataFrame
        One row per bin:

        ``time_utc``, ``time_local``
            Bin start (UTC and Sydney local time).
        ``n_readings``
            Valid readings in the bin.
        ``<model>_headcount``
            Expected mean number of people in the monitored rooms over the bin
            (>= 0; NaN when ``status`` is ``no_prediction``).
        ``status``
            ``ok``, ``warning`` (predicted, but see ``warnings``) or ``no_prediction``.
        ``warnings``
            Semicolon-separated reasons.
    """
    card = bundle["card"]
    feats = prepare_features(readings, card["bin_minutes"])
    usable = feats[REQUIRED_FEATURES].notna().all(axis=1).to_numpy()
    out = pd.DataFrame({
        "time_utc": feats["time_utc"],
        "time_local": feats["time_utc"].dt.tz_convert(config.TIMEZONE),
        "n_readings": feats["n_readings"],
    })
    for key, model in bundle["models"].items():
        pred = np.full(len(feats), np.nan)
        if usable.any():
            pred[usable] = model.predict(feats[usable])
        out[f"{key}_headcount"] = pred

    ranges = card["input_ranges_in_training"]
    reasons = []
    for i in range(len(feats)):
        row = feats.iloc[i]
        if not usable[i]:
            reasons.append("no CO2 reading in this bin" if pd.isna(row["co2"])
                           else "no CO2 reading in previous bin (rate of change unknown)")
            continue
        r = [f"imputed {f}" for f in models.ENV_FEATURES if f not in REQUIRED_FEATURES and pd.isna(row[f])]
        r += [f"{f} outside training range" for f, (lo, hi) in ranges.items()
              if pd.notna(row[f]) and not lo <= row[f] <= hi]
        reasons.append("; ".join(r))
    out["warnings"] = reasons
    out["status"] = np.where(~usable, "no_prediction", np.where(out["warnings"] == "", "ok", "warning"))
    return out.reset_index(drop=True)
