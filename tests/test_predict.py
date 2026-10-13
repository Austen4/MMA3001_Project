"""Tests for building_occupancy.predict and the scripts/predict.py command line."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from building_occupancy import config, predict

RNG = np.random.default_rng(3)
ROOT = Path(__file__).resolve().parents[1]


def _readings(hours=24 * 40, start="2024-03-04"):
    """Synthetic 10-minute readings with a daily occupancy-driven CO2 cycle."""
    t = pd.date_range(start, periods=hours * 6, freq="10min", tz="UTC")
    local_h = t.tz_convert(config.TIMEZONE).hour
    people = np.where((local_h >= 9) & (local_h < 17) & (t.dayofweek < 5), 8.0, 0.0)
    co2 = 420 + 15 * pd.Series(people).rolling(6, min_periods=1).mean().to_numpy() + RNG.normal(0, 3, len(t))
    return pd.DataFrame({"timestamp": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "co2": co2,
                         "temp": 21 + people / 4 + RNG.normal(0, 0.2, len(t)), "rh": 50.0,
                         "voc": 0.0, "pm25": 1.0}), people


@pytest.fixture(scope="module")
def bundle():
    raw, _ = _readings()
    feats = predict.prepare_features(predict.normalise_readings(raw))
    df = feats.dropna(subset=predict.REQUIRED_FEATURES).copy()
    df[config.TARGET] = np.clip((df["co2"] - 420) / 15, 0, None)      # known signal
    return predict.train_final_models(df.reset_index(drop=True))


# ---------------------------------------------------------------- input handling

def test_export_column_names_are_accepted():
    df = pd.DataFrame({"unix_time": [0, 600], "Carbon dioxide (ppm)": [450, 460], "Temperature (°C)": [21, 22]})
    out = predict.normalise_readings(df)
    assert list(out["co2"]) == [450, 460] and list(out["temp"]) == [21, 22]


def test_invalid_values_are_removed_and_counted():
    df = pd.DataFrame({"timestamp": ["2024-01-01T00:00Z", "2024-01-01T00:10Z", "bad", "2024-01-01T00:30Z",
                                     "2024-01-01T00:30Z"],
                       "co2": [450, "x", 460, 9999, 470], "rh": [50, 50, 50, 120, 50]})
    out = predict.normalise_readings(df)
    issues = out.attrs["issues"]
    assert len(out) == 3                                             # bad time and duplicate dropped
    assert issues["rows with unparseable timestamps (dropped)"] == 1
    assert issues["non-numeric co2 values (removed)"] == 1
    assert issues["co2 outside 300-5000 ppm (removed)"] == 1
    assert issues["rh outside 0-100 %RH (removed)"] == 1
    assert issues["duplicate timestamps (later copies dropped)"] == 1
    assert out["co2"].isna().sum() == 2


def test_naive_timestamps_are_utc():
    out = predict.normalise_readings(pd.DataFrame({"timestamp": ["1970-01-01 00:10"], "co2": [450]}))
    assert out.loc[0, "unix_time"] == 600


@pytest.mark.parametrize("df", [pd.DataFrame({"co2": [450]}),                       # no time column
                                pd.DataFrame({"timestamp": ["2024-01-01"], "temp": [20]}),  # no co2
                                pd.DataFrame({"timestamp": ["nonsense"], "co2": [450]})])   # no valid time
def test_unusable_input_raises(df):
    with pytest.raises(predict.InputError):
        predict.normalise_readings(df)


def test_read_readings_missing_file(tmp_path):
    with pytest.raises(predict.InputError):
        predict.read_readings(tmp_path / "nope.csv")


# ---------------------------------------------------------------- training, saving, predicting

def test_model_card_contents(bundle):
    card = bundle["card"]
    assert set(card["models"]) == set(predict.FINAL_MODELS)
    assert card["models"]["poly2"]["degree"] == 2
    assert card["models"]["poly2"]["n_coefficients"] == 54
    assert card["models"]["ridge_env_time"]["holdout_metrics"]["r2"] > 0.8   # strong synthetic signal
    lo, hi = card["input_ranges_in_training"]["co2"]
    assert lo < hi


def test_save_and_load_roundtrip(bundle, tmp_path):
    predict.save_bundle(bundle, tmp_path / "m.joblib", tmp_path / "m.json")
    loaded = predict.load_bundle(tmp_path / "m.joblib")
    assert loaded["card"] == bundle["card"]
    assert (tmp_path / "m.json").read_text(encoding="utf-8").startswith("{")


def test_load_bundle_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        predict.load_bundle(tmp_path / "missing.joblib")


def test_predictions_track_known_occupancy(bundle):
    raw, _ = _readings(hours=24 * 7, start="2024-05-06")
    out = predict.predict(predict.normalise_readings(raw), bundle)
    assert {"time_utc", "time_local", "n_readings", "ridge_env_time_headcount", "poly2_headcount",
            "status", "warnings"} <= set(out.columns)
    assert out.loc[0, "status"] == "no_prediction"                   # no previous hour
    ok = out[out["status"] != "no_prediction"]
    assert (ok[["ridge_env_time_headcount", "poly2_headcount"]] >= 0).all().all()
    local = ok["time_local"]
    busy = ok[(local.dt.hour >= 11) & (local.dt.hour < 16) & (local.dt.dayofweek < 5)]
    night = ok[(local.dt.hour < 6)]
    for col in ("ridge_env_time_headcount", "poly2_headcount"):
        assert busy[col].mean() > night[col].mean() + 3


def test_extrapolation_and_imputation_are_flagged(bundle):
    raw, _ = _readings(hours=4, start="2024-05-06")
    raw["co2"] = raw["co2"] + 1500                                   # far above training CO2
    raw = raw.drop(columns=["voc"])
    out = predict.predict(predict.normalise_readings(raw), bundle)
    flagged = out[out["status"] == "warning"]
    assert len(flagged) == len(out) - 1
    assert flagged["warnings"].str.contains("co2 outside training range").all()
    assert flagged["warnings"].str.contains("imputed voc").all()


def test_gap_in_readings_gives_no_prediction(bundle):
    raw, _ = _readings(hours=6, start="2024-05-06")
    t = pd.to_datetime(raw["timestamp"])
    raw = raw[(t < t.iloc[0] + pd.Timedelta(hours=2)) | (t >= t.iloc[0] + pd.Timedelta(hours=3))]
    out = predict.predict(predict.normalise_readings(raw), bundle)
    after_gap = out[out["time_utc"] == t.iloc[0] + pd.Timedelta(hours=3)]
    assert after_gap["status"].item() == "no_prediction"
    assert "previous bin" in after_gap["warnings"].item()


# ---------------------------------------------------------------- command line

def _cli():
    spec = importlib.util.spec_from_file_location("predict_cli", ROOT / "scripts" / "predict.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_end_to_end_and_exit_codes(bundle, tmp_path):
    cli = _cli()
    model_file = tmp_path / "m.joblib"
    predict.save_bundle(bundle, model_file, tmp_path / "m.json")
    raw, _ = _readings(hours=12, start="2024-05-06")
    src = tmp_path / "in.csv"
    raw.to_csv(src, index=False)
    assert cli.main([str(src), "--model-file", str(model_file)]) == 0
    assert (tmp_path / "in_predictions.csv").exists()
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp": ["2024-01-01"]}).to_csv(bad, index=False)
    assert cli.main([str(bad), "--model-file", str(model_file)]) == 1
    assert cli.main([str(src), "--model-file", str(tmp_path / "none.joblib")]) == 2
