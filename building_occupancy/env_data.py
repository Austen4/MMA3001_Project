"""
Environmental sensor data: raw export -> deduplicated readings -> cleaned readings.

The raw export (one CSV row per message received) stores each reading as a JSON list of
variables. The same reading is often received many times, so readings are deduplicated
on (sensor, timestamp). Cleaning then removes stray sensors and replayed records and
flags or blanks values the sensor itself reported as unreliable.
"""
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

KEY = ["sensorid", "unix_time"]
_ERROR_SUFFIX = " errorCode"


def parse_reading(jsondata):
    """Parse one raw ``jsondata`` payload into a timestamp and a dict of values.

    Parameters
    ----------
    jsondata : str
        JSON list of ``{"variable": {"name", "unit", ...}, "values": [{"time", "value",
        "errorCode"?}]}`` objects, as stored in the raw export.

    Returns
    -------
    unix_time : int
        Reading time, seconds since 1970-01-01 UTC.
    values : dict
        ``"<name> (<unit>)" -> value`` for every variable, plus ``"<name> errorCode"``
        where the sensor reported an error code. A variable whose value is absent maps
        to NaN.

    Raises
    ------
    ValueError
        If the payload has no timestamp or mixes several timestamps.
    """
    values, times = {}, set()
    for var in json.loads(jsondata):
        meta = var["variable"]
        column = f"{meta['name']} ({meta['unit']})"
        for v in var["values"]:
            times.add(int(v["time"]))
            values[column] = v.get("value", np.nan)
            if "errorCode" in v:
                values[meta["name"] + _ERROR_SUFFIX] = v["errorCode"]
    if len(times) != 1:
        raise ValueError(f"expected exactly one timestamp per reading, found {sorted(times)}")
    return times.pop(), values


def load_raw_environment(path=config.RAW_ENV_ZIP):
    """Read the raw environmental export (zip or csv) into one row per received message.

    Parameters
    ----------
    path : str or Path
        Raw export. A ``.zip`` must contain a single CSV.

    Returns
    -------
    pandas.DataFrame
        Columns ``sensorid`` (int), ``unix_time`` (int), ``createdate`` (str, as stored),
        then one column per measured variable and error code.
    """
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            (name,) = z.namelist()
            with z.open(name) as f:
                return _parse_raw_csv(io.TextIOWrapper(f, encoding="utf-8"))
    with open(path, encoding="utf-8") as f:
        return _parse_raw_csv(f)


def _parse_raw_csv(handle):
    """Stream-parse raw rows; the JSON column makes the file too large to load whole."""
    rows = []
    for row in csv.DictReader(handle):
        unix_time, values = parse_reading(row["jsondata"])
        rows.append({"sensorid": int(row["sensorid"]), "unix_time": unix_time,
                     "createdate": row["createdate"], **values})
    df = pd.DataFrame(rows)
    measures = [c for c in df.columns[3:] if not c.endswith(_ERROR_SUFFIX)]
    errors = [c for c in df.columns[3:] if c.endswith(_ERROR_SUFFIX)]
    return df[["sensorid", "unix_time", "createdate", *measures, *errors]]


def deduplicate(raw):
    """Collapse repeated copies of the same reading.

    Parameters
    ----------
    raw : pandas.DataFrame
        Output of :func:`load_raw_environment`.

    Returns
    -------
    pandas.DataFrame
        One row per (sensor, timestamp), sorted by sensor then time, keeping the earliest
        received copy. Adds ``reading_time_utc`` (``YYYY-mm-dd HH:MM:SS``),
        ``copies_in_source`` (number of copies received) and ``first_createdate``.
        ``attrs["conflicting_readings"]`` counts readings whose copies disagree.
    """
    missing = set(KEY + ["createdate"]) - set(raw.columns)
    if missing:
        raise ValueError(f"raw data is missing columns: {sorted(missing)}")
    created = pd.to_datetime(raw["createdate"], utc=True, format="mixed")
    ordered = raw.assign(_created=created).sort_values(KEY + ["_created"], kind="stable")
    measures = [c for c in raw.columns if c not in KEY + ["createdate"]]

    groups = ordered.groupby(KEY, sort=True)
    copies = groups.size()
    # Copies of one reading should be identical; count any that are not.
    conflicts = int((groups[measures].nunique(dropna=False) > 1).any(axis=1).sum())

    first = ordered.drop_duplicates(KEY, keep="first").set_index(KEY)
    out = first[measures].copy()
    out["copies_in_source"] = copies
    out["first_createdate"] = first["createdate"]
    out = out.reset_index()
    out.insert(1, "reading_time_utc",
               pd.to_datetime(out["unix_time"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S"))
    out.attrs["conflicting_readings"] = conflicts
    return out


def clean_environment(dedup, min_sensor_readings=config.MIN_SENSOR_READINGS,
                      isolation_gap_days=config.ISOLATION_GAP_DAYS):
    """Remove stray readings and flag or blank unreliable values.

    Rules, in order:

    1. Drop sensors with fewer than ``min_sensor_readings`` readings (stray devices).
    2. Drop readings more than ``isolation_gap_days`` from both neighbouring readings of
       the same sensor (replayed records far outside the sensor's real deployment).
    3. Blank formaldehyde values reported together with an error code and mark them in
       ``formaldehyde_suspect``.
    4. Flag CO2 at the sensor floor (``co2_at_floor``) and formaldehyde at its ceiling
       (``formaldehyde_at_ceiling``); these values are kept.
    5. Drop the ingestion columns ``copies_in_source`` and ``first_createdate``.

    Parameters
    ----------
    dedup : pandas.DataFrame
        Output of :func:`deduplicate`.
    min_sensor_readings : int
        Minimum readings for a sensor to be kept.
    isolation_gap_days : float
        Gap that makes a reading isolated.

    Returns
    -------
    pandas.DataFrame
        Cleaned readings sorted by sensor and time.
    """
    if min_sensor_readings < 1 or isolation_gap_days <= 0:
        raise ValueError("min_sensor_readings must be >= 1 and isolation_gap_days > 0")
    df = dedup.sort_values(KEY, kind="stable").reset_index(drop=True)

    counts = df["sensorid"].map(df["sensorid"].value_counts())
    df = df[counts >= min_sensor_readings].reset_index(drop=True)

    gap_s = isolation_gap_days * 86400
    by_sensor = df.groupby("sensorid")["unix_time"]
    gap_prev, gap_next = by_sensor.diff(), -by_sensor.diff(-1)
    isolated = (gap_prev.isna() | (gap_prev > gap_s)) & (gap_next.isna() | (gap_next > gap_s))
    df = df[~isolated].copy()

    hcho, err = "Formaldehyde (µg/m3)", "Formaldehyde errorCode"
    suspect = df[err].notna() & df[hcho].notna()
    df.loc[suspect, hcho] = np.nan
    df["formaldehyde_suspect"] = suspect
    df["co2_at_floor"] = df["Carbon dioxide (ppm)"] == config.CO2_SENSOR_FLOOR_PPM
    df["formaldehyde_at_ceiling"] = df[hcho] == config.FORMALDEHYDE_CEILING
    return df.drop(columns=["copies_in_source", "first_createdate"]).reset_index(drop=True)


def matches_reference(df, reference_csv, sort_by=KEY):
    """Check a generated table against a committed reference CSV, ignoring dtypes.

    Parameters
    ----------
    df : pandas.DataFrame
        Generated table.
    reference_csv : str or Path
        Reference file (UTF-8, optionally with BOM).
    sort_by : list of str
        Columns to sort both tables by before comparing.

    Returns
    -------
    (bool, str)
        Whether they match, and a short explanation.
    """
    ref = pd.read_csv(reference_csv, encoding="utf-8-sig")
    gen = pd.read_csv(io.StringIO(df.to_csv(index=False)))  # round-trip for identical parsing
    if list(ref.columns) != list(gen.columns):
        return False, f"column mismatch: {set(ref.columns) ^ set(gen.columns)}"
    if len(ref) != len(gen):
        return False, f"row count {len(gen)} vs reference {len(ref)}"
    ref = ref.sort_values(sort_by).reset_index(drop=True)
    gen = gen.sort_values(sort_by).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(gen, ref, check_dtype=False, check_exact=False)
    except AssertionError as e:
        return False, str(e).splitlines()[0]
    return True, f"identical ({len(ref)} rows)"
