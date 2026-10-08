"""Tests for building_occupancy.env_data."""
import json

import numpy as np
import pandas as pd
import pytest

from building_occupancy import env_data

CO2, HCHO, ERR = "Carbon dioxide (ppm)", "Formaldehyde (µg/m3)", "Formaldehyde errorCode"


def _payload(t, co2=450, hcho=10, error=None, hcho_missing=False):
    h = {"time": t} if hcho_missing else {"time": t, "value": hcho}
    if error is not None:
        h["errorCode"] = error
    return json.dumps([
        {"variable": {"name": "Carbon dioxide", "unit": "ppm"}, "values": [{"time": t, "value": co2}]},
        {"variable": {"name": "Formaldehyde", "unit": "µg/m3"}, "values": [h]},
    ])


def test_parse_reading_values_and_error_code():
    t, values = env_data.parse_reading(_payload(100, co2=500, hcho=12, error=8))
    assert t == 100
    assert values == {CO2: 500, HCHO: 12, ERR: 8}


def test_parse_reading_missing_value_is_nan():
    _, values = env_data.parse_reading(_payload(100, hcho_missing=True))
    assert np.isnan(values[HCHO])


def test_parse_reading_rejects_mixed_timestamps():
    bad = json.dumps([
        {"variable": {"name": "A", "unit": "x"}, "values": [{"time": 1, "value": 1}]},
        {"variable": {"name": "B", "unit": "x"}, "values": [{"time": 2, "value": 1}]},
    ])
    with pytest.raises(ValueError):
        env_data.parse_reading(bad)


def test_deduplicate_counts_copies_and_keeps_earliest_utc_copy():
    raw = pd.DataFrame({
        "sensorid": [1, 1, 1],
        "unix_time": [100, 100, 200],
        # 03:00+11 is 16:00 UTC; 17:30+10 is 07:30 UTC, i.e. earlier despite sorting later as text
        "createdate": ["2024-02-29 03:00:00+11", "2024-02-28 17:30:00+10", "2024-03-01 00:00:00+11"],
        CO2: [450, 450, 460],
    })
    out = env_data.deduplicate(raw)
    assert list(out["copies_in_source"]) == [2, 1]
    assert out.loc[0, "first_createdate"] == "2024-02-28 17:30:00+10"
    assert out.attrs["conflicting_readings"] == 0
    assert out.loc[0, "reading_time_utc"] == "1970-01-01 00:01:40"


def test_deduplicate_reports_conflicting_copies():
    raw = pd.DataFrame({"sensorid": [1, 1], "unix_time": [100, 100],
                        "createdate": ["2024-01-01 00:00:00+11", "2024-01-01 00:01:00+11"], CO2: [450, 999]})
    assert env_data.deduplicate(raw).attrs["conflicting_readings"] == 1


def _dedup_frame():
    day = 86400
    rows = [(9, 0, 450, np.nan, np.nan)]                                   # stray sensor, 1 reading
    rows += [(1, -500 * day, 450, np.nan, np.nan)]                         # isolated replayed record
    rows += [(1, i * 600, 400 + i, 20.0, np.nan) for i in range(5)]
    rows[3] = (1, 1200, 401, 30.0, 8.0)                                    # value with error code
    rows[4] = (1, 1800, 400, 55.0, np.nan)                                 # CO2 floor, HCHO ceiling
    df = pd.DataFrame(rows, columns=["sensorid", "unix_time", CO2, HCHO, ERR])
    df["copies_in_source"], df["first_createdate"] = 1, "x"
    return df


def test_clean_environment_rules():
    out = env_data.clean_environment(_dedup_frame(), min_sensor_readings=3, isolation_gap_days=180)
    assert set(out["sensorid"]) == {1}                     # stray sensor removed
    assert out["unix_time"].min() == 0                     # isolated record removed
    assert len(out) == 5
    suspect = out[out["formaldehyde_suspect"]]
    assert len(suspect) == 1 and suspect[HCHO].isna().all()
    assert out["co2_at_floor"].sum() == 2                  # CO2 == 400 at t=0 and t=1800
    assert out["formaldehyde_at_ceiling"].sum() == 1
    assert "copies_in_source" not in out and "first_createdate" not in out


def test_clean_environment_rejects_bad_settings():
    with pytest.raises(ValueError):
        env_data.clean_environment(_dedup_frame(), min_sensor_readings=0)
