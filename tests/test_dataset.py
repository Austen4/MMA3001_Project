"""Tests for building_occupancy.dataset."""
import numpy as np
import pandas as pd
import pytest

from building_occupancy import config, dataset


def _env(times_min, co2):
    n = len(times_min)
    df = pd.DataFrame({"sensorid": 1, "unix_time": np.asarray(times_min) * 60})
    for col in config.ENV_COLUMNS:
        df[col] = 20.0
    df["Carbon dioxide (ppm)"] = co2
    return df


def test_bin_environment_averages_readings():
    out = dataset.bin_environment(_env([0, 10, 30], [400, 420, 500]), 1, 30)
    assert list(out["co2"]) == [410, 500]
    assert list(out["n_readings"]) == [2, 1]


def test_bin_environment_unknown_sensor():
    with pytest.raises(ValueError):
        dataset.bin_environment(_env([0], [400]), 99, 30)


def test_backward_difference_and_gaps():
    binned = dataset.bin_environment(_env([0, 30, 60, 120], [400, 430, 445, 500]), 1, 30)
    f = dataset.add_features(binned, 30)
    # 30 ppm in 0.5 h = 60 ppm/h; 15 ppm in 0.5 h = 30 ppm/h; bin at 120 min follows a gap.
    assert np.isnan(f.loc[0, "dco2_dt"])
    assert f.loc[1, "dco2_dt"] == pytest.approx(60)
    assert f.loc[2, "dco2_dt"] == pytest.approx(30)
    assert np.isnan(f.loc[3, "dco2_dt"]) and np.isnan(f.loc[3, "dco2_dt_lag1"])
    assert f.loc[2, "dco2_dt_lag1"] == pytest.approx(60) and np.isnan(f.loc[1, "dco2_dt_lag1"])


def test_calendar_uses_local_bin_midpoint():
    # 2024-03-04 00:00 UTC is Monday 11:00 in Sydney (UTC+11); 30-min bin midpoint -> 11.25 h.
    t0 = int(pd.Timestamp("2024-03-04", tz="UTC").timestamp() // 60)
    f = dataset.add_features(dataset.bin_environment(_env([t0], [400]), 1, 30), 30)
    assert f.loc[0, "hour_local"] == pytest.approx(11.25)
    assert f.loc[0, "dow"] == 0 and not f.loc[0, "weekend"]


def test_build_dataset_requires_overlap():
    building = pd.DataFrame({"time_utc": pd.to_datetime([10 ** 9], unit="s", utc=True),
                             config.TARGET: [1.0], "rooms_occupied": [1.0]})
    with pytest.raises(ValueError):
        dataset.build_dataset(_env([0, 30], [400, 410]), building, 1, 30)
