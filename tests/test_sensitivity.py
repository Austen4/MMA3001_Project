"""Tests for building_occupancy.sensitivity and the hold-limit pass-through in occupancy_data."""
import numpy as np
import pandas as pd
import pytest

from building_occupancy import config, models, occupancy_data, sensitivity

RNG = np.random.default_rng(5)


def _dataset(days=60):
    t = pd.date_range("2024-03-04", periods=days * 24, freq="h", tz="UTC")
    local = t.tz_convert(config.TIMEZONE)
    df = pd.DataFrame(RNG.normal(size=(len(t), len(models.ENV_FEATURES))), columns=models.ENV_FEATURES)
    df["time_utc"] = t
    df["hour_local"] = local.hour + 0.5
    df["weekend"] = local.dayofweek >= 5
    df[config.TARGET] = np.clip(3 + 2 * df["co2"] + df["temp"], 0, None)
    return df


def test_ablation_sets_remove_each_group_once():
    sets = sensitivity.ablation_feature_sets()
    assert sets["all features"] == models.ENV_FEATURES
    for group, cols in sensitivity.FEATURE_GROUPS.items():
        kept = sets[f"without {group}"]
        assert not set(cols) & set(kept)
        assert len(kept) == len(models.ENV_FEATURES) - len(cols)
    assert sets["CO2 only (level + rate)"] == ["co2", "dco2_dt", "dco2_dt_lag1"]


def test_feature_groups_cover_every_feature_exactly_once():
    grouped = [f for cols in sensitivity.FEATURE_GROUPS.values() for f in cols]
    assert sorted(grouped) == sorted(models.ENV_FEATURES)


def test_model_factories_respect_features_and_alpha():
    f = sensitivity.model_factories(features=["co2"], alphas=[3.0])
    assert list(f) == list(sensitivity.MODEL_NAMES)
    ridge, poly = f[sensitivity.MODEL_NAMES[1]](), f[sensitivity.MODEL_NAMES[2]]()
    assert ridge.features == ["co2"] and list(ridge.alphas) == [3.0]
    assert poly.degrees == (2,) and poly.features == ["co2"]
    with pytest.raises(ValueError):
        sensitivity.model_factories(features=[])


def test_training_window_limits_training_rows():
    df = _dataset()
    full = sensitivity.holdout_scores(df, {"mean": models.MeanBaseline})
    short = sensitivity.holdout_scores(df, {"mean": models.MeanBaseline}, train_days=7)
    assert short["n_train"].item() == 7 * 24
    assert full["n_train"].item() > short["n_train"].item()
    assert full["n_test"].item() == short["n_test"].item()


@pytest.mark.parametrize("days", [0, -1, 1])
def test_training_window_validation(days):
    with pytest.raises(ValueError):
        sensitivity.holdout_scores(_dataset(), {"mean": models.MeanBaseline}, train_days=days)


def test_scenario_scores_adds_cv_summary():
    res = sensitivity.scenario_scores(_dataset(), {"mean": models.MeanBaseline})
    assert {"r2", "mae", "cv_r2_mean", "cv_r2_std", "cv_mae_mean"} <= set(res.columns)
    no_cv = sensitivity.scenario_scores(_dataset(), {"mean": models.MeanBaseline}, cv=False)
    assert "cv_r2_mean" not in no_cv


def test_bin_occupancy_passes_hold_limits():
    t = pd.to_datetime([0, 3 * 3600], unit="s", utc=True)
    events = pd.DataFrame({"floorspaceid": ["a", "a"], "occupancystatus": [occupancy_data.OCCUPIED,
                                                                          occupancy_data.NOT_OCCUPIED],
                           "headcount": [2.0, 0.0], "t": t})
    short = occupancy_data.bin_occupancy(events, 60, hold_occupied_s=3600)
    long = occupancy_data.bin_occupancy(events, 60, hold_occupied_s=4 * 3600)
    assert len(short) == 1 and len(long) == 3
