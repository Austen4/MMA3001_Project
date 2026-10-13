"""Tests for building_occupancy.plotting."""
import numpy as np
import pandas as pd
import pytest

from building_occupancy import config, plotting


def _test_rows(n=48):
    return pd.DataFrame({"time_utc": pd.date_range("2024-05-20", periods=n, freq="h", tz="UTC"),
                         config.TARGET: np.arange(n, dtype=float) % 10})


def test_prediction_table_columns_and_local_time():
    test = _test_rows(3)
    out = plotting.prediction_table(test, {"a": np.ones(3), "b": np.zeros(3)})
    assert list(out.columns) == ["time_utc", "time_local", "observed", "a", "b"]
    assert out["time_local"].iloc[0].hour == 10            # 00:00 UTC = 10:00 AEST in May


def test_prediction_table_rejects_misaligned_predictions():
    with pytest.raises(ValueError):
        plotting.prediction_table(_test_rows(3), {"a": np.ones(2)})


def test_plot_test_period_writes_png(tmp_path):
    test = _test_rows()
    path = tmp_path / "fig.png"
    plotting.plot_test_period(test, {"model": test[config.TARGET].to_numpy() + 1}, path, "caption", days=1)
    assert path.stat().st_size > 1000


def test_plot_test_period_handles_off_scale_predictions_and_long_captions(tmp_path):
    test = _test_rows()
    pred = test[config.TARGET].to_numpy().copy()
    pred[[5, 6, 20]] = 500.0                                # two excursions above the cap
    path = tmp_path / "fig.png"
    plotting.plot_test_period(test, {"m": pred}, path, "word " * 80, days=2)
    assert path.exists()


def test_plot_test_period_needs_predictions(tmp_path):
    with pytest.raises(ValueError):
        plotting.plot_test_period(_test_rows(), {}, tmp_path / "x.png", "caption")
