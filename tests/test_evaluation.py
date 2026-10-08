"""Tests for building_occupancy.evaluation."""
import numpy as np
import pandas as pd
import pytest

from building_occupancy import evaluation, models


def test_regression_metrics_known_values():
    m = evaluation.regression_metrics([1, 2, 3], [1, 2, 4])
    assert m["mae"] == pytest.approx(1 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(1 / 3))
    assert m["bias"] == pytest.approx(1 / 3)
    assert m["r2"] == pytest.approx(0.5)


@pytest.mark.parametrize("y, p", [([1, 2], [1]), ([1], [1]), ([1, np.nan], [1, 2])])
def test_regression_metrics_validation(y, p):
    with pytest.raises(ValueError):
        evaluation.regression_metrics(y, p)


def _frame(n):
    return pd.DataFrame({"time_utc": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")[::-1],
                         "building_headcount": np.arange(n, dtype=float)})


def test_chronological_split_orders_and_sizes():
    train, test = evaluation.chronological_split(_frame(100), test_fraction=0.25)
    assert len(train) == 75 and len(test) == 25
    assert train["time_utc"].max() < test["time_utc"].min()


@pytest.mark.parametrize("frac", [0, 1, 1.5])
def test_chronological_split_rejects_bad_fraction(frac):
    with pytest.raises(ValueError):
        evaluation.chronological_split(_frame(10), test_fraction=frac)


def test_rolling_origin_splits_are_expanding_and_disjoint():
    splits = evaluation.rolling_origin_splits(100, n_folds=3, min_train_fraction=0.4)
    assert len(splits) == 3
    for tr, te in splits:
        assert tr.max() < te.min() and len(set(tr) & set(te)) == 0
    assert [len(tr) for tr, _ in splits] == [40, 60, 80]


def test_cross_validate_runs_each_model_on_each_fold():
    res = evaluation.cross_validate({"mean": models.MeanBaseline}, _frame(60), n_folds=3)
    assert len(res) == 3 and set(res["fold"]) == {0, 1, 2}
