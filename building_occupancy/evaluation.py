"""
Validation utilities: chronological splits, rolling-origin cross-validation and metrics.

Occupancy is strongly autocorrelated and its level drifts over months, so random K-fold
splits would leak future information into training. Every split here trains on earlier
data and tests on later data.
"""
import time

import numpy as np
import pandas as pd

from . import config


def regression_metrics(y_true, y_pred):
    """Standard regression error metrics.

    Parameters
    ----------
    y_true, y_pred : array_like
        Observed and predicted values (same length, at least 2).

    Returns
    -------
    dict
        ``mae`` and ``rmse`` (target units), ``bias`` (mean prediction - mean observed)
        and ``r2`` (1 - SSE/SST; 0 = no better than the test-set mean, negative = worse).
    """
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    if y.shape != p.shape or y.size < 2:
        raise ValueError("y_true and y_pred must have the same length (>= 2)")
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("y_true and y_pred must be finite")
    err = p - y
    sst = ((y - y.mean()) ** 2).sum()
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()),
        "r2": float(1 - (err ** 2).sum() / sst) if sst > 0 else np.nan,
    }


def chronological_split(df, test_fraction=config.TEST_FRACTION, time_col="time_utc"):
    """Split into an earlier training set and a later test set.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with a time column.
    test_fraction : float
        Fraction of rows (the latest) held out, in (0, 1).
    time_col : str
        Column to order by.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        Training and test sets; every training time precedes every test time.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    df = df.sort_values(time_col).reset_index(drop=True)
    cut = int(round(len(df) * (1 - test_fraction)))
    if cut < 1 or cut >= len(df):
        raise ValueError("dataset too small for this test_fraction")
    return df.iloc[:cut], df.iloc[cut:]


def rolling_origin_splits(n, n_folds=config.CV_FOLDS, min_train_fraction=config.CV_MIN_TRAIN_FRACTION):
    """Expanding-window (rolling-origin) cross-validation indices.

    The first ``min_train_fraction`` of the rows is always training data; the remainder is
    cut into ``n_folds`` consecutive test blocks, each trained on everything before it.

    Parameters
    ----------
    n : int
        Number of time-ordered rows.
    n_folds : int
        Number of test blocks.
    min_train_fraction : float
        Size of the first training window, as a fraction of ``n``.

    Returns
    -------
    list of (numpy.ndarray, numpy.ndarray)
        ``(train_idx, test_idx)`` per fold.
    """
    if n_folds < 1 or not 0 < min_train_fraction < 1:
        raise ValueError("need n_folds >= 1 and 0 < min_train_fraction < 1")
    start = int(n * min_train_fraction)
    size = (n - start) // n_folds
    if start < 2 or size < 2:
        raise ValueError("too few rows for the requested folds")
    return [(np.arange(0, start + k * size), np.arange(start + k * size, start + (k + 1) * size))
            for k in range(n_folds)]


def evaluate(model_factories, train, test, target=config.TARGET):
    """Fit each model on ``train`` and score it on ``test``, with timings.

    Parameters
    ----------
    model_factories : dict
        ``name -> callable`` returning an unfitted model.
    train, test : pandas.DataFrame
        Training and test rows.
    target : str
        Target column.

    Returns
    -------
    (pandas.DataFrame, dict)
        One row of metrics per model (plus ``fit_s`` and ``predict_s`` wall-clock seconds),
        and the fitted models by name.
    """
    rows, fitted = [], {}
    for name, make in model_factories.items():
        model = make()
        t0 = time.perf_counter()
        model.fit(train, train[target].to_numpy())
        t1 = time.perf_counter()
        pred = model.predict(test)
        t2 = time.perf_counter()
        rows.append({"model": name, "n_train": len(train), "n_test": len(test),
                     **regression_metrics(test[target], pred), "fit_s": t1 - t0, "predict_s": t2 - t1})
        fitted[name] = model
    return pd.DataFrame(rows), fitted


def cross_validate(model_factories, df, target=config.TARGET, n_folds=config.CV_FOLDS,
                   min_train_fraction=config.CV_MIN_TRAIN_FRACTION):
    """Rolling-origin cross-validation of several models.

    Parameters
    ----------
    model_factories : dict
        ``name -> callable`` returning an unfitted model.
    df : pandas.DataFrame
        Time-ordered dataset.
    target : str
        Target column.
    n_folds, min_train_fraction
        See :func:`rolling_origin_splits`.

    Returns
    -------
    pandas.DataFrame
        One row per (model, fold) with metrics.
    """
    df = df.sort_values("time_utc").reset_index(drop=True)
    rows = []
    for k, (tr, te) in enumerate(rolling_origin_splits(len(df), n_folds, min_train_fraction)):
        res, _ = evaluate(model_factories, df.iloc[tr], df.iloc[te], target)
        rows.append(res.assign(fold=k, test_from=df["time_utc"].iloc[te[0]]))
    return pd.concat(rows, ignore_index=True)
