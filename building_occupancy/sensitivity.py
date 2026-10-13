"""
One-at-a-time sensitivity analysis of the final models.

Each setting is moved away from its baseline value while every other setting stays at
baseline, and the two final models plus the time-of-day baseline are re-validated on the
resulting rows. One-at-a-time variation shows how strongly each choice drives the result,
but not interactions between choices.
"""
import pandas as pd

from . import config, evaluation, models

# Feature groups removed together in the ablation (a variable and its derivatives).
FEATURE_GROUPS = {
    "CO2 level": ["co2"],
    "CO2 rate of change": ["dco2_dt", "dco2_dt_lag1"],
    "temperature": ["temp", "dtemp_dt"],
    "humidity": ["rh", "drh_dt"],
    "VOC": ["voc"],
    "PM2.5": ["pm25"],
}
MODEL_NAMES = ("Time-of-day baseline", "Ridge (environment + time)", "Polynomial ridge (degree 2)")


def model_factories(features=models.ENV_FEATURES, alphas=config.RIDGE_ALPHAS):
    """Factories for the time-of-day baseline and the two final models.

    Parameters
    ----------
    features : list of str
        Environmental features given to the final models.
    alphas : array_like
        Ridge penalties searched (a single value fixes the penalty).

    Returns
    -------
    dict
        ``name -> callable`` returning an unfitted model, keyed by :data:`MODEL_NAMES`.
    """
    features = list(features)
    if not features:
        raise ValueError("at least one feature is required")
    return {
        MODEL_NAMES[0]: models.TimeOfDayBaseline,
        MODEL_NAMES[1]: lambda: models.RidgeRegression(features, include_time=True, alphas=alphas),
        MODEL_NAMES[2]: lambda: models.PolynomialRidgeRegression(degrees=2, features=features, alphas=alphas),
    }


def ablation_feature_sets(features=models.ENV_FEATURES, groups=FEATURE_GROUPS):
    """Feature sets for the ablation: all, each group removed, and CO2 only.

    Parameters
    ----------
    features : list of str
        Full feature list.
    groups : dict
        ``group name -> features in the group``.

    Returns
    -------
    dict
        ``label -> feature list``, starting with ``"all features"``.
    """
    sets = {"all features": list(features)}
    for name, cols in groups.items():
        sets[f"without {name}"] = [f for f in features if f not in cols]
    sets["CO2 only (level + rate)"] = [f for f in features
                                       if f in groups["CO2 level"] + groups["CO2 rate of change"]]
    return sets


def holdout_scores(df, factories, test_fraction=config.TEST_FRACTION, train_days=None, target=config.TARGET):
    """Chronological hold-out scores, optionally with a shortened training window.

    Parameters
    ----------
    df : pandas.DataFrame
        Modelling dataset.
    factories : dict
        ``name -> callable`` returning an unfitted model.
    test_fraction : float
        Fraction of the latest rows held out.
    train_days : float, optional
        Train only on the ``train_days`` days immediately before the hold-out.
    target : str
        Target column.

    Returns
    -------
    pandas.DataFrame
        Output of :func:`building_occupancy.evaluation.evaluate`.

    Raises
    ------
    ValueError
        If ``train_days`` is not positive or leaves fewer than 48 training rows.
    """
    train, test = evaluation.chronological_split(df, test_fraction)
    if train_days is not None:
        if train_days <= 0:
            raise ValueError("train_days must be positive")
        train = train[train["time_utc"] >= test["time_utc"].iloc[0] - pd.Timedelta(days=train_days)]
        if len(train) < 48:
            raise ValueError(f"only {len(train)} training rows in the last {train_days} days")
    res, _ = evaluation.evaluate(factories, train, test, target)
    return res


def scenario_scores(df, factories, cv=True, **holdout_kwargs):
    """Hold-out scores plus (optionally) rolling-origin CV summary for one scenario.

    Parameters
    ----------
    df : pandas.DataFrame
        Modelling dataset for the scenario.
    factories : dict
        ``name -> callable`` returning an unfitted model.
    cv : bool
        Also run :func:`building_occupancy.evaluation.cross_validate`.
    **holdout_kwargs
        Passed to :func:`holdout_scores`.

    Returns
    -------
    pandas.DataFrame
        One row per model: hold-out ``mae``, ``rmse``, ``bias``, ``r2``, timings and, if
        ``cv``, ``cv_r2_mean``, ``cv_r2_std``, ``cv_mae_mean``.
    """
    res = holdout_scores(df, factories, **holdout_kwargs)
    if cv:
        folds = evaluation.cross_validate(factories, df)
        summary = folds.groupby("model").agg(cv_r2_mean=("r2", "mean"), cv_r2_std=("r2", "std"),
                                             cv_mae_mean=("mae", "mean")).reset_index()
        res = res.merge(summary, on="model", how="left")
    return res
