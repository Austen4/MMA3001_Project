"""Tests for the SVD ridge path and polynomial ridge regression."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from building_occupancy import models

RNG = np.random.default_rng(1)


def test_ridge_path_matches_sklearn_for_every_alpha():
    X = RNG.normal(size=(300, 12))
    y = X @ RNG.normal(size=12) + RNG.normal(size=300)
    X, y = X - X.mean(0), y - y.mean()
    alphas = np.array([1e-4, 0.1, 10.0, 1e3])
    B = models.ridge_path_svd(X, y, alphas)
    assert B.shape == (12, 4)
    for j, a in enumerate(alphas):
        ref = Ridge(alpha=a, fit_intercept=False, solver="svd").fit(X, y).coef_
        assert np.allclose(B[:, j], ref, atol=1e-10)


def test_ridge_path_tends_to_least_squares_as_alpha_vanishes():
    X = RNG.normal(size=(200, 5))
    y = RNG.normal(size=200)
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.allclose(models.ridge_path_svd(X, y, [1e-10])[:, 0], ols, atol=1e-8)


def test_ridge_path_shrinks_monotonically():
    X = RNG.normal(size=(100, 6))
    y = RNG.normal(size=100)
    norms = np.linalg.norm(models.ridge_path_svd(X, y, np.logspace(-3, 3, 7)), axis=0)
    assert np.all(np.diff(norms) < 0)


def test_ridge_path_handles_more_terms_than_rows():
    X = RNG.normal(size=(20, 50))
    assert np.isfinite(models.ridge_path_svd(X, RNG.normal(size=20), [1.0])).all()


@pytest.mark.parametrize("alphas", [[0.0], [-1.0]])
def test_ridge_path_rejects_non_positive_alpha(alphas):
    with pytest.raises(ValueError):
        models.ridge_path_svd(np.ones((5, 2)), np.ones(5), alphas)


@pytest.mark.parametrize("k, d, expected", [(9, 1, 9), (9, 2, 54), (9, 3, 219), (9, 4, 714), (2, 2, 5)])
def test_number_of_polynomial_terms(k, d, expected):
    assert models.n_polynomial_terms(k, d) == expected


def _quadratic_data(n=1500):
    df = pd.DataFrame(RNG.normal(size=(n, len(models.ENV_FEATURES))), columns=models.ENV_FEATURES)
    df["co2"] = 450 + 40 * df["co2"]                        # realistic scale: tests pre-scaling
    z = (df["co2"] - 450) / 40
    y = 6 + 2 * z ** 2 + 1.5 * z * df["temp"] + RNG.normal(scale=0.2, size=n)
    return df, y.to_numpy()


def test_polynomial_ridge_selects_degree_two_for_quadratic_signal():
    df, y = _quadratic_data()
    model = models.PolynomialRidgeRegression(degrees=(1, 2, 3)).fit(df, y)
    assert model.degree_ == 2
    assert model.n_terms_ == models.n_polynomial_terms(len(models.ENV_FEATURES), 2)
    r2 = 1 - ((model.predict(df) - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert r2 > 0.97
    assert abs(model.coef_).idxmax() == "co2^2"


def test_degree_one_polynomial_ridge_is_linear_ridge():
    df, y = _quadratic_data(600)
    poly = models.PolynomialRidgeRegression(degrees=1).fit(df, y)
    lin = models.RidgeRegression().fit(df, y)
    # Same features, preprocessing and penalty grid; selection criteria differ slightly
    # (clipped vs unclipped inner predictions), so compare predictions loosely.
    assert np.corrcoef(poly.predict(df), lin.predict(df))[0, 1] > 0.999


def test_polynomial_ridge_imputes_missing_and_never_predicts_negative():
    df, y = _quadratic_data(600)
    df.loc[::7, "pm25"] = np.nan
    pred = models.PolynomialRidgeRegression(degrees=2).fit(df, y - 8).predict(df)
    assert np.isfinite(pred).all() and pred.min() == 0


@pytest.mark.parametrize("degrees", [0, (1, 2.5), ()])
def test_polynomial_ridge_rejects_bad_degrees(degrees):
    with pytest.raises(ValueError):
        models.PolynomialRidgeRegression(degrees=degrees)


def test_polynomial_ridge_missing_feature():
    with pytest.raises(ValueError):
        models.PolynomialRidgeRegression(degrees=1).fit(pd.DataFrame({"co2": [1.0] * 10}), np.ones(10))
