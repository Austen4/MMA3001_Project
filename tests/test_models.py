"""Tests for building_occupancy.models, including a manufactured-solution check."""
import numpy as np
import pandas as pd
import pytest

from building_occupancy import models

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- solvers

def test_triangular_solvers_match_numpy():
    A = RNG.normal(size=(6, 6)) + 6 * np.eye(6)
    b = RNG.normal(size=6)
    U, L = np.triu(A), np.tril(A)
    assert np.allclose(models.back_substitution(U, b), np.linalg.solve(U, b))
    assert np.allclose(models.forward_substitution(L, b), np.linalg.solve(L, b))


@pytest.mark.parametrize("solver", [models.solve_lstsq_qr, models.solve_lstsq_normal])
def test_least_squares_matches_lapack(solver):
    X = np.column_stack([np.ones(200), RNG.normal(size=(200, 4))])
    y = X @ np.array([1.0, -2.0, 0.5, 3.0, 0.0]) + RNG.normal(scale=0.1, size=200)
    assert np.allclose(solver(X, y), np.linalg.lstsq(X, y, rcond=None)[0], atol=1e-10)


def test_rank_deficient_design_raises():
    x = RNG.normal(size=50)
    X = np.column_stack([np.ones(50), x, 2 * x])
    with pytest.raises(np.linalg.LinAlgError):
        models.solve_lstsq_qr(X, x)


def test_lagged_co2_with_its_derivative_is_detected_as_collinear():
    # co2[t-1] = co2[t] - dt * dco2_dt[t] exactly, so these three columns are dependent.
    co2 = 450 + np.cumsum(RNG.normal(size=100))
    dt = 0.5
    X = np.column_stack([np.ones(99), co2[1:], np.diff(co2) / dt, co2[:-1]])
    with pytest.raises(np.linalg.LinAlgError):
        models.solve_lstsq_qr(X, RNG.normal(size=99))


def test_qr_more_accurate_than_normal_equations_when_ill_conditioned():
    # Polynomial in a variable far from zero: cond(X) ~ 1e7, cond(X^T X) ~ 1e14.
    x = np.linspace(1000, 1010, 400)
    X = np.column_stack([np.ones_like(x), x - 1000, (x - 1000) ** 2, (x - 1005) ** 3 * 1e-3])
    X[:, 1:] += 1000
    beta_true = np.array([1.0, 2.0, -1.0, 0.5])
    y = X @ beta_true
    err_qr = np.linalg.norm(models.solve_lstsq_qr(X, y) - beta_true)
    err_ne = np.linalg.norm(models.solve_lstsq_normal(X, y) - beta_true)
    assert np.linalg.cond(X) > 1e5
    assert err_qr < err_ne


@pytest.mark.parametrize("X, y", [(np.ones((3, 2)), np.ones(4)), (np.ones((1, 2)), np.ones(1)),
                                  (np.array([[1.0, np.nan], [1.0, 2.0], [1.0, 3.0]]), np.ones(3))])
def test_least_squares_input_validation(X, y):
    with pytest.raises(ValueError):
        models.solve_lstsq_qr(X, y)


# ---------------------------------------------------------------- manufactured solution

def simulate_co2(volume, flow, c_out, gen_m3_h, occupants, step_h):
    """Exact solution of V dC/dt = N G - Q (C - C_out) for piecewise-constant N."""
    c = np.empty(len(occupants) + 1)
    c[0] = c_out
    decay = np.exp(-flow * step_h / volume)
    for i, n in enumerate(occupants):
        c_eq = c_out + n * gen_m3_h * 1e6 / flow
        c[i + 1] = c_eq + (c[i] - c_eq) * decay
    return c


def test_mass_balance_recovers_known_room_physics():
    volume, flow, c_out, gen = 600.0, 600.0, 420.0, 0.0187   # 1 air change per hour
    step_h = 1 / 60
    blocks = RNG.integers(0, 25, size=200)                    # occupancy changes every 30 min
    n = np.repeat(blocks, 30).astype(float)
    c = simulate_co2(volume, flow, c_out, gen, n, step_h)
    df = pd.DataFrame({"co2": c[1:], "dco2_dt": np.diff(c) / step_h})
    model = models.MassBalanceRegression().fit(df, n)
    phys = model.implied_physics(gen)
    # Backward difference + end-of-step CO2 bias is ~ step / (2 * time constant) < 1%.
    assert phys["volume_m3"] == pytest.approx(volume, rel=0.02)
    assert phys["ventilation_m3_per_h"] == pytest.approx(flow, rel=0.02)
    assert phys["supply_co2_ppm"] == pytest.approx(c_out, rel=0.01)
    assert np.mean(np.abs(model.predict(df) - n)) < 0.5


# ---------------------------------------------------------------- models

def _calendar(n):
    return pd.DataFrame({"hour_local": np.tile(np.arange(24) + 0.5, n // 24 + 1)[:n],
                         "weekend": np.zeros(n, bool)})


def test_time_of_day_baseline_cell_means_and_fallback():
    df = _calendar(48)
    y = df["hour_local"].to_numpy()
    model = models.TimeOfDayBaseline().fit(df, y)
    assert np.allclose(model.predict(df), y)
    unseen = pd.DataFrame({"hour_local": [3.5], "weekend": [True]})
    assert model.predict(unseen)[0] == pytest.approx(y.mean())


def test_mean_baseline():
    assert np.all(models.MeanBaseline().fit(None, [1, 2, 3]).predict(range(4)) == 2)


def test_ridge_fits_linear_signal_and_never_predicts_negative():
    n = 600
    df = pd.DataFrame(RNG.normal(size=(n, len(models.ENV_FEATURES))), columns=models.ENV_FEATURES)
    df.iloc[::50, 4] = np.nan                                  # imputation path
    y = 5 + 2 * df["co2"].to_numpy() + 1.5 * df["dco2_dt"].to_numpy() + RNG.normal(scale=0.1, size=n)
    model = models.RidgeRegression().fit(df, y)
    pred = model.predict(df)
    assert model.alpha_ in model.alphas
    assert np.corrcoef(pred, np.clip(y, 0, None))[0, 1] > 0.95
    assert (pred >= 0).all()


def test_ridge_with_time_features_needs_calendar():
    df = pd.DataFrame(RNG.normal(size=(100, len(models.ENV_FEATURES))), columns=models.ENV_FEATURES)
    with pytest.raises(KeyError):
        models.RidgeRegression(include_time=True).fit(df, np.ones(100))


def test_linear_model_missing_feature():
    with pytest.raises(ValueError):
        models.MassBalanceRegression().fit(pd.DataFrame({"co2": [1.0, 2.0, 3.0]}), [1, 2, 3])


def test_ols_predictions_clipped_at_zero():
    df = pd.DataFrame({"x": np.arange(10.0)})
    model = models.LinearLeastSquares(["x"]).fit(df, np.arange(10.0) - 5)
    assert model.predict(df).min() == 0
