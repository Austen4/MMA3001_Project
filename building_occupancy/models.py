"""
Linear models for building occupancy and the baselines they are compared against.

All models share one interface::

    model.fit(df, y)        # df: modelling dataset rows, y: target values
    model.predict(df)       # -> numpy array, clipped at 0 (headcount cannot be negative)

Linear numerical method
-----------------------
:class:`MassBalanceRegression` fits the well-mixed-zone CO2 mass balance

    V dC/dt = N G - Q (C - C_out)    =>    N = a dC/dt + b C + c

(V zone volume, Q ventilation flow, G CO2 generation per person, C_out supply CO2) by
linear least squares, solved with our own QR-decomposition + back-substitution solver
(:func:`solve_lstsq_qr`). :func:`solve_lstsq_normal` (normal equations + Cholesky) is
provided for the numerical stability comparison.

Machine learning
----------------
:class:`RidgeRegression` is L2-regularised linear regression on all environmental features,
with the penalty chosen by time-ordered cross-validation (scikit-learn).

:class:`PolynomialRidgeRegression` extends it to polynomial features of degree 1-4. Because
the number of terms grows combinatorially, it computes the whole regularisation path from a
single SVD per fold (:func:`ridge_path_svd`) instead of refitting for every penalty.
"""
from math import comb

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from . import config

MASS_BALANCE_FEATURES = ["dco2_dt", "co2"]
ENV_FEATURES = ["co2", "dco2_dt", "dco2_dt_lag1", "temp", "dtemp_dt",
                "rh", "drh_dt", "voc", "pm25"]


# ---------------------------------------------------------------- numerical solvers

# Relative pivot size below which a triangular factor is treated as singular. For the R
# factor of a QR decomposition this means cond(X) > ~1e10: coefficients would keep fewer
# than ~6 significant digits in double precision, so the design is effectively
# rank-deficient (exact collinearities leave pivots of ~1e-14 from rounding alone).
SINGULAR_RTOL = 1e-10


def back_substitution(R, b, rtol=SINGULAR_RTOL):
    """Solve ``R x = b`` for upper-triangular ``R``.

    Parameters
    ----------
    R : (n, n) array_like
        Upper-triangular matrix.
    b : (n,) array_like
        Right-hand side.
    rtol : float
        Singularity tolerance relative to the largest diagonal entry.

    Returns
    -------
    numpy.ndarray
        Solution ``x``.

    Raises
    ------
    numpy.linalg.LinAlgError
        If ``R`` is numerically singular (a diagonal entry is below
        ``rtol * max|diag(R)|``), i.e. the design matrix is (nearly) rank deficient.
    """
    R, b = np.asarray(R, float), np.asarray(b, float)
    n = R.shape[0]
    diag = np.abs(np.diag(R))
    if diag.min() <= rtol * diag.max():
        raise np.linalg.LinAlgError("matrix is numerically singular (rank-deficient design)")
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - R[i, i + 1:] @ x[i + 1:]) / R[i, i]
    return x


def forward_substitution(L, b, rtol=SINGULAR_RTOL):
    """Solve ``L x = b`` for lower-triangular ``L``.

    Parameters
    ----------
    L : (n, n) array_like
        Lower-triangular matrix.
    b : (n,) array_like
        Right-hand side.
    rtol : float
        Singularity tolerance relative to the largest diagonal entry.

    Returns
    -------
    numpy.ndarray
        Solution ``x``.

    Raises
    ------
    numpy.linalg.LinAlgError
        If ``L`` is numerically singular.
    """
    L, b = np.asarray(L, float), np.asarray(b, float)
    n = L.shape[0]
    diag = np.abs(np.diag(L))
    if diag.min() <= rtol * diag.max():
        raise np.linalg.LinAlgError("matrix is numerically singular")
    x = np.zeros(n)
    for i in range(n):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]
    return x


def solve_lstsq_qr(X, y):
    """Least-squares solution of ``X beta ~= y`` via Householder QR.

    ``X = Q R`` (reduced), so the normal equations reduce to ``R beta = Q^T y``. This
    works with ``cond(X)`` rather than ``cond(X)^2``, so it is the stable choice.

    Parameters
    ----------
    X : (m, n) array_like
        Design matrix, ``m >= n``, full column rank.
    y : (m,) array_like
        Observations.

    Returns
    -------
    numpy.ndarray
        Coefficients ``beta`` of length ``n``.
    """
    X, y = _check_design(X, y)
    Q, R = np.linalg.qr(X, mode="reduced")
    return back_substitution(R, Q.T @ y)


def solve_lstsq_normal(X, y):
    """Least-squares solution via the normal equations ``X^T X beta = X^T y``.

    Solved with a Cholesky factorisation ``X^T X = L L^T`` and two triangular solves.
    Cheaper than QR but squares the condition number, so it loses accuracy when the
    columns of ``X`` are nearly collinear. Provided for comparison only.

    Parameters
    ----------
    X : (m, n) array_like
        Design matrix, full column rank.
    y : (m,) array_like
        Observations.

    Returns
    -------
    numpy.ndarray
        Coefficients ``beta``.
    """
    X, y = _check_design(X, y)
    L = np.linalg.cholesky(X.T @ X)
    return back_substitution(L.T, forward_substitution(L, X.T @ y))


def ridge_path_svd(X, y, alphas):
    """Ridge coefficients for many penalties from a single thin SVD.

    With ``X = U S V^T``, the ridge solution for penalty ``alpha`` is
    ``beta = V diag(s / (s^2 + alpha)) U^T y``. One SVD (~``6 m n^2 + 20 n^3`` flops,
    Golub & Van Loan, *Matrix Computations*, 4th ed., sec. 8.6) therefore gives every
    penalty, instead of one factorisation per penalty. Working from the SVD of ``X``
    rather than ``X^T X`` also avoids squaring the condition number.

    ``X`` and ``y`` should be centred; no intercept is fitted.

    Parameters
    ----------
    X : (m, n) array_like
        Design matrix (any shape; ``n > m`` is allowed).
    y : (m,) array_like
        Observations.
    alphas : array_like
        Penalties, all > 0.

    Returns
    -------
    numpy.ndarray
        Coefficients, shape ``(n, len(alphas))``.
    """
    X, y = np.asarray(X, float), np.asarray(y, float)
    alphas = np.atleast_1d(np.asarray(alphas, float))
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError(f"incompatible shapes X{X.shape}, y{y.shape}")
    if np.any(alphas <= 0):
        raise ValueError("ridge penalties must be positive")
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError("X and y must not contain NaN or inf")
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    shrink = s[:, None] / (s[:, None] ** 2 + alphas[None, :])   # (k, n_alphas)
    return Vt.T @ (shrink * (U.T @ y)[:, None])


def n_polynomial_terms(n_features, degree):
    """Number of polynomial terms of degree 1..``degree`` in ``n_features`` variables.

    Parameters
    ----------
    n_features, degree : int

    Returns
    -------
    int
        ``C(n_features + degree, degree) - 1`` (all monomials except the constant).
    """
    return comb(n_features + degree, degree) - 1


def _check_design(X, y):
    """Validate shapes and finiteness of a least-squares problem."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError(f"incompatible shapes X{X.shape}, y{y.shape}")
    if X.shape[0] < X.shape[1]:
        raise ValueError("underdetermined system: fewer observations than coefficients")
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError("X and y must not contain NaN or inf")
    return X, y


# ---------------------------------------------------------------- feature helpers

def time_features(df, harmonics=2):
    """Smooth time-of-day features: sine/cosine pairs, a weekend flag and their interaction.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``hour_local`` (0-24) and ``weekend`` (bool).
    harmonics : int
        Number of sine/cosine pairs.

    Returns
    -------
    pandas.DataFrame
        ``2 * harmonics * 2 + 1`` columns aligned with ``df``.
    """
    ang = 2 * np.pi * df["hour_local"].to_numpy(float) / 24
    we = df["weekend"].to_numpy(float)
    out = {"weekend": we}
    for k in range(1, harmonics + 1):
        for name, f in (("sin", np.sin), ("cos", np.cos)):
            v = f(k * ang)
            out[f"{name}{k}"] = v
            out[f"weekend:{name}{k}"] = v * we
    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------- baselines

class MeanBaseline:
    """Predicts the training-set mean for every row (no inputs)."""

    name = "Mean baseline"

    def fit(self, df, y):
        """Store the mean of ``y``. Returns ``self``."""
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, df):
        """Return the stored mean for every row of ``df``."""
        return np.full(len(df), self.mean_)


class TimeOfDayBaseline:
    """Mean target for each (hour of day, weekday/weekend) cell; uses no sensor data.

    This is the bar an environmental model must beat to show the sensors add information
    beyond the building's routine.
    """

    name = "Time-of-day baseline"

    def fit(self, df, y):
        """Average ``y`` within each hour x day-type cell. Returns ``self``."""
        key = self._key(df)
        self.cells_ = pd.Series(np.asarray(y, float), index=key).groupby(level=[0, 1]).mean()
        self.fallback_ = float(np.mean(y))
        return self

    def predict(self, df):
        """Cell mean for each row; overall training mean for cells never seen."""
        return self.cells_.reindex(self._key(df)).fillna(self.fallback_).to_numpy()

    @staticmethod
    def _key(df):
        return pd.MultiIndex.from_arrays([np.floor(df["hour_local"]).astype(int), df["weekend"].astype(bool)])


# ---------------------------------------------------------------- linear models

class LinearLeastSquares:
    """Ordinary least squares with intercept, solved by :func:`solve_lstsq_qr`.

    Parameters
    ----------
    features : list of str
        Columns of the dataset used as regressors. Rows must not contain NaN.
    solver : {"qr", "normal"}
        Least-squares solver.
    name : str, optional
        Display name.

    Attributes
    ----------
    coef_ : pandas.Series
        Fitted coefficients, including ``intercept``.
    condition_number_ : float
        2-norm condition number of the training design matrix.
    """

    def __init__(self, features, solver="qr", name=None):
        if solver not in ("qr", "normal"):
            raise ValueError("solver must be 'qr' or 'normal'")
        self.features = list(features)
        self.solver = solver
        self.name = name or f"OLS ({', '.join(self.features)})"

    def _design(self, df):
        missing = set(self.features) - set(df.columns)
        if missing:
            raise ValueError(f"dataset is missing features: {sorted(missing)}")
        X = df[self.features].to_numpy(float)
        return np.column_stack([np.ones(len(X)), X])

    def fit(self, df, y):
        """Fit coefficients by least squares. Returns ``self``."""
        X = self._design(df)
        solve = solve_lstsq_qr if self.solver == "qr" else solve_lstsq_normal
        beta = solve(X, np.asarray(y, float))
        self.coef_ = pd.Series(beta, index=["intercept", *self.features])
        self.condition_number_ = float(np.linalg.cond(X))
        return self

    def predict(self, df):
        """Linear prediction, clipped at zero."""
        return np.clip(self._design(df) @ self.coef_.to_numpy(), 0, None)


class MassBalanceRegression(LinearLeastSquares):
    """Occupancy from the CO2 mass balance: ``N = a dC/dt + b C + c``.

    Derived from ``V dC/dt = N G - Q (C - C_out)`` for a well-mixed zone with
    ``C`` in ppm and ``dC/dt`` in ppm/h, giving ``a = V / (G 1e6)``,
    ``b = Q / (G 1e6)`` and ``c = -b C_out``.
    """

    def __init__(self, solver="qr"):
        super().__init__(MASS_BALANCE_FEATURES, solver=solver, name="CO2 mass balance (OLS)")

    def implied_physics(self, co2_generation_m3_per_h=config.CO2_GENERATION_M3_PER_H):
        """Physical quantities implied by the fitted coefficients.

        Parameters
        ----------
        co2_generation_m3_per_h : float
            CO2 generation per person (m3/h).

        Returns
        -------
        dict
            ``volume_m3`` (V), ``ventilation_m3_per_h`` (Q), ``air_changes_per_h`` (Q/V)
            and ``supply_co2_ppm`` (C_out). A physically consistent fit has V, Q > 0 and
            C_out near outdoor CO2 (~420 ppm).
        """
        a, b, c = self.coef_["dco2_dt"], self.coef_["co2"], self.coef_["intercept"]
        g = co2_generation_m3_per_h * 1e6
        volume, flow = a * g, b * g
        return {
            "volume_m3": volume,
            "ventilation_m3_per_h": flow,
            "air_changes_per_h": flow / volume if volume else np.nan,
            "supply_co2_ppm": -c / b if b else np.nan,
        }


class RidgeRegression:
    """L2-regularised linear regression with time-ordered selection of the penalty.

    Pipeline: median imputation -> standardisation -> ridge. The penalty ``alpha`` is
    chosen from ``alphas`` by :class:`sklearn.model_selection.TimeSeriesSplit` cross
    validation on the training data only (minimising mean absolute error).

    Parameters
    ----------
    features : list of str
        Environmental feature columns (may contain NaN).
    include_time : bool
        Also use :func:`time_features`.
    alphas : array_like
        Candidate penalties.
    inner_splits : int
        Number of time-series splits for choosing ``alpha``.
    name : str, optional
        Display name.

    Attributes
    ----------
    alpha_ : float
        Selected penalty.
    coef_ : pandas.Series
        Coefficients on the standardised features (comparable magnitudes).
    """

    def __init__(self, features=ENV_FEATURES, include_time=False, alphas=config.RIDGE_ALPHAS,
                 inner_splits=config.RIDGE_INNER_SPLITS, name=None):
        self.features = list(features)
        self.include_time = include_time
        self.alphas = np.asarray(alphas, float)
        self.inner_splits = inner_splits
        self.name = name or ("Ridge (environment + time)" if include_time else "Ridge (environment)")

    def _design(self, df):
        missing = set(self.features) - set(df.columns)
        if missing:
            raise ValueError(f"dataset is missing features: {sorted(missing)}")
        X = df[self.features].astype(float)
        if self.include_time:
            X = pd.concat([X, time_features(df)], axis=1)
        return X

    def fit(self, df, y):
        """Choose ``alpha`` by time-series CV, then refit on all of ``df``. Returns ``self``."""
        X = self._design(df)
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge())
        search = GridSearchCV(pipe, {"ridge__alpha": self.alphas},
                              cv=TimeSeriesSplit(n_splits=self.inner_splits),
                              scoring="neg_mean_absolute_error")
        search.fit(X, np.asarray(y, float))
        self.model_ = search.best_estimator_
        self.alpha_ = float(search.best_params_["ridge__alpha"])
        self.coef_ = pd.Series(self.model_[-1].coef_, index=X.columns)
        return self

    def predict(self, df):
        """Ridge prediction, clipped at zero."""
        return np.clip(self.model_.predict(self._design(df)), 0, None)


class _PolynomialDesign:
    """Training-set preprocessing for polynomial ridge: impute, scale, expand, scale.

    Inputs are standardised *before* expansion so powers of large raw values (CO2 ~ 450 ppm)
    do not make the design matrix ill-conditioned, and the expanded terms are standardised
    again so the ridge penalty treats every term equally. All statistics come from the
    training rows only.
    """

    def __init__(self, degree):
        self.degree = degree

    def fit_transform(self, raw):
        """Learn preprocessing from ``raw`` (n x features array) and return the design."""
        self.median_ = np.nanmedian(raw, axis=0)
        filled = np.where(np.isnan(raw), self.median_, raw)
        self.mean_, self.std_ = filled.mean(axis=0), _safe_std(filled)
        self.poly_ = PolynomialFeatures(self.degree, include_bias=False)
        expanded = self.poly_.fit_transform((filled - self.mean_) / self.std_)
        self.emean_, self.estd_ = expanded.mean(axis=0), _safe_std(expanded)
        return (expanded - self.emean_) / self.estd_

    def transform(self, raw):
        """Apply the learned preprocessing to new rows."""
        filled = np.where(np.isnan(raw), self.median_, raw)
        expanded = self.poly_.transform((filled - self.mean_) / self.std_)
        return (expanded - self.emean_) / self.estd_


def _safe_std(a):
    """Column standard deviations with zeros replaced by 1 (constant columns)."""
    s = a.std(axis=0)
    return np.where(s > 0, s, 1.0)


class PolynomialRidgeRegression:
    """Ridge regression on polynomial expansions of the environmental features.

    Degree and penalty are chosen jointly by time-ordered cross-validation on the
    training data (minimum mean absolute error), using :func:`ridge_path_svd` so each
    candidate degree needs one SVD per fold regardless of how many penalties are tried.

    Parameters
    ----------
    degrees : int or sequence of int
        Candidate polynomial degrees (an int fixes the degree).
    features : list of str
        Environmental feature columns (NaN allowed; median-imputed).
    alphas : array_like
        Candidate penalties (> 0).
    inner_splits : int
        Number of :class:`~sklearn.model_selection.TimeSeriesSplit` folds.
    name : str, optional
        Display name.

    Attributes
    ----------
    degree_, alpha_ : int, float
        Selected degree and penalty.
    n_terms_ : int
        Number of polynomial terms in the fitted model.
    cv_mae_ : pandas.DataFrame
        Inner-CV mean absolute error for every (degree, alpha).
    coef_ : pandas.Series
        Coefficients on the standardised polynomial terms.
    """

    def __init__(self, degrees=config.POLY_DEGREES, features=ENV_FEATURES, alphas=config.RIDGE_ALPHAS,
                 inner_splits=config.RIDGE_INNER_SPLITS, name=None):
        self.degrees = (degrees,) if np.isscalar(degrees) else tuple(degrees)
        if not self.degrees or any(int(d) != d or d < 1 for d in self.degrees):
            raise ValueError("degrees must be positive integers")
        self.features = list(features)
        self.alphas = np.asarray(alphas, float)
        self.inner_splits = inner_splits
        if name is None:
            name = (f"Polynomial ridge (degree {self.degrees[0]})" if len(self.degrees) == 1
                    else "Polynomial ridge (degree selected by CV)")
        self.name = name

    def _raw(self, df):
        missing = set(self.features) - set(df.columns)
        if missing:
            raise ValueError(f"dataset is missing features: {sorted(missing)}")
        return df[self.features].to_numpy(float)

    def fit(self, df, y):
        """Select degree and penalty by inner CV, then refit on all of ``df``. Returns ``self``."""
        raw, y = self._raw(df), np.asarray(y, float)
        rows = []
        for degree in self.degrees:
            fold_mae = []
            for tr, va in TimeSeriesSplit(n_splits=self.inner_splits).split(raw):
                design = _PolynomialDesign(degree)
                X_tr = design.fit_transform(raw[tr])
                y_mean = y[tr].mean()
                B = ridge_path_svd(X_tr, y[tr] - y_mean, self.alphas)
                pred = np.clip(design.transform(raw[va]) @ B + y_mean, 0, None)
                fold_mae.append(np.abs(pred - y[va][:, None]).mean(axis=0))
            rows += [{"degree": degree, "alpha": a, "mae": m}
                     for a, m in zip(self.alphas, np.mean(fold_mae, axis=0))]
        self.cv_mae_ = pd.DataFrame(rows)
        best = self.cv_mae_.loc[self.cv_mae_["mae"].idxmin()]
        self.degree_, self.alpha_ = int(best["degree"]), float(best["alpha"])

        self.design_ = _PolynomialDesign(self.degree_)
        X = self.design_.fit_transform(raw)
        self.y_mean_ = y.mean()
        beta = ridge_path_svd(X, y - self.y_mean_, [self.alpha_])[:, 0]
        self.n_terms_ = X.shape[1]
        self.coef_ = pd.Series(beta, index=self.design_.poly_.get_feature_names_out(self.features))
        return self

    def predict(self, df):
        """Polynomial ridge prediction, clipped at zero."""
        X = self.design_.transform(self._raw(df))
        return np.clip(X @ self.coef_.to_numpy() + self.y_mean_, 0, None)


def default_models():
    """Factories for the models compared in the experiments.

    Returns
    -------
    dict
        ``name -> callable`` returning a fresh, unfitted model.
    """
    return {
        "Mean baseline": MeanBaseline,
        "Time-of-day baseline": TimeOfDayBaseline,
        "CO2 mass balance (OLS)": MassBalanceRegression,
        "Ridge (environment)": lambda: RidgeRegression(include_time=False),
        "Ridge (environment + time)": lambda: RidgeRegression(include_time=True),
    }
