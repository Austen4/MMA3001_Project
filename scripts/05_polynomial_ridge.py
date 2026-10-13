"""
Step 5 - polynomial ridge regression: degree sweep, conditioning and solver benchmark.

For every environmental sensor and bin width:

* Degree sweep (sensitivity analysis): polynomial ridge of each degree in
  config.POLY_DEGREES, penalty chosen by inner time-series CV; scored on the training data,
  the chronological hold-out and rolling-origin CV.
* Comparison: polynomial ridge with the degree chosen by inner CV on the training data
  only, against the time-of-day baseline and linear ridge on the same hold-out.

For the 60-minute datasets:

* Conditioning of the polynomial design with and without standardising inputs.
* Benchmark of the penalty search: one SVD per fold (``ridge_path_svd``) against one
  Cholesky-based ridge fit per penalty, with timings and approximate flop counts.

Outputs go to results/polynomial_ridge/.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402
from sklearn.preprocessing import PolynomialFeatures  # noqa: E402

from building_occupancy import config, evaluation, models, plotting  # noqa: E402  (sets Agg backend)
from building_occupancy.plotting import INK, INK_MUTED, SURFACE  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OUT = config.RESULTS_DIR / "polynomial_ridge"
PLOT_SENSOR, PLOT_BIN = 6012002000869, 60   # same configuration as the linear-model figure
CV_COLOR, HOLDOUT_COLOR = plotting.CATEGORICAL[:2]
Y_FLOOR = -1.0   # lowest R2 drawn in the degree-sweep figure


def load(sensor, m):
    """Load one modelling dataset."""
    return pd.read_csv(config.PROCESSED_DIR / f"dataset_{sensor}_{m}min.csv", parse_dates=["time_utc"])


def degree_sweep(df, tag):
    """Train, hold-out and rolling-CV scores for each fixed degree.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, dict)
        Per-degree hold-out/train results, per-fold CV results, the hold-out rows, and
        hold-out predictions keyed ``"Degree d (n terms)"``.
    """
    train, test = evaluation.chronological_split(df)
    factories = {d: (lambda d=d: models.PolynomialRidgeRegression(degrees=d)) for d in config.POLY_DEGREES}
    res, fitted = evaluation.evaluate(factories, train, test)
    res = res.rename(columns={"model": "degree"})
    predictions = {}
    for i, d in enumerate(res["degree"]):
        m = fitted[d]
        train_scores = evaluation.regression_metrics(train[config.TARGET], m.predict(train))
        res.loc[i, ["train_r2", "train_mae"]] = train_scores["r2"], train_scores["mae"]
        res.loc[i, ["alpha", "n_terms"]] = m.alpha_, m.n_terms_
        predictions[f"Degree {d} ({m.n_terms_} terms)"] = m.predict(test)
    cv = evaluation.cross_validate(factories, df).rename(columns={"model": "degree"})
    return res.assign(**tag), cv.assign(**tag), test, predictions


def comparison(df, tag):
    """Hold-out comparison of the CV-selected polynomial model against reference models."""
    train, test = evaluation.chronological_split(df)
    factories = {
        "Time-of-day baseline": models.TimeOfDayBaseline,
        "Ridge (environment)": lambda: models.RidgeRegression(),
        "Polynomial ridge (degree selected by CV)": lambda: models.PolynomialRidgeRegression(),
    }
    res, fitted = evaluation.evaluate(factories, train, test)
    sel = fitted["Polynomial ridge (degree selected by CV)"]
    res["selected_degree"] = np.where(res["model"].str.startswith("Polynomial"), sel.degree_, np.nan)
    res["selected_alpha"] = np.where(res["model"].str.startswith("Polynomial"), sel.alpha_, np.nan)
    return res.assign(**tag)


def conditioning(df, tag):
    """Condition number of the polynomial design under different preprocessing.

    ``raw``: monomials of the raw inputs; ``scaled inputs``: inputs standardised before
    expansion; ``scaled inputs + terms``: as used by the model.
    """
    train, _ = evaluation.chronological_split(df)
    raw = train[models.ENV_FEATURES].to_numpy(float)
    raw = np.where(np.isnan(raw), np.nanmedian(raw, axis=0), raw)
    rows = []
    for d in config.POLY_DEGREES:
        poly = PolynomialFeatures(d, include_bias=True)
        z = (raw - raw.mean(0)) / raw.std(0)
        design = models._PolynomialDesign(d)
        rows.append({**tag, "degree": d, "n_terms": models.n_polynomial_terms(raw.shape[1], d),
                     "cond_raw": np.linalg.cond(poly.fit_transform(raw)),
                     "cond_scaled_inputs": np.linalg.cond(poly.fit_transform(z)),
                     "cond_scaled_inputs_and_terms": np.linalg.cond(design.fit_transform(train[models.ENV_FEATURES]
                                                                                       .to_numpy(float)))})
    return rows


def benchmark(df, tag, repeats=3):
    """Time the inner-CV penalty search: SVD path vs one Cholesky ridge fit per penalty.

    Both use identical design matrices and folds; the table also reports the largest
    difference between the two methods' CV error curves and approximate flop counts
    (thin SVD ~ 6 m n^2 + 20 n^3; ridge via normal equations ~ m n^2 + n^3 / 3 per penalty).
    """
    train, _ = evaluation.chronological_split(df)
    raw = train[models.ENV_FEATURES].to_numpy(float)
    y = train[config.TARGET].to_numpy(float)
    alphas = config.RIDGE_ALPHAS
    rows = []
    for d in config.POLY_DEGREES:
        folds = []
        for tr, va in TimeSeriesSplit(n_splits=config.RIDGE_INNER_SPLITS).split(raw):
            design = models._PolynomialDesign(d)
            folds.append((design.fit_transform(raw[tr]), y[tr] - y[tr].mean(), design.transform(raw[va]),
                          y[va], y[tr].mean()))

        def path():
            return np.array([np.abs(Xv @ models.ridge_path_svd(Xt, yt, alphas) + ym - yv[:, None]).mean(0)
                             for Xt, yt, Xv, yv, ym in folds])

        def loop():
            return np.array([[np.abs(Ridge(alpha=a, fit_intercept=False, solver="cholesky").fit(Xt, yt)
                                     .predict(Xv) + ym - yv).mean() for a in alphas]
                             for Xt, yt, Xv, yv, ym in folds])

        timings = {}
        for name, fn in (("svd_path", path), ("per_alpha_cholesky", loop)):
            t0 = time.perf_counter()
            for _ in range(repeats):
                mae = fn()
            timings[name] = ((time.perf_counter() - t0) / repeats, mae)
        mn = [(Xt.shape[0], Xt.shape[1]) for Xt, *_ in folds]
        flops_path = sum(6 * m * n ** 2 + 20 * n ** 3 for m, n in mn)
        flops_loop = len(alphas) * sum(m * n ** 2 + n ** 3 / 3 for m, n in mn)
        rows.append({**tag, "degree": d, "n_terms": mn[0][1], "n_alphas": len(alphas),
                     "svd_path_s": timings["svd_path"][0], "per_alpha_s": timings["per_alpha_cholesky"][0],
                     "speedup": timings["per_alpha_cholesky"][0] / timings["svd_path"][0],
                     "svd_path_gflop": flops_path / 1e9, "per_alpha_gflop": flops_loop / 1e9,
                     "svd_path_gflop_per_s": flops_path / 1e9 / timings["svd_path"][0],
                     "per_alpha_gflop_per_s": flops_loop / 1e9 / timings["per_alpha_cholesky"][0],
                     "max_cv_mae_difference": float(np.abs(timings["svd_path"][1]
                                                           - timings["per_alpha_cholesky"][1]).max())})
    return rows


def main():
    """Run all polynomial-ridge experiments and write results."""
    OUT.mkdir(parents=True, exist_ok=True)
    sweep, cv, comp, cond, bench = [], [], [], [], []
    plot_test, plot_preds, plot_selected = None, {}, None
    for sensor in config.ENV_SENSORS:
        for m in config.BIN_MINUTES:
            t0 = time.perf_counter()
            df, tag = load(sensor, m), {"sensor": sensor, "bin_minutes": m}
            s, c, test, preds = degree_sweep(df, tag)
            sweep.append(s)
            cv.append(c)
            comp.append(comparison(df, tag))
            if (sensor, m) == (PLOT_SENSOR, PLOT_BIN):
                plot_test, plot_preds = test, preds
                plot_selected = int(comp[-1]["selected_degree"].dropna().iloc[0])
            if m == 60:
                cond += conditioning(df, tag)
                bench += benchmark(df, tag)
            print(f"sensor {sensor}, {m:>3} min: done ({time.perf_counter() - t0:.0f} s)")

    sweep, cv, comp = (pd.concat(x, ignore_index=True) for x in (sweep, cv, comp))
    cv_summary = (cv.groupby(["sensor", "bin_minutes", "degree"])[["mae", "r2"]].agg(["mean", "std"]))
    cv_summary.columns = ["_".join(c) for c in cv_summary.columns]
    sweep.to_csv(OUT / "degree_sweep_holdout.csv", index=False)
    cv.to_csv(OUT / "degree_sweep_cv_folds.csv", index=False)
    cv_summary.round(4).to_csv(OUT / "degree_sweep_cv_summary.csv")
    comp.to_csv(OUT / "comparison_holdout.csv", index=False)
    pd.DataFrame(cond).to_csv(OUT / "conditioning.csv", index=False)
    pd.DataFrame(bench).to_csv(OUT / "solver_benchmark.csv", index=False)

    print("\nDegree sweep (train R2 / hold-out R2 / CV mean R2):")
    table = sweep.merge(cv_summary.reset_index(), on=["sensor", "bin_minutes", "degree"])
    print(table[["sensor", "bin_minutes", "degree", "n_terms", "alpha", "train_r2", "r2", "r2_mean", "r2_std",
                 "fit_s"]].round(3).to_string(index=False))
    print("\nHold-out comparison:")
    print(comp[["sensor", "bin_minutes", "model", "mae", "r2", "selected_degree", "fit_s"]]
          .round(3).to_string(index=False))
    print("\nConditioning:")
    print(pd.DataFrame(cond).to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("\nPenalty-search benchmark:")
    print(pd.DataFrame(bench).round(4).to_string(index=False))

    plot_sweep(table)
    plotting.prediction_table(plot_test, plot_preds).to_csv(OUT / "test_period_predictions.csv", index=False)
    plotting.plot_test_period(
        plot_test, plot_preds, OUT / "test_period_predictions.png",
        f"Polynomial ridge by degree: building headcount (4 monitored rooms), observed in black - sensor "
        f"{PLOT_SENSOR}, {PLOT_BIN}-min bins, first 14 days of hold-out. Degree chosen by CV: {plot_selected}")
    print(f"\nResults written to {OUT}")


def plot_sweep(table):
    """R2 against polynomial degree: training, hold-out and rolling-CV mean +/- 1 sd."""
    sensors, bins = list(config.ENV_SENSORS), list(config.BIN_MINUTES)
    degrees = list(config.POLY_DEGREES)
    fig, axes = plt.subplots(len(sensors), len(bins), figsize=(11, 3.2 * len(sensors) + 0.6),
                             sharex=True, sharey=True, facecolor=SURFACE)
    for r, sensor in enumerate(sensors):
        for c, m in enumerate(bins):
            ax = axes[r, c]
            d = table[(table["sensor"] == sensor) & (table["bin_minutes"] == m)].sort_values("degree")
            plotting.style_axes(ax)
            ax.axhline(0, color=INK_MUTED, lw=1)
            ax.fill_between(d["degree"], d["r2_mean"] - d["r2_std"], d["r2_mean"] + d["r2_std"],
                            color=CV_COLOR, alpha=0.12, lw=0)
            ax.plot(d["degree"], d["train_r2"], "--", color=INK_MUTED, lw=2, label="Training")
            for col, color, label in (("r2_mean", CV_COLOR, "Rolling CV (mean ± 1 sd)"),
                                      ("r2", HOLDOUT_COLOR, "Hold-out")):
                # Values below the axis floor are drawn on the floor and labelled.
                shown = d[col].clip(lower=Y_FLOOR)
                ax.plot(d["degree"], shown, "-o", color=color, lw=2, ms=8, mec=SURFACE, mew=2, label=label)
                for x, v in zip(d["degree"], d[col]):
                    if v < Y_FLOOR:
                        ax.annotate(f"{v:.2f}", (x, Y_FLOOR), textcoords="offset points", xytext=(-10, 8),
                                    ha="right", fontsize=8, color=INK)
            ax.set_xticks(degrees, [f"{g}\n({models.n_polynomial_terms(len(models.ENV_FEATURES), g)})"
                                    for g in degrees])
            ax.set_ylim(Y_FLOOR - 0.05, 1)
            ax.set_title(f"Sensor {sensor}, {m}-min bins", loc="left", fontsize=10, color=INK, fontweight="bold")
            if r == len(sensors) - 1:
                ax.set_xlabel("Polynomial degree (number of terms)", fontsize=9, color=INK_MUTED)
            if c == 0:
                ax.set_ylabel("R²", fontsize=9, color=INK_MUTED)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "degree_sweep.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
