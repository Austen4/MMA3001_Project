"""
Step 4 - train and validate the linear models.

For every environmental sensor and bin width, compares on identical rows:

    Mean baseline, Time-of-day baseline       (no sensor data)
    CO2 mass balance (OLS)                    linear numerical method, own QR solver
    Ridge (environment)                       machine learning, environmental features only
    Ridge (environment + time)                environmental + time-of-day features

Validation: a chronological hold-out (last config.TEST_FRACTION of the rows) and
rolling-origin cross-validation over the whole dataset. Also compares the QR and
normal-equation least-squares solvers for accuracy, conditioning and speed.

Outputs go to results/linear_models/.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from building_occupancy import config, evaluation, models  # noqa: E402

OUT = config.RESULTS_DIR / "linear_models"
PLOT_SENSOR, PLOT_BIN = 6012002000869, 60
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e6e5e0", "#fcfcfb"
SERIES = {"Time-of-day baseline": "#2a78d6", "CO2 mass balance (OLS)": "#eb6834",
          "Ridge (environment)": "#1baf7a", "Ridge (environment + time)": "#eda100"}
SENSOR_COLORS = ["#2a78d6", "#eb6834"]


def solver_comparison(df, features, repeats=20):
    """Compare QR, normal-equation and LAPACK least-squares solutions on one design.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset rows (rows with NaN in ``features`` are dropped).
    features : list of str
        Regressors; an intercept column is added.
    repeats : int
        Timing repetitions.

    Returns
    -------
    list of dict
        Per solver: condition numbers, relative coefficient error against
        ``numpy.linalg.lstsq`` (SVD-based reference) and mean solve time.
    """
    d = df.dropna(subset=features)
    X = np.column_stack([np.ones(len(d)), d[features].to_numpy(float)])
    y = d[config.TARGET].to_numpy(float)
    ref = np.linalg.lstsq(X, y, rcond=None)[0]
    cond_x = np.linalg.cond(X)
    rows = []
    for name, solve in (("QR + back-substitution (own)", models.solve_lstsq_qr),
                        ("Normal equations + Cholesky (own)", models.solve_lstsq_normal),
                        ("numpy.linalg.lstsq (SVD reference)", lambda A, b: np.linalg.lstsq(A, b, rcond=None)[0])):
        t0 = time.perf_counter()
        for _ in range(repeats):
            beta = solve(X, y)
        rows.append({"solver": name, "n_rows": len(X), "n_coef": X.shape[1], "cond_X": cond_x,
                     "cond_XtX": cond_x ** 2,
                     "rel_coef_error": float(np.linalg.norm(beta - ref) / np.linalg.norm(ref)),
                     "solve_ms": (time.perf_counter() - t0) / repeats * 1e3})
    return rows


def main():
    """Run all experiments and write results."""
    OUT.mkdir(parents=True, exist_ok=True)
    holdout, cv, physics, ridge_coefs, solvers, test_preds = [], [], [], [], [], {}
    for sensor in config.ENV_SENSORS:
        for m in config.BIN_MINUTES:
            df = pd.read_csv(config.PROCESSED_DIR / f"dataset_{sensor}_{m}min.csv", parse_dates=["time_utc"])
            tag = {"sensor": sensor, "bin_minutes": m}
            train, test = evaluation.chronological_split(df)

            res, fitted = evaluation.evaluate(models.default_models(), train, test)
            holdout.append(res.assign(**tag, test_from=test["time_utc"].iloc[0]))
            cv.append(evaluation.cross_validate(models.default_models(), df).assign(**tag))

            mb = fitted["CO2 mass balance (OLS)"]
            physics.append({**tag, **mb.coef_.add_prefix("coef_").to_dict(),
                            "condition_number": mb.condition_number_, **mb.implied_physics()})
            for name in ("Ridge (environment)", "Ridge (environment + time)"):
                r = fitted[name]
                ridge_coefs.append(pd.DataFrame({**tag, "model": name, "alpha": r.alpha_,
                                                 "feature": r.coef_.index, "std_coef": r.coef_.to_numpy()}))
            for feats, label in ((models.MASS_BALANCE_FEATURES, "mass balance"),
                                 (models.ENV_FEATURES, "all environmental (unscaled)")):
                solvers += [{**tag, "design": label, **row} for row in solver_comparison(df, feats)]
            if (sensor, m) == (PLOT_SENSOR, PLOT_BIN):
                test_preds = {"test": test, **{n: fitted[n].predict(test) for n in SERIES}}
            print(f"sensor {sensor}, {m:>3} min: done ({len(df):,} rows)")

    holdout = pd.concat(holdout, ignore_index=True)
    cv = pd.concat(cv, ignore_index=True)
    cv_summary = (cv.groupby(["sensor", "bin_minutes", "model"], sort=False)[["mae", "rmse", "r2"]]
                  .agg(["mean", "std"]).round(3))
    cv_summary.columns = ["_".join(c) for c in cv_summary.columns]
    holdout.to_csv(OUT / "holdout_metrics.csv", index=False)
    cv.to_csv(OUT / "cv_folds.csv", index=False)
    cv_summary.to_csv(OUT / "cv_summary.csv")
    pd.DataFrame(physics).to_csv(OUT / "mass_balance_fit.csv", index=False)
    pd.concat(ridge_coefs, ignore_index=True).to_csv(OUT / "ridge_coefficients.csv", index=False)
    pd.DataFrame(solvers).to_csv(OUT / "solver_comparison.csv", index=False)

    print("\nHold-out R2 (last 25% of each dataset):")
    print(holdout.pivot_table(index=["sensor", "bin_minutes"], columns="model", values="r2",
                              sort=False).round(3).to_string())
    print("\nRolling-origin CV, mean R2 (std):")
    print(cv_summary[["r2_mean", "r2_std"]].unstack("model").round(3).to_string())
    print("\nMass-balance fit:")
    print(pd.DataFrame(physics).round(4).to_string(index=False))

    plot_test_period(test_preds)
    plot_r2(holdout)
    print(f"\nResults written to {OUT}")


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def plot_test_period(p, days=14):
    """Observed vs predicted building headcount over the first ``days`` of the test period."""
    test = p["test"]
    sel = (test["time_utc"] < test["time_utc"].iloc[0] + pd.Timedelta(days=days)).to_numpy()
    t = test["time_utc"][sel].dt.tz_convert(config.TIMEZONE).dt.tz_localize(None)
    names = list(SERIES)
    fig, axes = plt.subplots(len(names), 1, figsize=(11, 2.1 * len(names) + 0.8), sharex=True,
                             sharey=True, facecolor=SURFACE)
    for ax, name in zip(axes, names):
        _style(ax)
        ax.plot(t, test[config.TARGET][sel], color=INK, lw=1.2, label="Observed")
        ax.plot(t, p[name][sel], color=SERIES[name], lw=2, label=name)
        ax.set_title(name, loc="left", fontsize=10, color=INK, fontweight="bold")
        ax.set_ylim(bottom=0)
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    axes[-1].set_xlabel("Sydney local time", fontsize=9, color=INK_MUTED)
    fig.text(0.01, 0.985, f"Building headcount (4 monitored rooms), observed in black - sensor {PLOT_SENSOR}, "
             f"{PLOT_BIN}-min bins, first {days} days of hold-out", fontsize=9, color=INK_MUTED, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "test_period_predictions.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_r2(holdout):
    """Hold-out R2 per model, one panel per bin width, one marker colour per sensor."""
    model_order = list(models.default_models())
    bins = list(config.BIN_MINUTES)
    fig, axes = plt.subplots(1, len(bins), figsize=(11, 3.4), sharey=True, facecolor=SURFACE)
    lo = min(-0.5, holdout["r2"].min() - 0.05)
    for ax, m in zip(axes, bins):
        _style(ax)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.axvline(0, color=INK_MUTED, lw=1)
        for i, (sensor, color) in enumerate(zip(config.ENV_SENSORS, SENSOR_COLORS)):
            d = holdout[(holdout["sensor"] == sensor) & (holdout["bin_minutes"] == m)].set_index("model")
            y = np.arange(len(model_order)) + (i - 0.5) * 0.25
            ax.plot(d.loc[model_order, "r2"].clip(lower=lo), y, "o", ms=8, color=color, mec=SURFACE, mew=2,
                    label=f"Sensor {sensor}")
        ax.set_yticks(range(len(model_order)), model_order)
        ax.invert_yaxis()
        ax.set_xlim(lo, 1)
        ax.set_title(f"{m}-min bins", loc="left", fontsize=10, color=INK, fontweight="bold")
        ax.set_xlabel("Hold-out R²", fontsize=9, color=INK_MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "holdout_r2.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
