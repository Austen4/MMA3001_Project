"""
Step 7 - one-at-a-time sensitivity analysis of the final models.

Baseline: sensor config.FINAL_SENSOR, config.FINAL_BIN_MINUTES bins, all settings as in
config.py. Each setting below is varied alone; the time-of-day baseline and both final
models are re-validated (chronological hold-out, plus rolling-origin CV where the setting
does not itself define the split):

    target construction   hold limit after NotOccupied, hold limit after occupied states,
                          minimum bin coverage, rooms summed into the target
    validation design     test fraction, training-window length          (hold-out only)
    model inputs          feature-group ablation
    regularisation        fixed ridge penalty instead of CV-selected      (hold-out only)

Outputs go to results/sensitivity/.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from building_occupancy import config, dataset, occupancy_data, plotting, sensitivity  # noqa: E402  (Agg backend)
from building_occupancy.plotting import INK, INK_MUTED, SURFACE  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OUT = config.RESULTS_DIR / "sensitivity"
BIN = config.FINAL_BIN_MINUTES
COLORS = dict(zip(sensitivity.MODEL_NAMES, ["#2a78d6", "#eda100", "#eb6834"]))   # as in earlier figures
PANELS = [  # parameter key, axis label, baseline value (as stored in the table)
    ("hold_not_occupied_h", "Hold limit after NotOccupied (h)", "72"),
    ("hold_occupied_h", "Hold limit after occupied states (h)", "1"),
    ("min_coverage", "Minimum bin coverage", "0.5"),
    ("rooms", "Rooms summed into the target", "all 4 rooms"),
    ("test_fraction", "Hold-out fraction", "0.25"),
    ("train_days", "Training window (days before hold-out)", "all"),
]


def main():
    """Run every scenario and write tables and figures."""
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    env = pd.read_csv(config.ENV_CLEAN_CSV, encoding="utf-8-sig")
    events = occupancy_data.load_occupancy_events(config.RAW_OCC_ZIP, floorspaces=config.BUILDING_ROOMS)
    print(f"loaded {len(events):,} occupancy events ({time.perf_counter() - t_start:.0f} s)")

    def build(binned, rooms=tuple(config.BUILDING_ROOMS), min_coverage=config.MIN_BIN_COVERAGE):
        building = occupancy_data.building_total(binned, rooms, min_coverage)
        return dataset.build_dataset(env, building, config.FINAL_SENSOR, BIN)

    rows = []

    def record(parameter, value, df, cv=True, factories=None, **kw):
        t0 = time.perf_counter()
        res = sensitivity.scenario_scores(df, factories or sensitivity.model_factories(), cv=cv, **kw)
        rows.append(res.assign(parameter=parameter, value=str(value), n_rows=len(df)))
        best = res.set_index("model")["r2"]
        print(f"  {parameter} = {value}: rows {len(df):,}, hold-out R2 "
              + ", ".join(f"{m.split(' (')[0]} {best[m]:.2f}" for m in best.index)
              + f"  ({time.perf_counter() - t0:.0f} s)")

    base_binned = occupancy_data.bin_occupancy(events, BIN)
    base = build(base_binned)

    print("target construction")
    for h in config.SENS_HOLD_NOT_OCCUPIED_H:
        binned = base_binned if h * 3600 == config.HOLD_NOT_OCCUPIED_S else occupancy_data.bin_occupancy(
            events, BIN, hold_not_occupied_s=h * 3600)
        record("hold_not_occupied_h", f"{h:g}", build(binned))
    for h in config.SENS_HOLD_OCCUPIED_H:
        binned = base_binned if h * 3600 == config.HOLD_OCCUPIED_S else occupancy_data.bin_occupancy(
            events, BIN, hold_occupied_s=int(h * 3600))
        record("hold_occupied_h", f"{h:g}", build(binned))
    for c in config.SENS_MIN_COVERAGE:
        record("min_coverage", f"{c:g}", build(base_binned, min_coverage=c))
    record("rooms", "all 4 rooms", base)
    for fs, label in config.BUILDING_ROOMS.items():
        record("rooms", label, build(base_binned, rooms=[fs]))

    print("validation design")
    for f in config.SENS_TEST_FRACTION:
        record("test_fraction", f"{f:g}", base, cv=False, test_fraction=f)
    for d in config.SENS_TRAIN_DAYS:
        record("train_days", "all" if d is None else d, base, cv=False, train_days=d)

    print("feature ablation")
    for label, feats in sensitivity.ablation_feature_sets().items():
        record("features", label, base, factories=sensitivity.model_factories(features=feats))

    print("fixed ridge penalty")
    for a in config.SENS_FIXED_ALPHAS:
        f = sensitivity.model_factories(alphas=[a])
        f.pop(sensitivity.MODEL_NAMES[0])
        record("alpha", f"{a:.0e}", base, cv=False, factories=f)
    selected = sensitivity.scenario_scores(base, sensitivity.model_factories(), cv=False)

    table = pd.concat(rows, ignore_index=True)
    cols = ["parameter", "value", "model", "n_rows", "n_train", "n_test", "mae", "rmse", "bias", "r2",
            "cv_r2_mean", "cv_r2_std", "cv_mae_mean", "fit_s"]
    table = table[[c for c in cols if c in table.columns]]
    table.to_csv(OUT / "sensitivity.csv", index=False)
    summary = sensitivity_summary(table)
    summary.to_csv(OUT / "sensitivity_summary.csv", index=False)
    print("\nRange of hold-out R2 across each parameter's values:")
    print(summary.round(3).to_string(index=False))

    plot_parameters(table)
    plot_ablation(table)
    plot_alpha(table, selected)
    print(f"\nResults written to {OUT}  ({time.perf_counter() - t_start:.0f} s)")


def sensitivity_summary(table):
    """Spread of hold-out R2 and MAE across each parameter's values, per model."""
    g = table.groupby(["parameter", "model"], sort=False)
    return g.agg(r2_min=("r2", "min"), r2_max=("r2", "max"), mae_min=("mae", "min"),
                 mae_max=("mae", "max")).assign(r2_range=lambda d: d.r2_max - d.r2_min).reset_index()


def plot_parameters(table):
    """Hold-out R2 against each varied setting; the baseline value is shaded."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), sharey=True, facecolor=SURFACE)
    for ax, (param, label, base_value) in zip(axes.ravel(), PANELS):
        plotting.style_axes(ax)
        d = table[table["parameter"] == param]
        values = list(dict.fromkeys(d["value"]))
        x = np.arange(len(values))
        ax.axvspan(values.index(base_value) - 0.3, values.index(base_value) + 0.3, color="#e6e5e0", lw=0)
        ax.axhline(0, color=INK_MUTED, lw=1)
        for model in sensitivity.MODEL_NAMES:
            r2 = d[d["model"] == model].set_index("value").loc[values, "r2"]
            ax.plot(x, r2, "-o", color=COLORS[model], lw=2, ms=7, mec=SURFACE, mew=1.5, label=model)
        ax.set_xticks(x, values, fontsize=8, rotation=20 if param == "rooms" else 0)
        ax.set_title(label, loc="left", fontsize=10, color=INK, fontweight="bold")
    for ax in axes[:, 0]:
        ax.set_ylabel("Hold-out R²", fontsize=9, color=INK_MUTED)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    fig.text(0.01, 0.005, f"One setting varied at a time from the baseline (shaded); sensor {config.FINAL_SENSOR}, "
             f"{BIN}-min bins. Rolling-CV scores are in sensitivity.csv.", fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(OUT / "parameter_sensitivity.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_ablation(table):
    """Hold-out and rolling-CV R2 of the final models for each feature set."""
    d = table[table["parameter"] == "features"]
    sets = list(dict.fromkeys(d["value"]))
    baseline_r2 = d[d["model"] == sensitivity.MODEL_NAMES[0]]["r2"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True, facecolor=SURFACE)
    for ax, (col, title) in zip(axes, (("r2", "Hold-out R²"), ("cv_r2_mean", "Rolling-CV mean R²"))):
        plotting.style_axes(ax)
        ax.grid(axis="x", color="#e6e5e0", lw=0.8)
        ref = baseline_r2 if col == "r2" else d[d["model"] == sensitivity.MODEL_NAMES[0]][col].iloc[0]
        ax.axvline(ref, color=COLORS[sensitivity.MODEL_NAMES[0]], lw=1.5, ls="--",
                   label=f"{sensitivity.MODEL_NAMES[0]} (no sensor)")
        for i, model in enumerate(sensitivity.MODEL_NAMES[1:]):
            vals = d[d["model"] == model].set_index("value").loc[sets, col]
            ax.plot(vals, np.arange(len(sets)) + (i - 0.5) * 0.25, "o", ms=8, color=COLORS[model],
                    mec=SURFACE, mew=1.5, label=model)
        ax.set_yticks(range(len(sets)), sets, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "feature_ablation.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_alpha(table, selected):
    """Hold-out R2 with the ridge penalty fixed, against the CV-selected penalty."""
    d = table[table["parameter"] == "alpha"]
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    plotting.style_axes(ax)
    for model in sensitivity.MODEL_NAMES[1:]:
        m = d[d["model"] == model]
        ax.plot(m["value"].astype(float), m["r2"], "-o", color=COLORS[model], lw=2, ms=6, mec=SURFACE,
                mew=1.5, label=f"{model}, fixed α")
        ax.axhline(selected.set_index("model").loc[model, "r2"], color=COLORS[model], lw=1.2, ls="--",
                   label=f"{model}, α chosen by inner CV")
    ax.set_xscale("log")
    ax.set_xlabel("Ridge penalty α", fontsize=9, color=INK_MUTED)
    ax.set_ylabel("Hold-out R²", fontsize=9, color=INK_MUTED)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "alpha_sensitivity.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
