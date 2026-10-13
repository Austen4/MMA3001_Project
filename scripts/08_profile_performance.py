"""
Step 8 - profiling, timing and memory measurements.

1. Pipeline stages: wall time and peak resident memory. Heavy stages (raw-data loading)
   run in a fresh process each, so the peak belongs to that stage alone.
2. Occupancy loader, original vs optimised: cProfile hotspots, time, peak memory, and a
   check that both return an identical table.
3. Timestamp parsing scaling: pandas' general parser vs parse-unique-strings-once (the
   original approach) vs the fixed-offset vectorised parser, for growing sample sizes.
4. Interval binning: vectorised NumPy implementation vs a plain Python loop (speed and
   agreement).
5. Final-model fitting: cProfile, measured time, estimated flops, achieved GFLOP/s,
   arithmetic intensity and the share of time actually spent in linear algebra.

Outputs go to results/performance/.
"""
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from building_occupancy import (config, dataset, models, occupancy_data, performance,  # noqa: E402
                                plotting, predict)  # (plotting sets the Agg backend)
from building_occupancy.plotting import INK, INK_MUTED, SURFACE  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OUT = config.RESULTS_DIR / "performance"
PARSE_SIZES = (10_000, 100_000, 1_000_000)
BIN_SIZES = (1_000, 10_000, 100_000)
# Earlier/alternative approaches in blue and orange; the method now used is always aqua.
ORIGINAL, ALTERNATIVE, CURRENT = plotting.CATEGORICAL[0], plotting.CATEGORICAL[1], plotting.CATEGORICAL[2]


def traced(func, *args, **kwargs):
    """Run ``func`` in-process; return (result, seconds, tracemalloc peak MB)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    seconds = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1] / 2 ** 20
    tracemalloc.stop()
    return result, seconds, peak


def main():
    """Run all measurements and write tables, reports and the figure."""
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    stages = []

    # 1-2. Heavy stages, each in a fresh process ---------------------------------------
    print("isolated stages (fresh process each)")
    for label, func, args in (("1 prepare environment data", performance.stage_prepare_environment, ()),
                              ("2a load occupancy - original loader", performance.stage_load_occupancy,
                               ("categorical",)),
                              ("2a load occupancy - optimised loader", performance.stage_load_occupancy,
                               ("fixed_offset",))):
        r = performance.run_isolated(func, *args)
        stages.append({"stage": label, "seconds": r["seconds"], "peak_rss_mb": r["peak_rss_mb"],
                       "baseline_rss_mb": r["baseline_rss_mb"], "rows_out": r["result"],
                       "memory_measure": "peak resident set, fresh process"})
        print(f"  {label}: {r['seconds']:.1f} s, peak {r['peak_rss_mb']:.0f} MB "
              f"(baseline {r['baseline_rss_mb']:.0f} MB)")

    print("profiling the two occupancy loaders")
    old, old_prof, old_txt = performance.profile_call(performance.load_occupancy_events_categorical,
                                                      config.RAW_OCC_ZIP, floorspaces=config.BUILDING_ROOMS)
    new, new_prof, new_txt = performance.profile_call(occupancy_data.load_occupancy_events,
                                                      config.RAW_OCC_ZIP, floorspaces=config.BUILDING_ROOMS)
    pd.testing.assert_frame_equal(old, new)
    identical = True
    del old
    (OUT / "hotspots_load_occupancy_original.txt").write_text(old_txt, encoding="utf-8")
    (OUT / "hotspots_load_occupancy_optimised.txt").write_text(new_txt, encoding="utf-8")
    loader = pd.DataFrame([
        {"loader": name, "profiled_total_s": prof["tottime_s"].sum(),
         "read_csv_s": performance.self_time(prof, r"read|parser|_libs"),
         "timestamp_parsing_s": performance.self_time(prof, r"to_datetime|tslib|strptime|parsing|array_strptime"),
         "categorical_s": performance.self_time(prof, r"[Cc]ategor|factorize")}
        for name, prof in (("original (categorical + general parser)", old_prof),
                           ("optimised (string + fixed-offset parser)", new_prof))])
    loader["identical_output"] = identical
    loader.to_csv(OUT / "loader_profile_breakdown.csv", index=False)
    print(loader.round(1).to_string(index=False))

    # 3. Timestamp parsing scaling -----------------------------------------------------
    print("timestamp parsing benchmark")
    raw_ts = pd.read_csv(config.RAW_OCC_ZIP, usecols=["collecteddate"], dtype=str, nrows=max(PARSE_SIZES))
    methods = {
        "pandas general parser (format='mixed')": lambda s: pd.to_datetime(s, utc=True, format="mixed"),
        "categorical, parse unique strings once": lambda s: (lambda c: pd.to_datetime(
            c.cat.categories, utc=True, format="mixed").take(c.cat.codes.to_numpy()))(s.astype("category")),
        "fixed-offset vectorised (current)": occupancy_data.parse_offset_timestamps,
    }
    parse_rows = []
    for n in PARSE_SIZES:
        s = raw_ts["collecteddate"].iloc[:n]
        ref = None
        for name, f in methods.items():
            t0 = time.perf_counter()
            out = f(s)
            sec = time.perf_counter() - t0
            ref = out if ref is None else ref
            parse_rows.append({"n_rows": n, "n_unique": s.nunique(), "method": name, "seconds": sec,
                               "rows_per_s": n / sec, "matches_reference": bool(np.array_equal(
                                   np.asarray(out), np.asarray(ref)))})
    parse = pd.DataFrame(parse_rows)
    parse.to_csv(OUT / "parse_benchmark.csv", index=False)
    print(parse.round(3).to_string(index=False))

    # 4. In-process light stages and the binning benchmark -----------------------------
    print("in-process stages")
    env = pd.read_csv(config.ENV_CLEAN_CSV, encoding="utf-8-sig")
    for m in config.BIN_MINUTES:
        binned, sec, peak = traced(occupancy_data.bin_occupancy, new, m)
        stages.append({"stage": f"2b bin occupancy ({m}-min)", "seconds": sec, "peak_rss_mb": np.nan,
                       "tracemalloc_peak_mb": peak, "rows_out": len(binned),
                       "memory_measure": "tracemalloc peak (Python/NumPy allocations)"})
        if m == config.FINAL_BIN_MINUTES:
            final_binned = binned
    total, sec, peak = traced(occupancy_data.building_total, final_binned)
    stages.append({"stage": "2c building total (60-min)", "seconds": sec, "tracemalloc_peak_mb": peak,
                   "rows_out": len(total), "memory_measure": "tracemalloc peak (Python/NumPy allocations)"})
    df, sec, peak = traced(dataset.build_dataset, env, total, config.FINAL_SENSOR, config.FINAL_BIN_MINUTES)
    stages.append({"stage": "3 build dataset (final sensor, 60-min)", "seconds": sec, "tracemalloc_peak_mb": peak,
                   "rows_out": len(df), "memory_measure": "tracemalloc peak (Python/NumPy allocations)"})

    print("binning: vectorised vs Python loop")
    room = new[new["floorspaceid"] == next(iter(config.BUILDING_ROOMS))]
    t_s = ((room["t"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy(np.int64)
    st, hc = room["occupancystatus"].to_numpy(), room["headcount"].to_numpy(float)
    bin_rows = []
    for n in BIN_SIZES:
        args = (t_s[:n], st[:n], hc[:n], 3600)
        t0 = time.perf_counter()
        vec = occupancy_data.bin_state_intervals(*args)
        t1 = time.perf_counter()
        loop = performance.bin_state_intervals_loop(*args)
        t2 = time.perf_counter()
        diff = float(np.abs(vec[["coverage", "headcount", "occupied_share"]].to_numpy()
                            - loop[["coverage", "headcount", "occupied_share"]].to_numpy()).max())
        bin_rows.append({"n_events": n, "vectorised_s": t1 - t0, "loop_s": t2 - t1,
                         "speedup": (t2 - t1) / (t1 - t0), "bins": len(vec), "max_abs_difference": diff})
    binning = pd.DataFrame(bin_rows)
    binning.to_csv(OUT / "binning_benchmark.csv", index=False)
    print(binning.round(4).to_string(index=False))
    del new

    # 5. Final-model fitting -------------------------------------------------------------
    print("final-model fitting")
    n = len(df)
    fit_rows = []
    for key, (name, make) in predict.FINAL_MODELS.items():
        model, prof, txt = performance.profile_call(make().fit, df, df[config.TARGET].to_numpy())
        (OUT / f"hotspots_fit_{key}.txt").write_text(txt, encoding="utf-8")
        _, sec, peak = traced(make().fit, df, df[config.TARGET].to_numpy())
        if key == "poly2":
            p = model.n_terms_
            flops = performance.svd_path_fit_flops(n, p)
            kernel = "thin SVD per fold (all penalties)"
        else:
            p = len(model.coef_)
            flops = performance.ridge_fit_flops(n, p, len(model.alphas))
            kernel = "Cholesky ridge solve per penalty per fold"
        linalg = performance.self_time(prof, r"linalg|lapack|_gesdd|svd|cholesky|cho_|solve")
        fit_rows.append({"model": name, "n_rows": n, "n_columns": p, "kernel": kernel,
                         "estimated_gflop": flops / 1e9, "fit_s": sec, "achieved_gflop_per_s": flops / 1e9 / sec,
                         "arithmetic_intensity_flop_per_byte": performance.gram_arithmetic_intensity(p),
                         "profiled_s": prof["tottime_s"].sum(), "linear_algebra_s": linalg,
                         "linear_algebra_share": linalg / prof["tottime_s"].sum(),
                         "tracemalloc_peak_mb": peak})
        stages.append({"stage": f"6 fit final model: {name}", "seconds": sec, "tracemalloc_peak_mb": peak,
                       "rows_out": n, "memory_measure": "tracemalloc peak (Python/NumPy allocations)"})
    fits = pd.DataFrame(fit_rows)
    fits.to_csv(OUT / "fit_flops.csv", index=False)
    print(fits.round(4).to_string(index=False))

    readings = predict.read_readings(config.ROOT / "examples" / "example_readings.csv")
    bundle = predict.load_bundle()
    out, sec, peak = traced(predict.predict, readings, bundle)
    stages.append({"stage": "predict tool: one week (940 readings)", "seconds": sec, "tracemalloc_peak_mb": peak,
                   "rows_out": len(out), "memory_measure": "tracemalloc peak (Python/NumPy allocations)"})

    stage_table = pd.DataFrame(stages)
    stage_table.to_csv(OUT / "stage_timings.csv", index=False)
    print(stage_table[["stage", "seconds", "peak_rss_mb", "tracemalloc_peak_mb", "rows_out"]]
          .round(2).to_string(index=False))
    plot(stage_table, parse, binning)
    print(f"\nResults written to {OUT}  ({time.perf_counter() - t_start:.0f} s)")


def plot(stages, parse, binning):
    """Four panels: loader comparison (time, memory), parse scaling, binning scaling."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), facecolor=SURFACE)
    load = stages[stages["stage"].str.startswith("2a")]
    labels = ["original", "optimised"]
    for ax, col, title in ((axes[0, 0], "seconds", "Load occupancy events: wall time (s)"),
                           (axes[0, 1], "peak_rss_mb", "Load occupancy events: peak memory (MB)")):
        plotting.style_axes(ax)
        ax.grid(False)
        ax.grid(axis="x", color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
        vals = load[col].to_numpy()
        ax.barh(labels, vals, color=[ORIGINAL, CURRENT], height=0.5)
        for y, v in enumerate(vals):
            ax.text(v, y, f" {v:.0f}", va="center", fontsize=9, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(0, vals.max() * 1.18)
        ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
    ax = axes[1, 0]
    plotting.style_axes(ax)
    for color, (name, d) in zip((ALTERNATIVE, ORIGINAL, CURRENT), parse.groupby("method", sort=False)):
        ax.loglog(d["n_rows"], d["seconds"], "-o", color=color, lw=2, ms=6, label=name)
    ax.set_xlabel("Timestamps parsed", fontsize=9, color=INK_MUTED)
    ax.set_title("Timestamp parsing: seconds (log-log)", loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    ax = axes[1, 1]
    plotting.style_axes(ax)
    ax.loglog(binning["n_events"], binning["loop_s"], "-o", color=ALTERNATIVE, lw=2, ms=6, label="Python loop")
    ax.loglog(binning["n_events"], binning["vectorised_s"], "-o", color=CURRENT, lw=2, ms=6,
              label="NumPy vectorised (current)")
    ax.set_xlabel("Occupancy events binned", fontsize=9, color=INK_MUTED)
    ax.set_title("Interval binning: seconds (log-log)", loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "performance.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
