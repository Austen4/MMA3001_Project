"""
Performance measurement: timing, peak memory, profiling and operation-count estimates.

Heavy stages are measured in a fresh child process (:func:`run_isolated`) so the peak
resident memory reported belongs to that stage alone and includes memory allocated by C
code (pandas' CSV parser, NumPy, BLAS) that Python-level tracers cannot see.

The module also keeps two *reference implementations* used as performance baselines and as
correctness checks for the optimised code:

* :func:`load_occupancy_events_categorical` - the original loader (categorical timestamps,
  pandas' general date parser), replaced by
  :func:`building_occupancy.occupancy_data.load_occupancy_events`;
* :func:`bin_state_intervals_loop` - a plain Python loop equivalent of the vectorised
  :func:`building_occupancy.occupancy_data.bin_state_intervals`.
"""
import cProfile
import io
import os
import pstats
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from . import config, env_data, occupancy_data

# ---------------------------------------------------------------- measurement


def peak_rss_bytes():
    """Peak resident memory of the current process so far, in bytes.

    Uses the peak working set on Windows (psutil) and ``ru_maxrss`` elsewhere.

    Returns
    -------
    int
    """
    if sys.platform == "win32":
        import psutil
        return int(psutil.Process().memory_info().peak_wset)
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)   # macOS bytes, Linux KiB


def current_rss_bytes():
    """Current resident memory of this process, in bytes (psutil)."""
    import psutil
    return int(psutil.Process().memory_info().rss)


def _timed_child(func, args, kwargs):
    """Child-process body: baseline memory, run ``func``, report time and peak memory."""
    base = current_rss_bytes()
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    seconds = time.perf_counter() - t0
    return {"seconds": seconds, "peak_rss_mb": peak_rss_bytes() / 2 ** 20,
            "baseline_rss_mb": base / 2 ** 20, "result": result, "pid": os.getpid()}


def run_isolated(func, *args, **kwargs):
    """Run ``func(*args, **kwargs)`` in a fresh process; return time, peak memory and result.

    ``func`` must be importable (defined at module level) and return something picklable.

    Returns
    -------
    dict
        ``seconds`` (wall time of the call), ``peak_rss_mb`` (peak resident memory of the
        child, including interpreter and imports), ``baseline_rss_mb`` (resident memory
        just before the call), ``result``.
    """
    with ProcessPoolExecutor(max_workers=1) as pool:
        return pool.submit(_timed_child, func, args, kwargs).result()


def profile_call(func, *args, top=15, **kwargs):
    """Run ``func`` under :mod:`cProfile`.

    Parameters
    ----------
    func : callable
    *args, **kwargs
        Arguments for ``func``.
    top : int
        Number of functions to keep.

    Returns
    -------
    (object, pandas.DataFrame, str)
        The result; every profiled function (``function``, ``ncalls``, ``tottime_s`` =
        time in the function itself, ``cumtime_s`` = including callees), sorted by
        cumulative time; and the pstats text report of the ``top`` functions.
    """
    prof = cProfile.Profile()
    result = prof.runcall(func, *args, **kwargs)
    stats = pstats.Stats(prof)
    rows = []
    for (filename, line, name), (cc, nc, tt, ct, _) in stats.stats.items():
        where = os.path.basename(filename) if filename != "~" else ""
        rows.append({"function": f"{where}:{line}({name})" if where else name,
                     "ncalls": nc, "tottime_s": tt, "cumtime_s": ct})
    table = pd.DataFrame(rows).sort_values("cumtime_s", ascending=False).reset_index(drop=True)
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(top)
    return result, table, buf.getvalue()


def self_time(profile_table, pattern):
    """Seconds spent inside functions whose name matches regular expression ``pattern``.

    Parameters
    ----------
    profile_table : pandas.DataFrame
        Second output of :func:`profile_call`.
    pattern : str
        Regular expression matched against the ``function`` column.

    Returns
    -------
    float
        Sum of ``tottime_s`` (self time, so nested matches are not double counted).
    """
    hit = profile_table["function"].str.contains(pattern, regex=True)
    return float(profile_table.loc[hit, "tottime_s"].sum())


# ---------------------------------------------------------------- isolated stages (picklable)


def stage_prepare_environment():
    """Pipeline step 1 work: raw environmental export -> cleaned readings. Returns row count."""
    raw = env_data.load_raw_environment(config.RAW_ENV_ZIP)
    return len(env_data.clean_environment(env_data.deduplicate(raw)))


def stage_load_occupancy(method="fixed_offset"):
    """Load the occupancy events with the current (``fixed_offset``) or original
    (``categorical``) loader. Returns the row count."""
    loader = {"fixed_offset": occupancy_data.load_occupancy_events,
              "categorical": load_occupancy_events_categorical}[method]
    return len(loader(config.RAW_OCC_ZIP, floorspaces=config.BUILDING_ROOMS))


# ---------------------------------------------------------------- reference implementations


def load_occupancy_events_categorical(path=config.RAW_OCC_ZIP, floorspaces=None):
    """Original occupancy loader, kept as the performance baseline.

    Reads timestamps as a categorical column and parses each distinct string once with
    pandas' general (``format="mixed"``) parser. This assumed timestamps repeat heavily;
    in this export 74% are unique, so it is slower and uses more memory than
    :func:`building_occupancy.occupancy_data.load_occupancy_events`, which returns an
    identical table.
    """
    cats = {c: "category" for c in ("floorspaceid", "occupancystatus", "collecteddate")}
    df = pd.read_csv(path, usecols=[*cats, "headcount"], dtype={**cats, "headcount": "float32"})
    if floorspaces is not None:
        df = df[df["floorspaceid"].isin(list(floorspaces))]
    parsed = pd.to_datetime(df["collecteddate"].cat.categories, utc=True, format="mixed")
    df["t"] = parsed.take(df["collecteddate"].cat.codes.to_numpy())
    df["floorspaceid"] = df["floorspaceid"].astype(str)
    df["occupancystatus"] = df["occupancystatus"].astype(str)
    df = df.drop(columns="collecteddate").drop_duplicates()
    return df.sort_values(["floorspaceid", "t"], kind="stable").reset_index(drop=True)


def bin_state_intervals_loop(t_s, status, headcount, bin_s,
                             hold_not_occupied_s=config.HOLD_NOT_OCCUPIED_S,
                             hold_occupied_s=config.HOLD_OCCUPIED_S):
    """Plain-Python reference for :func:`building_occupancy.occupancy_data.bin_state_intervals`.

    Same inputs and output; walks every event and every bin it touches in a Python loop.
    Used to verify the vectorised version and to measure its speed-up.
    """
    acc = {}
    n = len(t_s)
    for i in range(n - 1):
        start = int(t_s[i])
        hold = hold_not_occupied_s if status[i] == occupancy_data.NOT_OCCUPIED else hold_occupied_s
        end = min(int(t_s[i + 1]), start + hold)
        occ = 1.0 if status[i] == occupancy_data.OCCUPIED else 0.0
        while start < end:
            b = start // bin_s
            piece_end = min(end, (b + 1) * bin_s)
            d = piece_end - start
            cell = acc.setdefault(b, [0.0, 0.0, 0.0])
            cell[0] += d
            cell[1] += headcount[i] * d
            cell[2] += occ * d
            start = piece_end
    bins = sorted(acc)
    return pd.DataFrame({
        "bin_start_s": [b * bin_s for b in bins],
        "coverage": [acc[b][0] / bin_s for b in bins],
        "headcount": [acc[b][1] / acc[b][0] for b in bins],
        "occupied_share": [acc[b][2] / acc[b][0] for b in bins],
    })


# ---------------------------------------------------------------- operation counts


def lstsq_cholesky_flops(m, n):
    """Approximate flops of one ridge solve via the normal equations and Cholesky.

    Forming ``X^T X`` (~``m n^2``, using symmetry) plus the factorisation (~``n^3 / 3``);
    triangular solves (~``2 n^2``) are negligible.
    """
    return m * n ** 2 + n ** 3 / 3


def thin_svd_flops(m, n):
    """Approximate flops of a thin SVD of an ``m x n`` matrix, ``m >= n`` (R-SVD,
    Golub & Van Loan, 4th ed., sec. 8.6): ``6 m n^2 + 20 n^3``."""
    return 6 * m * n ** 2 + 20 * n ** 3


def inner_fold_sizes(n_rows, n_splits=config.RIDGE_INNER_SPLITS):
    """Training-set sizes of the inner ``TimeSeriesSplit`` folds for ``n_rows`` rows."""
    return [len(tr) for tr, _ in TimeSeriesSplit(n_splits=n_splits).split(np.zeros(n_rows))]


def ridge_fit_flops(n_rows, n_features, n_alphas, n_splits=config.RIDGE_INNER_SPLITS):
    """Approximate flops to fit :class:`building_occupancy.models.RidgeRegression`:
    one Cholesky ridge solve per penalty per inner fold, plus the final refit."""
    folds = inner_fold_sizes(n_rows, n_splits)
    return n_alphas * sum(lstsq_cholesky_flops(m, n_features) for m in folds) + \
        lstsq_cholesky_flops(n_rows, n_features)


def gram_arithmetic_intensity(n_columns):
    """Arithmetic intensity (flops per byte) of forming ``X^T X`` for float64 ``X``.

    ``m n^2`` flops over ``8 m n`` bytes read gives ``n / 8``: with the 18-54 columns used
    here (2-7 flop/byte) these kernels are memory-bandwidth bound on typical CPUs, whose
    balance point is ~10-20 flop/byte.
    """
    return n_columns / 8


def svd_path_fit_flops(n_rows, n_terms, n_splits=config.RIDGE_INNER_SPLITS):
    """Approximate flops to fit :class:`building_occupancy.models.PolynomialRidgeRegression`
    for one degree: one thin SVD per inner fold (all penalties) plus the final refit."""
    folds = inner_fold_sizes(n_rows, n_splits)
    return sum(thin_svd_flops(m, n_terms) for m in folds) + thin_svd_flops(n_rows, n_terms)
