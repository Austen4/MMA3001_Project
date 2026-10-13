"""
Occupancy sensor data: event stream -> fixed-interval headcounts -> building total.

The occupancy export is event-driven: a row is written only when a room's status or
headcount changes. Each row's state is therefore held until the next row (within limits,
see :mod:`building_occupancy.config`) and integrated over fixed time bins, giving the
time-weighted mean headcount and the fraction of each bin a room was occupied.
"""
import numpy as np
import pandas as pd

from . import config

OCCUPIED, NOT_OCCUPIED = "CurrentlyOccupied", "NotOccupied"


def load_occupancy_events(path=config.RAW_OCC_ZIP, floorspaces=None):
    """Read the raw occupancy export.

    Parameters
    ----------
    path : str or Path
        Raw export (``.zip`` containing one CSV, or the CSV itself).
    floorspaces : iterable of str, optional
        Keep only these floorspace ids.

    Returns
    -------
    pandas.DataFrame
        Columns ``floorspaceid``, ``occupancystatus``, ``headcount`` (float) and ``t``
        (UTC timestamp), sorted by floorspace then time, exact duplicate rows removed.
    """
    cats = {c: "category" for c in ("floorspaceid", "occupancystatus", "collecteddate")}
    df = pd.read_csv(path, usecols=[*cats, "headcount"], dtype={**cats, "headcount": "float32"})
    if floorspaces is not None:
        df = df[df["floorspaceid"].isin(list(floorspaces))]
    # Timestamps repeat heavily, so parse each distinct string once.
    parsed = pd.to_datetime(df["collecteddate"].cat.categories, utc=True, format="mixed")
    df["t"] = parsed.take(df["collecteddate"].cat.codes.to_numpy())
    df["floorspaceid"] = df["floorspaceid"].astype(str)
    df["occupancystatus"] = df["occupancystatus"].astype(str)
    df = df.drop(columns="collecteddate").drop_duplicates()
    return df.sort_values(["floorspaceid", "t"], kind="stable").reset_index(drop=True)


def bin_state_intervals(t_s, status, headcount, bin_s,
                        hold_not_occupied_s=config.HOLD_NOT_OCCUPIED_S,
                        hold_occupied_s=config.HOLD_OCCUPIED_S):
    """Integrate a piecewise-constant occupancy state over fixed time bins.

    Each event's state lasts until the next event, but at most ``hold_not_occupied_s``
    after a ``NotOccupied`` event and ``hold_occupied_s`` after any other status. The
    last event has no known end and contributes nothing.

    Parameters
    ----------
    t_s : array_like of int
        Event times in seconds since the epoch, non-decreasing.
    status : array_like of str
        Occupancy status of each event.
    headcount : array_like of float
        Headcount of each event.
    bin_s : int
        Bin width in seconds. Bins are aligned to the epoch.
    hold_not_occupied_s, hold_occupied_s : int
        Maximum time a state is carried forward.

    Returns
    -------
    pandas.DataFrame
        One row per bin with any coverage: ``bin_start_s``, ``coverage`` (fraction of the
        bin covered, 0-1), ``headcount`` (time-weighted mean over the covered part) and
        ``occupied_share`` (fraction of the covered part with status CurrentlyOccupied).
    """
    t = np.asarray(t_s, dtype=np.int64)
    status = np.asarray(status)
    hc = np.asarray(headcount, dtype=float)
    if not (len(t) == len(status) == len(hc)):
        raise ValueError("t_s, status and headcount must have the same length")
    if bin_s <= 0:
        raise ValueError("bin_s must be positive")
    if np.any(np.diff(t) < 0):
        raise ValueError("t_s must be sorted in non-decreasing order")
    empty = pd.DataFrame({"bin_start_s": [], "coverage": [], "headcount": [], "occupied_share": []})
    if len(t) < 2:
        return empty

    occ = (status == OCCUPIED).astype(float)
    hold = np.where(status == NOT_OCCUPIED, hold_not_occupied_s, hold_occupied_s)
    end = np.minimum(np.append(t[1:], t[-1]), t + hold)
    keep = end > t
    t, end, hc, occ = t[keep], end[keep], hc[keep], occ[keep]
    if len(t) == 0:
        return empty

    # Split every state interval at bin boundaries: interval i spans bins first[i]..last[i].
    first, last = t // bin_s, (end - 1) // bin_s
    n = last - first + 1
    idx = np.repeat(np.arange(len(t)), n)
    bins = np.repeat(first, n) + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    dur = np.minimum(end[idx], (bins + 1) * bin_s) - np.maximum(t[idx], bins * bin_s)

    agg = (pd.DataFrame({"bin": bins, "dur": dur, "hc": hc[idx] * dur, "occ": occ[idx] * dur})
           .groupby("bin").sum())
    return pd.DataFrame({
        "bin_start_s": agg.index.to_numpy() * bin_s,
        "coverage": agg["dur"].to_numpy() / bin_s,
        "headcount": (agg["hc"] / agg["dur"]).to_numpy(),
        "occupied_share": (agg["occ"] / agg["dur"]).to_numpy(),
    })


def bin_occupancy(events, bin_minutes, hold_not_occupied_s=config.HOLD_NOT_OCCUPIED_S,
                  hold_occupied_s=config.HOLD_OCCUPIED_S):
    """Time-weighted occupancy per floorspace on a fixed time grid.

    Parameters
    ----------
    events : pandas.DataFrame
        Output of :func:`load_occupancy_events`.
    bin_minutes : int
        Bin width in minutes (must divide 1440 so bins align with days).
    hold_not_occupied_s, hold_occupied_s : int
        Carry-forward limits, see :func:`bin_state_intervals`.

    Returns
    -------
    pandas.DataFrame
        Columns ``floorspaceid``, ``time_utc`` (bin start), ``coverage``, ``headcount``,
        ``occupied_share``.
    """
    if bin_minutes <= 0 or 1440 % bin_minutes:
        raise ValueError("bin_minutes must be a positive divisor of 1440")
    bin_s = int(bin_minutes * 60)
    parts = []
    for fs, g in events.groupby("floorspaceid", sort=True):
        t_s = (g["t"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)
        b = bin_state_intervals(t_s.to_numpy(np.int64), g["occupancystatus"].to_numpy(),
                                g["headcount"].to_numpy(), bin_s, hold_not_occupied_s, hold_occupied_s)
        b.insert(0, "floorspaceid", fs)
        parts.append(b)
    out = pd.concat(parts, ignore_index=True)
    out.insert(1, "time_utc", pd.to_datetime(out.pop("bin_start_s").astype(np.int64), unit="s", utc=True))
    return out


def building_total(binned, rooms=tuple(config.BUILDING_ROOMS), min_coverage=config.MIN_BIN_COVERAGE):
    """Sum room occupancy into a building-total series.

    A bin is kept only if every room in ``rooms`` has at least ``min_coverage`` of it
    covered, so the total always refers to the same set of rooms.

    Parameters
    ----------
    binned : pandas.DataFrame
        Output of :func:`bin_occupancy`.
    rooms : sequence of str
        Floorspace ids to sum.
    min_coverage : float
        Minimum per-room bin coverage (0-1).

    Returns
    -------
    pandas.DataFrame
        Columns ``time_utc``, ``building_headcount`` (sum of mean headcounts) and
        ``rooms_occupied`` (sum of occupied shares, i.e. expected number of occupied rooms).

    Raises
    ------
    ValueError
        If a requested room has no data.
    """
    rooms = list(rooms)
    absent = set(rooms) - set(binned["floorspaceid"])
    if absent:
        raise ValueError(f"no occupancy data for rooms: {sorted(absent)}")
    ok = binned[binned["floorspaceid"].isin(rooms) & (binned["coverage"] >= min_coverage)]
    wide = ok.pivot(index="time_utc", columns="floorspaceid", values=["headcount", "occupied_share"])
    wide = wide.dropna()
    return pd.DataFrame({
        "time_utc": wide.index,
        "building_headcount": wide["headcount"][rooms].sum(axis=1).to_numpy(),
        "rooms_occupied": wide["occupied_share"][rooms].sum(axis=1).to_numpy(),
    }).reset_index(drop=True)
