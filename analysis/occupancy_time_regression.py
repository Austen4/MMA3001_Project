"""
Occupancy vs time-of-day regression, per room.

The occupancy sensor is event-driven: a row is written only when a room's status or
headcount changes, so long silences (overnight, weekends) mean "no change", not "no data".
This script converts that stream into time-weighted hourly values per floorspace, keeps
the rooms with enough coverage, and regresses occupancy on local time of day
(Australia/Sydney) for two responses:

  headcount      mean people in the room over the hour
  occupied_share fraction of the hour the room was CurrentlyOccupied

Two models per room and response:

  harmonic   y ~ K sine/cosine pairs of hour-of-day, with a separate curve for weekends
             (OLS for headcount, fractional-logit GLM for occupied_share)
  hour_means mean of y for each hour x weekday/weekend cell -- equivalent to a
             dummy-variable regression; the ceiling for any time-of-day-only model

Models are scored on a chronological hold-out (last 20% of each room's hours).
Coefficients are from a full-data fit with HAC (autocorrelation-robust) standard errors.

Run from the repo root:  python analysis/occupancy_time_regression.py
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "5occupancySensor_MayToDec2024_9MRows.zip"
LOCATIONS = ROOT / "data" / "sensor_locations_occ.csv"
OUT = ROOT / "analysis" / "output"
TZ = "Australia/Sydney"

# How long the last reported state is assumed to persist when no new row arrives.
HOLD_NOT_OCCUPIED_S = 72 * 3600  # empty rooms legitimately go quiet overnight/weekends
HOLD_OCCUPIED_S = 3600           # an occupied room that stops reporting is treated as missing

# "Enough data" rules.
MIN_HOUR_COVERAGE = 0.5  # keep an hour if at least half of it is covered
MIN_HOURS_PER_DAY = 20   # a valid day has at least this many kept hours
MIN_VALID_DAYS = 60      # a room needs at least this many valid days ...
MIN_CELL_OBS = 20        # ... and this many hours in every hour x day-type cell

K_HARMONICS = 3
TEST_FRAC = 0.2
HAC_LAGS = 24

WEEKDAY_COLOR, WEEKEND_COLOR = "#2a78d6", "#eb6834"
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e6e5e0", "#fcfcfb"


def load_events():
    cats = {c: "category" for c in ("floorspaceid", "occupancystatus", "collecteddate")}
    df = pd.read_csv(DATA, usecols=[*cats, "headcount"], dtype={**cats, "headcount": "float32"})
    # Timestamps repeat heavily, so parse each distinct string once.
    parsed = pd.to_datetime(df.collecteddate.cat.categories, utc=True, format="mixed")
    df["t"] = parsed.take(df.collecteddate.cat.codes.to_numpy())
    df = df.drop(columns="collecteddate").drop_duplicates()
    return df.sort_values(["floorspaceid", "t"], kind="stable")


def to_hourly(g):
    """Time-weighted hourly headcount and occupied share for one floorspace's events."""
    t = ((g.t - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy(np.int64)
    status = g.occupancystatus.astype(str).to_numpy()
    hc = g.headcount.to_numpy(float)
    occ = (status == "CurrentlyOccupied").astype(float)
    hold = np.where(status == "NotOccupied", HOLD_NOT_OCCUPIED_S, HOLD_OCCUPIED_S)
    end = np.minimum(np.append(t[1:], t[-1]), t + hold)
    keep = end > t
    t, end, hc, occ = t[keep], end[keep], hc[keep], occ[keep]

    # Split each state interval into pieces at hour boundaries.
    first, last = t // 3600, (end - 1) // 3600
    n = last - first + 1
    idx = np.repeat(np.arange(len(t)), n)
    hour = np.repeat(first, n) + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    dur = np.minimum(end[idx], (hour + 1) * 3600) - np.maximum(t[idx], hour * 3600)

    agg = (pd.DataFrame({"hour": hour, "dur": dur, "hc": hc[idx] * dur, "occ": occ[idx] * dur})
           .groupby("hour").sum())
    return pd.DataFrame({
        "time_utc": pd.to_datetime(agg.index * 3600, unit="s", utc=True),
        "coverage": agg.dur.to_numpy() / 3600,
        "headcount": (agg.hc / agg.dur).to_numpy(),
        "occupied_share": (agg.occ / agg.dur).to_numpy(),
    })


def room_labels(spaces):
    locs = pd.read_csv(LOCATIONS, dtype=str).dropna(subset=["SpaceId"])
    rooms = locs.groupby("SpaceId").room_key.agg(lambda s: "/".join(sorted(set(s))))
    return {s: f"{rooms[s]} ({s[:8]})" if s in rooms else f"Unmapped ({s[:8]})" for s in spaces}


def harmonic_X(hour, weekend):
    ang = 2 * np.pi * (np.asarray(hour) + 0.5) / 24
    base = {"const": np.ones(len(ang))}
    for k in range(1, K_HARMONICS + 1):
        base[f"sin{k}"], base[f"cos{k}"] = np.sin(k * ang), np.cos(k * ang)
    X = pd.DataFrame(base)
    return pd.concat([X, X.mul(np.asarray(weekend, float), axis=0).add_prefix("weekend:")], axis=1)


def fit_harmonic(y, X, response, cov="nonrobust"):
    kw = {"cov_type": "HAC", "cov_kwds": {"maxlags": HAC_LAGS}} if cov == "HAC" else {}
    if response == "headcount":
        return sm.OLS(y, X).fit(**kw)
    return sm.GLM(y, X, family=sm.families.Binomial()).fit(**kw)


def scores(y, pred):
    resid = y - pred
    return {"r2": 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum(),
            "rmse": np.sqrt((resid ** 2).mean()), "mae": resid.abs().mean()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading events ...")
    events = load_events()

    hourly = pd.concat(
        [to_hourly(g).assign(floorspaceid=fs) for fs, g in events.groupby("floorspaceid", observed=True)],
        ignore_index=True)
    hourly = hourly[hourly.coverage >= MIN_HOUR_COVERAGE].copy()
    local = hourly.time_utc.dt.tz_convert(TZ)
    hourly["date_local"] = local.dt.date
    hourly["hour"] = local.dt.hour
    hourly["weekend"] = local.dt.dayofweek >= 5
    labels = room_labels(hourly.floorspaceid.unique())
    hourly["room"] = hourly.floorspaceid.map(labels)
    hourly.to_csv(OUT / "hourly_occupancy.csv", index=False)

    # Which rooms have enough data?
    rows = []
    for fs, g in hourly.groupby("floorspaceid"):
        per_day = g.groupby("date_local").size()
        rows.append({
            "floorspaceid": fs, "room": labels[fs],
            "first_hour": g.time_utc.min(), "last_hour": g.time_utc.max(),
            "hours_kept": len(g), "valid_days": int((per_day >= MIN_HOURS_PER_DAY).sum()),
            "min_cell_obs": int(g.groupby(["hour", "weekend"]).size().reindex(
                pd.MultiIndex.from_product([range(24), [False, True]]), fill_value=0).min()),
        })
    coverage = pd.DataFrame(rows)
    coverage["qualifies"] = (coverage.valid_days >= MIN_VALID_DAYS) & (coverage.min_cell_obs >= MIN_CELL_OBS)
    coverage.to_csv(OUT / "room_data_coverage.csv", index=False)
    print(coverage.drop(columns="floorspaceid").to_string(index=False))

    metrics, coefs, profiles = [], [], []
    grid = pd.DataFrame([(h, w) for w in (False, True) for h in range(24)], columns=["hour", "weekend"])
    for fs in coverage.loc[coverage.qualifies, "floorspaceid"]:
        g = hourly[hourly.floorspaceid == fs].sort_values("time_utc").reset_index(drop=True)
        split = int(len(g) * (1 - TEST_FRAC))
        train, test = g.iloc[:split], g.iloc[split:]
        prof = grid.assign(room=labels[fs])
        for resp in ("headcount", "occupied_share"):
            # Hold-out scoring.
            m = fit_harmonic(train[resp], harmonic_X(train.hour, train.weekend), resp)
            cell = train.groupby(["hour", "weekend"])[resp].mean()
            preds = {
                "harmonic": m.predict(harmonic_X(test.hour, test.weekend)),
                "hour_means": cell.reindex(pd.MultiIndex.from_frame(test[["hour", "weekend"]])).to_numpy(),
            }
            for name, p in preds.items():
                metrics.append({"room": labels[fs], "response": resp, "model": name,
                                "n_train": len(train), "n_test": len(test),
                                **scores(test[resp].reset_index(drop=True), pd.Series(np.asarray(p)))})

            # Full-data fit for coefficients and the plotted profile.
            full = fit_harmonic(g[resp], harmonic_X(g.hour, g.weekend), resp, cov="HAC")
            coefs.append(pd.DataFrame({
                "room": labels[fs], "response": resp, "term": full.params.index,
                "coef": full.params.to_numpy(), "std_err_hac": full.bse.to_numpy(),
                "p_value": full.pvalues.to_numpy()}))
            obs = g.groupby(["hour", "weekend"])[resp]
            prof[f"{resp}_fitted"] = full.predict(harmonic_X(prof.hour, prof.weekend)).to_numpy()
            idx = pd.MultiIndex.from_frame(prof[["hour", "weekend"]])
            prof[f"{resp}_observed_mean"] = obs.mean().reindex(idx).to_numpy()
            prof[f"{resp}_q25"] = obs.quantile(0.25).reindex(idx).to_numpy()
            prof[f"{resp}_q75"] = obs.quantile(0.75).reindex(idx).to_numpy()
        profiles.append(prof)

    metrics = pd.DataFrame(metrics)
    profiles = pd.concat(profiles, ignore_index=True)
    metrics.to_csv(OUT / "model_metrics.csv", index=False)
    pd.concat(coefs, ignore_index=True).to_csv(OUT / "model_coefficients.csv", index=False)
    profiles.to_csv(OUT / "predicted_profiles.csv", index=False)

    print("\nHold-out performance:")
    print(metrics.pivot_table(index=["room", "response"], columns="model", values="r2").round(3).to_string())
    peaks = (profiles[~profiles.weekend].sort_values("headcount_fitted")
             .groupby("room").tail(1)[["room", "hour", "headcount_fitted"]])
    print("\nWeekday peak (harmonic fit):")
    print(peaks.to_string(index=False))

    plot_profiles(profiles)
    print(f"\nOutputs written to {OUT}")


def plot_profiles(profiles):
    rooms = list(dict.fromkeys(profiles.room))
    panels = [("headcount", "Mean headcount"), ("occupied_share", "Share of hour occupied")]
    fig, axes = plt.subplots(len(rooms), 2, figsize=(11, 2.6 * len(rooms) + 0.8),
                             sharex=True, squeeze=False, facecolor=SURFACE)
    for r, room in enumerate(rooms):
        p = profiles[profiles.room == room]
        for c, (resp, ylabel) in enumerate(panels):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            for weekend, color, name in ((False, WEEKDAY_COLOR, "Weekday"), (True, WEEKEND_COLOR, "Weekend")):
                d = p[p.weekend == weekend]
                ax.fill_between(d.hour, d[f"{resp}_q25"], d[f"{resp}_q75"], color=color, alpha=0.12, lw=0)
                ax.plot(d.hour, d[f"{resp}_fitted"], color=color, lw=2, label=f"{name} — fitted")
                ax.plot(d.hour, d[f"{resp}_observed_mean"], "o", ms=4, color=color, mec=SURFACE, mew=1,
                        label=f"{name} — observed mean")
            ax.set_ylim(bottom=0, top=1 if resp == "occupied_share" else None)
            ax.set_xticks(range(0, 25, 3))
            ax.set_xlim(-0.5, 23.5)
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            if c == 0:
                ax.set_title(room, loc="left", fontsize=10, color=INK, fontweight="bold")
            if r == 0:
                ax.text(0, 1.18, ylabel, transform=ax.transAxes, fontsize=9, color=INK_MUTED)
            if r == len(rooms) - 1:
                ax.set_xlabel("Hour of day (Sydney local time)", fontsize=9, color=INK_MUTED)
    handles, lbls = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper center", ncol=4, frameon=False, fontsize=9, labelcolor=INK)
    fig.text(0.01, 0.005, "Shaded band: interquartile range of hourly values. Room labels from "
             "sensor_locations_occ.csv (sheet marked 'Incorrect Data' — verify).",
             fontsize=7.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(OUT / "occupancy_by_hour.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
