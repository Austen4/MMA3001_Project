"""
Occupancy prediction: expected headcount in each monitored room, and in all monitored
rooms combined, for any day and time.

Built from the occupancy data only. The environmental data can't contribute yet: no
environmental sensor in the current data sits in a monitored room (see README / analysis
notes), and live sensor readings wouldn't exist for a future time anyway.

Model, one per room (plus one for the combined total), fitted on the most recent
TRAIN_MONTHS of that room's data so it reflects current usage:

    log E[headcount] = day-of-week level + month-of-year level
                       + hour-of-day curve (K sine/cosine pairs, separate shape on weekends)

fitted as a Poisson GLM on hourly time-weighted mean headcount, so predictions are never
negative. Month terms (a proxy for semester/break periods) are only used when the room has
data for all 12 months. The "typical range" shown is the 10th-90th percentile of observed
hourly headcounts for that room, hour and weekday/weekend over the training window.

Usage (all times are Sydney local time):
    python analysis/occupancy_model.py train
    python analysis/occupancy_model.py predict 2026-05-04 14:00
    python analysis/occupancy_model.py predict Monday 2pm
    python analysis/occupancy_model.py predict Monday --month 7
    python analysis/occupancy_model.py predict 2026-05-04            # 24-hour profile

From Python:
    from occupancy_model import predict
    predict("Friday 10:30")
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
HOURLY = OUT / "hourly_occupancy.csv"
MODEL_FILE = OUT / "occupancy_model.json"
VALIDATION_FILE = OUT / "occupancy_model_validation.csv"
TZ = "Australia/Sydney"

TRAIN_MONTHS = 12
TEST_MONTHS = 3
INACTIVE_AFTER_DAYS = 90  # rooms with no data this long before the dataset end are "inactive"
K_HARMONICS = 4
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TOTAL = "All monitored rooms"


# ---------------------------------------------------------------- data

def load_hourly():
    """Hourly time-weighted headcount per room (built by occupancy_time_regression.py)."""
    if not HOURLY.exists():
        sys.path.insert(0, str(HERE))
        import occupancy_time_regression
        occupancy_time_regression.main()
    h = pd.read_csv(HOURLY, usecols=["room", "time_utc", "headcount"], parse_dates=["time_utc"])
    return add_calendar(h)


def add_calendar(h):
    local = h.time_utc.dt.tz_convert(TZ)
    return h.assign(hour=local.dt.hour + 0.5, dow=local.dt.dayofweek, month=local.dt.month)


def total_series(h, rooms):
    """Combined headcount for the given rooms, only for hours where every room reported."""
    wide = h[h.room.isin(rooms)].pivot_table(index="time_utc", columns="room", values="headcount")
    wide = wide.dropna()
    return add_calendar(pd.DataFrame({"room": TOTAL, "time_utc": wide.index, "headcount": wide.sum(axis=1).values}))


# ---------------------------------------------------------------- model

def design(hour, dow, month, use_month):
    hour, dow, month = (np.asarray(x, dtype=float) for x in (hour, dow, month))
    weekend = (dow >= 5).astype(float)
    ang = 2 * np.pi * hour / 24
    X = {"const": np.ones(len(hour))}
    for k in range(1, K_HARMONICS + 1):
        s, c = np.sin(k * ang), np.cos(k * ang)
        X[f"sin{k}"], X[f"cos{k}"] = s, c
        X[f"weekend:sin{k}"], X[f"weekend:cos{k}"] = s * weekend, c * weekend
    for d in range(1, 7):
        X[f"dow{d}"] = (dow == d).astype(float)
    if use_month:
        for m in range(2, 13):
            X[f"month{m}"] = (month == m).astype(float)
    return pd.DataFrame(X)


def fit(df):
    use_month = df.month.nunique() == 12
    X = design(df.hour, df.dow, df.month, use_month)
    res = sm.GLM(df.headcount.to_numpy(), X, family=sm.families.Poisson()).fit()
    # Observed 10th/90th percentiles by weekday/weekend x hour: "all" months, and per month.
    weekend = df.dow >= 5
    hour = df.hour.astype(int)
    q = {"q10": {}, "q90": {}}
    for key, sel in [("all", slice(None))] + [(str(m), df.month == m) for m in sorted(df.month.unique())]:
        d = df[sel]
        for name, p in (("q10", 0.10), ("q90", 0.90)):
            tab = d.headcount.groupby([weekend[sel], hour[sel]]).quantile(p).unstack().reindex(
                index=[False, True], columns=range(24))
            q[name][key] = tab.round(2).values.tolist()  # [weekday[24], weekend[24]], NaN if unseen
    return {"params": res.params.to_dict(), "use_month": use_month, **q}


def expected(entity, hour, dow, month):
    X = design(hour, dow, month, entity["use_month"])
    beta = pd.Series(entity["params"])
    return np.exp(X[beta.index].to_numpy() @ beta.to_numpy())


def last_months(df, months, end=None):
    end = df.time_utc.max() if end is None else end
    return df[(df.time_utc > end - pd.DateOffset(months=months)) & (df.time_utc <= end)]


# ---------------------------------------------------------------- train

def train():
    OUT.mkdir(parents=True, exist_ok=True)
    h = load_hourly()
    data_end = h.time_utc.max()
    spans = h.groupby("room").time_utc.agg(["min", "max"])
    active = spans.index[spans["max"] >= data_end - pd.Timedelta(days=INACTIVE_AFTER_DAYS)].tolist()
    series = {room: g for room, g in h.groupby("room")}
    series[TOTAL] = total_series(h, active)

    # Validation: train on TRAIN_MONTHS, test on the following TEST_MONTHS (the latest data).
    rows = []
    for name, df in series.items():
        test_start = df.time_utc.max() - pd.DateOffset(months=TEST_MONTHS)
        test = df[df.time_utc > test_start]
        tr = last_months(df, TRAIN_MONTHS, end=test_start)
        ent = fit(tr)
        base = tr.groupby([tr.dow >= 5, tr.hour])["headcount"].mean()
        preds = {
            "this model": expected(ent, test.hour, test.dow, test.month),
            "baseline (hour x weekday/weekend mean)": base.reindex(
                pd.MultiIndex.from_arrays([test.dow >= 5, test.hour])).to_numpy(),
        }
        for model, p in preds.items():
            err = test.headcount.to_numpy() - p
            rows.append({"room": name, "model": model, "test_from": f"{test_start:%Y-%m-%d}",
                         "test_hours": len(test), "mae": np.abs(err).mean(),
                         "r2": 1 - (err ** 2).sum() / ((test.headcount - test.headcount.mean()) ** 2).sum()})
    val = pd.DataFrame(rows)
    val.to_csv(VALIDATION_FILE, index=False)

    # Final models on the most recent TRAIN_MONTHS of each series.
    model = {"timezone": TZ, "data_end": f"{data_end:%Y-%m-%d}", "train_months": TRAIN_MONTHS,
             "total_rooms": active, "entities": {}}
    for name, df in series.items():
        tr = last_months(df, TRAIN_MONTHS)
        model["entities"][name] = {
            **fit(tr), "active": name == TOTAL or name in active,
            "data_end": f"{df.time_utc.max():%Y-%m-%d}",
            "train_from": f"{tr.time_utc.min():%Y-%m-%d}",
        }
    MODEL_FILE.write_text(json.dumps(model, indent=1).replace("NaN", "null"))

    print("Validation on the latest", TEST_MONTHS, "months of each room (headcount):")
    print(val.pivot_table(index="room", columns="model", values=["mae", "r2"]).round(2).to_string())
    print(f"\nModel saved to {MODEL_FILE}")
    print("Combined total covers:", ", ".join(active))


# ---------------------------------------------------------------- predict

def parse_when(text, month=None):
    """-> (dow, month or None, hour or None, label). Accepts 'Monday 2pm', '2026-05-04 14:00', ..."""
    tokens = text.replace(",", " ").split()
    abbrev = [d[:3].lower() for d in DAYS]
    day_tok = [t for t in tokens if t[:3].lower() in abbrev and t.isalpha()]
    rest = " ".join(t for t in tokens if t not in day_tok)
    has_time = any(":" in t or t.lower().rstrip(".").endswith(("am", "pm")) for t in tokens)
    if day_tok:
        dow = abbrev.index(day_tok[0][:3].lower())
        hour = None
        if rest:
            ts = pd.Timestamp(rest)
            hour = ts.hour + ts.minute / 60
        label = DAYS[dow] + (f" in {pd.Timestamp(2000, month, 1):%B}" if month else " (typical across the year)")
    else:
        ts = pd.Timestamp(rest)
        dow, month = ts.dayofweek, ts.month
        hour = ts.hour + ts.minute / 60 if has_time else None
        label = f"{DAYS[dow]} {ts:%d %B %Y}"
    if hour is not None:
        label += f", {int(hour):02d}:{round(hour % 1 * 60):02d}"
    return dow, month, hour, label


def load_model():
    if not MODEL_FILE.exists():
        raise SystemExit("No trained model found - run:  python analysis/occupancy_model.py train")
    return json.loads(MODEL_FILE.read_text())


def _expected_any_month(ent, hour, dow, month):
    hour = np.atleast_1d(hour).astype(float)
    if month is not None:
        return expected(ent, hour, np.full(len(hour), dow), np.full(len(hour), month))
    # No month given: average over the 12 months of the year.
    return np.mean([expected(ent, hour, np.full(len(hour), dow), np.full(len(hour), m))
                    for m in range(1, 13)], axis=0)


def predict(when, month=None):
    """Expected headcount per room and combined. One row per room (single time) or a
    24-hour profile (rooms as columns) if no time is given."""
    model = load_model()
    dow, month, hour, label = parse_when(when, month)
    ents = model["entities"]
    names = [n for n in ents if n != TOTAL] + [TOTAL]
    if hour is None:
        hours = np.arange(24)
        prof = pd.DataFrame({n: _expected_any_month(ents[n], hours + 0.5, dow, month) for n in names},
                            index=pd.Index([f"{h:02d}:00" for h in hours], name="hour"))
        prof[TOTAL] = prof[model["total_rooms"]].sum(axis=1)
        prof.attrs["label"] = label
        return prof
    rows = []
    wk, hr = int(dow >= 5), int(hour) % 24
    for n in names:
        e = ents[n]
        key = str(month) if month is not None and str(month) in e["q10"] else "all"
        low, high = e["q10"][key][wk][hr], e["q90"][key][wk][hr]
        if low is None or high is None:
            key = "all"
            low, high = e["q10"][key][wk][hr], e["q90"][key][wk][hr]
        rows.append({"room": n, "expected_people": float(_expected_any_month(e, hour, dow, month)[0]),
                     "range_low": low, "range_high": high,
                     "range_basis": "same month" if key != "all" else "all months",
                     "note": "" if e["active"] else f"no data since {e['data_end']}; not in total"})
    out = pd.DataFrame(rows).set_index("room")
    out.loc[TOTAL, "expected_people"] = out.loc[model["total_rooms"], "expected_people"].sum()
    out.attrs["label"] = label
    return out


def print_prediction(when, month=None):
    res = predict(when, month)
    print(f"\nPredicted occupancy - {res.attrs['label']} (Sydney time)\n")
    if "expected_people" in res:
        for room, r in res.iterrows():
            if room == TOTAL:
                print("-" * 78)
            rng = f"{r.range_low:.0f}-{r.range_high:.0f}"
            print(f"{room:<24} {r.expected_people:6.1f} people   typical range {rng:>6}  ({r.range_basis})   {r.note}")
        print("\nExpected = average number of people present over that hour."
              "\nTypical range = 10th-90th percentile of observed hourly values at this hour and day type"
              "\n(weekday/weekend) in the training window, for the same month where data exists.")
    else:
        print(res.round(1).to_string())
        print("\nValues = expected average number of people present during each hour.")
    model = load_model()
    print(f"'{TOTAL}' = sum of {len(model['total_rooms'])} active rooms, not the whole building.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train", help="fit models from analysis/output/hourly_occupancy.csv")
    p = sub.add_parser("predict", help="predict headcount for a day and time")
    p.add_argument("when", nargs="+", help="e.g. 2026-05-04 14:00 | Monday 2pm | Friday")
    p.add_argument("--month", type=int, choices=range(1, 13), help="month (1-12) when giving a weekday")
    args = ap.parse_args()
    if args.cmd == "train":
        train()
    else:
        print_prediction(" ".join(args.when), args.month)


if __name__ == "__main__":
    main()
