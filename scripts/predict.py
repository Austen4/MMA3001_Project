"""
Predict building occupancy from environmental sensor readings.

Usage (from the repository root)::

    python scripts/predict.py examples/example_readings.csv
    python scripts/predict.py my_readings.csv -o my_predictions.csv

Input: CSV with a ``timestamp`` (ISO 8601; no offset = UTC) or ``unix_time`` column and
``co2`` (ppm, required), ``temp`` (degC), ``rh`` (%RH), ``voc`` (ppb), ``pm25`` (ug/m3).
Output: CSV with, per hour, the expected number of people in the monitored rooms from
both final models, a ``status`` (ok / warning / no_prediction) and ``warnings``.
Full specification: ``building_occupancy.predict``.

Exit codes: 0 success, 1 unusable input, 2 trained models missing.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from building_occupancy import config, predict  # noqa: E402


def main(argv=None):
    """Command-line entry point. Returns the process exit code."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="CSV of environmental readings")
    ap.add_argument("-o", "--output", help="output CSV (default: <input>_predictions.csv)")
    ap.add_argument("--model-file", default=config.FINAL_MODEL_FILE, help="trained model bundle")
    args = ap.parse_args(argv)

    try:
        bundle = predict.load_bundle(args.model_file)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    try:
        readings = predict.read_readings(args.input)
    except predict.InputError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    t = readings["unix_time"]
    print(f"read {len(readings):,} valid readings, "
          f"{_fmt(t.min())} to {_fmt(t.max())}")
    for issue, n in readings.attrs["issues"].items():
        print(f"  note: {issue}: {n}")

    out = predict.predict(readings, bundle)
    output = Path(args.output) if args.output else Path(args.input).with_name(Path(args.input).stem + "_predictions.csv")
    out.to_csv(output, index=False)

    card = bundle["card"]
    print(f"\n{len(out)} {card['bin_minutes']}-min periods: "
          + ", ".join(f"{n} {s}" for s, n in out["status"].value_counts().items()))
    for key in predict.FINAL_MODELS:
        col = out[f"{key}_headcount"]
        if col.notna().any():
            peak = out.loc[col.idxmax(), "time_local"]
            print(f"  {card['models'][key]['name']:<30} mean {col.mean():5.1f} people, "
                  f"peak {col.max():5.1f} at {peak:%a %d %b %H:%M}")
    print(f"\npredictions written to {output}")
    print(f"scope: trained on sensor {card['sensor']} ({card['sensor_location']}); "
          f"headcount = monitored rooms only, see models/final_models.json")
    return 0


def _fmt(unix_s):
    import pandas as pd
    return pd.Timestamp(int(unix_s), unit="s", tz="UTC").tz_convert(config.TIMEZONE).strftime("%Y-%m-%d %H:%M %Z")


if __name__ == "__main__":
    sys.exit(main())
