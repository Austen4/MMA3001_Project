"""
Step 6 - train and save the final models, and demonstrate the prediction tool.

data/processed/dataset_<FINAL_SENSOR>_<FINAL_BIN_MINUTES>min.csv
  -> models/final_models.joblib       fitted Ridge (environment + time) and polynomial ridge (degree 2)
  -> models/final_models.json         model card: data, settings, hold-out metrics, training ranges
  -> examples/example_readings.csv    one week of raw readings from the final sensor (tool input example)
  -> results/final_models/            example predictions vs observed headcount (table and figure)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from building_occupancy import config, plotting, predict  # noqa: E402

OUT = config.RESULTS_DIR / "final_models"
EXAMPLE_INPUT = config.ROOT / "examples" / "example_readings.csv"
EXAMPLE_WEEK_UTC = ("2024-05-19 14:00", "2024-05-26 14:00")   # Mon 20 - Sun 26 May 2024, Sydney time


def write_example_input():
    """Export one week of the final sensor's cleaned readings in the tool's input format."""
    env = pd.read_csv(config.ENV_CLEAN_CSV, encoding="utf-8-sig")
    env = env[env["sensorid"] == config.FINAL_SENSOR]
    t = pd.to_datetime(env["unix_time"], unit="s", utc=True)
    week = env[(t >= pd.Timestamp(EXAMPLE_WEEK_UTC[0], tz="UTC")) & (t < pd.Timestamp(EXAMPLE_WEEK_UTC[1], tz="UTC"))]
    cols = {long: short for short, (long, _, _) in config.PREDICTION_INPUTS.items()}
    example = week[list(cols)].rename(columns=cols)
    example.insert(0, "timestamp", pd.to_datetime(week["unix_time"], unit="s", utc=True)
                   .dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy())
    EXAMPLE_INPUT.parent.mkdir(parents=True, exist_ok=True)
    example.to_csv(EXAMPLE_INPUT, index=False)
    return len(example)


def main():
    """Train, save, and run the prediction tool on the example week."""
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.PROCESSED_DIR / f"dataset_{config.FINAL_SENSOR}_{config.FINAL_BIN_MINUTES}min.csv",
                     parse_dates=["time_utc"])
    bundle = predict.train_final_models(df)
    predict.save_bundle(bundle)
    for key, entry in bundle["card"]["models"].items():
        m = entry["holdout_metrics"]
        print(f"{entry['name']:<30} alpha={entry['alpha']:.3g}  coefficients={entry['n_coefficients']:>3}  "
              f"hold-out MAE {m['mae']:.2f}, R2 {m['r2']:.2f}")
    print(f"saved {config.FINAL_MODEL_FILE.relative_to(config.ROOT)} and {config.FINAL_MODEL_CARD.name}")

    n = write_example_input()
    print(f"example input: {n} readings -> {EXAMPLE_INPUT.relative_to(config.ROOT)}")

    # Demonstration: run the tool exactly as a user would, then compare with observed headcount.
    readings = predict.read_readings(EXAMPLE_INPUT)
    pred = predict.predict(readings, predict.load_bundle())
    building = pd.read_csv(config.PROCESSED_DIR / f"building_total_{config.FINAL_BIN_MINUTES}min.csv",
                           parse_dates=["time_utc"])
    pred = pred.merge(building[["time_utc", config.TARGET]], on="time_utc", how="left")
    pred.to_csv(OUT / "example_week_predictions.csv", index=False)
    print(pred["status"].value_counts().to_string())

    shown = pred.dropna(subset=[config.TARGET, *[f"{k}_headcount" for k in predict.FINAL_MODELS]])
    plotting.plot_test_period(
        shown, {bundle["card"]["models"][k]["name"]: shown[f"{k}_headcount"].to_numpy() for k in predict.FINAL_MODELS},
        OUT / "example_week_predictions.png",
        "Final models applied with scripts/predict.py to examples/example_readings.csv (Mon 20 - Sun 26 May 2024). "
        "Observed headcount in black. Note: this week lies inside the final models' training period, so it "
        "demonstrates the tool, not accuracy - see models/final_models.json for hold-out metrics.", days=7,
        metric_label="in-sample")
    print(f"results written to {OUT}")


if __name__ == "__main__":
    main()
