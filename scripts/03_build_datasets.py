"""
Step 3 - modelling datasets.

data/processed/env_cleaned.csv + data/processed/building_total_<m>min.csv
  -> data/processed/dataset_<sensor>_<m>min.csv

one dataset per environmental sensor in config.ENV_SENSORS and bin width in config.BIN_MINUTES.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from building_occupancy import config, dataset  # noqa: E402


def main():
    """Run the dataset-building step."""
    env = pd.read_csv(config.ENV_CLEAN_CSV, encoding="utf-8-sig")
    for m in config.BIN_MINUTES:
        building = pd.read_csv(config.PROCESSED_DIR / f"building_total_{m}min.csv", parse_dates=["time_utc"])
        for sensor, where in config.ENV_SENSORS.items():
            df = dataset.build_dataset(env, building, sensor, m)
            df.to_csv(config.PROCESSED_DIR / f"dataset_{sensor}_{m}min.csv", index=False)
            print(f"sensor {sensor} ({where}), {m:>3} min: {len(df):,} rows "
                  f"{df['time_utc'].min():%Y-%m-%d} to {df['time_utc'].max():%Y-%m-%d}")


if __name__ == "__main__":
    main()
