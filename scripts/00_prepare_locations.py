"""
Step 0 - sensor locations.

data/Sensor ID and Locations.xlsx
  -> data/processed/sensor_locations_env.csv
  -> data/processed/sensor_locations_occ.csv

with standardised room labels so environmental and occupancy sensors can be matched by
building and room (used to choose and document the sensors and rooms in config.py).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from building_occupancy import config, locations  # noqa: E402


def main():
    """Run the location preparation step."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    env, occ = locations.load_locations(config.LOCATIONS_XLSX)
    env.to_csv(config.ENV_LOCATIONS_CSV, index=False)
    occ.to_csv(config.OCC_LOCATIONS_CSV, index=False)
    shared = sorted(set(env["room_key"]) & set(occ["room_key"]))
    print(f"environmental sensors: {len(env)}, occupancy sensors: {len(occ)}, "
          f"rooms with both: {len(shared)}")
    for sensor in config.ENV_SENSORS:
        row = env[env["ID"] == str(sensor)]
        where = row["room_key"].iloc[0] + " " + row["Area"].iloc[0] if len(row) else "not in workbook"
        print(f"  modelled sensor {sensor}: {where}")


if __name__ == "__main__":
    main()
