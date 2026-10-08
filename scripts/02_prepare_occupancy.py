"""
Step 2 - occupancy data.

Raw export (data/5occupancySensor_MayToDec2024_9MRows.zip, ~9.1 M events)
  -> data/processed/occupancy_rooms_<m>min.csv   time-weighted occupancy per room
  -> data/processed/building_total_<m>min.csv    building-total target

for every bin width m in config.BIN_MINUTES.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from building_occupancy import config, occupancy_data  # noqa: E402


def main():
    """Run the occupancy preparation step."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    events = occupancy_data.load_occupancy_events(config.RAW_OCC_ZIP, floorspaces=config.BUILDING_ROOMS)
    print(f"events for {events['floorspaceid'].nunique()} rooms: {len(events):,}  "
          f"({time.perf_counter() - t0:.1f} s)")

    for m in config.BIN_MINUTES:
        t0 = time.perf_counter()
        rooms = occupancy_data.bin_occupancy(events, m)
        total = occupancy_data.building_total(rooms)
        rooms.to_csv(config.PROCESSED_DIR / f"occupancy_rooms_{m}min.csv", index=False)
        total.to_csv(config.PROCESSED_DIR / f"building_total_{m}min.csv", index=False)
        print(f"{m:>3} min bins: {len(total):,} building-total rows, "
              f"mean headcount {total[config.TARGET].mean():.2f}  ({time.perf_counter() - t0:.1f} s)")


if __name__ == "__main__":
    main()
