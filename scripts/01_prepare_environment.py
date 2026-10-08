"""
Step 1 - environmental data.

Raw export (data/5EnvSensor_MayToDec2024_180kRows.zip)
  -> data/processed/env_deduplicated.csv   one row per (sensor, timestamp)
  -> data/processed/env_cleaned.csv        stray readings removed, unreliable values flagged

Both outputs are checked against the reference copies committed in data/ to confirm the
cleaning is reproducible from the raw export.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from building_occupancy import config, env_data  # noqa: E402


def main():
    """Run the environmental preparation step."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    raw = env_data.load_raw_environment(config.RAW_ENV_ZIP)
    print(f"raw messages: {len(raw):,}  ({time.perf_counter() - t0:.1f} s)")

    dedup = env_data.deduplicate(raw)
    print(f"unique readings: {len(dedup):,}  conflicting copies: {dedup.attrs['conflicting_readings']}")
    dedup.to_csv(config.ENV_DEDUP_CSV, index=False, encoding="utf-8-sig")

    clean = env_data.clean_environment(dedup)
    print(f"cleaned readings: {len(clean):,}  sensors: {sorted(clean['sensorid'].unique())}")
    clean.to_csv(config.ENV_CLEAN_CSV, index=False, encoding="utf-8-sig")

    for out, ref in ((dedup, config.REFERENCE_ENV_DEDUP_CSV), (clean, config.REFERENCE_ENV_CLEAN_CSV)):
        ok, msg = env_data.matches_reference(out, ref)
        print(f"{'OK ' if ok else 'DIFF'} vs {ref.name}: {msg}")


if __name__ == "__main__":
    main()
