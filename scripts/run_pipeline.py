"""
Run the full workflow in order: raw data -> processed data -> datasets -> models.

Usage (from the repository root)::

    python scripts/run_pipeline.py            # all steps
    python scripts/run_pipeline.py 3 4        # only steps 3 and 4
"""
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = {
    "1": "01_prepare_environment.py",
    "2": "02_prepare_occupancy.py",
    "3": "03_build_datasets.py",
    "4": "04_train_linear_models.py",
}


def main(argv):
    """Run the requested steps (default: all)."""
    selected = argv or list(STEPS)
    unknown = [s for s in selected if s not in STEPS]
    if unknown:
        raise SystemExit(f"unknown step(s) {unknown}; choose from {list(STEPS)}")
    for step in selected:
        print(f"\n=== Step {step}: {STEPS[step]} ===")
        t0 = time.perf_counter()
        runpy.run_path(str(HERE / STEPS[step]), run_name="__main__")
        print(f"--- step {step} finished in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main(sys.argv[1:])
