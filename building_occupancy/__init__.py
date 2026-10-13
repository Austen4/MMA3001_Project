"""
.. include:: ../README.md

# API reference

The `building_occupancy` package (docstrings in NumPy style):

| Module | Purpose |
|---|---|
| `building_occupancy.config` | paths, sensor/room identifiers and every tunable setting |
| `building_occupancy.locations` | sensor location workbook -> standardised room labels |
| `building_occupancy.env_data` | raw environmental export -> deduplicated -> cleaned readings |
| `building_occupancy.occupancy_data` | occupancy event stream -> fixed time bins -> building total |
| `building_occupancy.dataset` | binned environmental features joined to the building-total target |
| `building_occupancy.models` | least-squares solvers (QR, normal equations, SVD ridge path), CO2 mass balance, ridge, polynomial ridge, baselines |
| `building_occupancy.evaluation` | chronological split, rolling-origin cross-validation, metrics |
| `building_occupancy.plotting` | shared figure style and prediction plots |
| `building_occupancy.predict` | final models: training, saving, input validation and prediction |

The pipeline scripts in `scripts/` and the prediction tool `scripts/predict.py` call these
modules; their usage is described above.
"""
