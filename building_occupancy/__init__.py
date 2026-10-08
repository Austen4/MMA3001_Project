"""
Building occupancy estimation from indoor environmental sensor data.

Modules
-------
config
    Paths, sensor/room identifiers and every tunable setting in one place.
env_data
    Raw environmental sensor export -> deduplicated readings -> cleaned readings.
occupancy_data
    Event-driven occupancy export -> fixed-interval time-weighted headcounts and a
    building-total target.
dataset
    Joins binned environmental features to the building-total target.
models
    Linear least-squares solvers (QR and normal equations), the CO2 mass-balance
    regression, ridge regression and the baseline models they are compared against.
evaluation
    Chronological splitting, rolling-origin cross-validation and error metrics.
"""
