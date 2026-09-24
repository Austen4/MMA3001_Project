# MMA3001_Project

## Setup

```
pip install -r requirements.txt
```

## Occupancy prediction

Predicts the expected number of people in each monitored room, and in all monitored rooms
combined, for a given day and time (Sydney local time).

```
python analysis/occupancy_model.py train                    # fit models (run once, or after new data)
python analysis/occupancy_model.py predict 2026-05-04 14:00  # a specific date and time
python analysis/occupancy_model.py predict Monday 2pm        # a typical Monday across the year
python analysis/occupancy_model.py predict Monday 2pm --month 7
python analysis/occupancy_model.py predict 2026-05-04        # 24-hour profile
```

`train` needs `analysis/output/hourly_occupancy.csv`; if it's missing it is rebuilt from
`data/5occupancySensor_MayToDec2024_9MRows.zip` via `analysis/occupancy_time_regression.py`.

The model currently uses occupancy data only: none of the environmental sensors in `data/`
is located in a monitored room, so environmental readings can't yet be linked to occupancy.
