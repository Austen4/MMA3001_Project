"""
Modelling dataset: binned environmental features joined to the building-total target.

Derivatives are backward finite differences, so every feature at time t uses only data
up to t and the model could run in real time.
"""
import pandas as pd

from . import config

# Calendar columns are kept only so baselines that use time of day can be compared
# against the environmental models on exactly the same rows.
CALENDAR_COLUMNS = ["hour_local", "dow", "weekend"]


def bin_environment(env_clean, sensorid, bin_minutes):
    """Average one sensor's readings over fixed time bins.

    Parameters
    ----------
    env_clean : pandas.DataFrame
        Cleaned readings (see :func:`building_occupancy.env_data.clean_environment`).
    sensorid : int
        Sensor to use.
    bin_minutes : int
        Bin width in minutes.

    Returns
    -------
    pandas.DataFrame
        ``time_utc`` (bin start), the short feature columns of ``config.ENV_COLUMNS``
        (co2 ppm, temp degC, rh %, voc ppb, pm25 ug/m3, pressure mb) and ``n_readings``.
    """
    env = env_clean[env_clean["sensorid"] == sensorid]
    if env.empty:
        raise ValueError(f"no readings for sensor {sensorid}")
    missing = set(config.ENV_COLUMNS) - set(env.columns)
    if missing:
        raise ValueError(f"environmental data is missing columns: {sorted(missing)}")
    t = pd.to_datetime(env["unix_time"], unit="s", utc=True).dt.floor(f"{bin_minutes}min")
    g = env[list(config.ENV_COLUMNS)].rename(columns=config.ENV_COLUMNS).groupby(t)
    out = g.mean()
    out["n_readings"] = g.size()
    return out.rename_axis("time_utc").reset_index()


def add_features(env_binned, bin_minutes):
    """Add finite-difference, lag and calendar features.

    Parameters
    ----------
    env_binned : pandas.DataFrame
        Output of :func:`bin_environment`.
    bin_minutes : int
        Bin width used to build ``env_binned``.

    Returns
    -------
    pandas.DataFrame
        Input columns plus:

        ``dco2_dt``, ``dtemp_dt``, ``drh_dt``
            Backward differences per hour, e.g. ``(co2[t] - co2[t-1]) / dt`` in ppm/h.
            NaN where the previous bin is missing.
        ``dco2_dt_lag1``
            ``dco2_dt`` one bin earlier (NaN across gaps).
        ``hour_local``, ``dow``, ``weekend``
            Local (Sydney) bin-midpoint hour, day of week (0 = Monday) and weekend flag.
    """
    df = env_binned.sort_values("time_utc").reset_index(drop=True)
    step = pd.Timedelta(minutes=bin_minutes)
    gap = df["time_utc"].diff()
    contiguous = gap == step
    dt_h = bin_minutes / 60
    for col in ("co2", "temp", "rh"):
        df[f"d{col}_dt"] = (df[col].diff() / dt_h).where(contiguous)
    # Rate one bin earlier. (Lagged CO2 levels are not used: co2[t-1] = co2[t] - dt * dco2_dt
    # exactly, so they would make the design matrix rank-deficient.)
    df["dco2_dt_lag1"] = df["dco2_dt"].shift(1).where(contiguous)

    local = (df["time_utc"] + step / 2).dt.tz_convert(config.TIMEZONE)
    df["hour_local"] = local.dt.hour + local.dt.minute / 60
    df["dow"] = local.dt.dayofweek
    df["weekend"] = df["dow"] >= 5
    return df


def build_dataset(env_clean, building, sensorid, bin_minutes,
                  required=("co2", "dco2_dt")):
    """Join one sensor's features to the building-total target.

    Parameters
    ----------
    env_clean : pandas.DataFrame
        Cleaned environmental readings.
    building : pandas.DataFrame
        Building total on the same bin grid (see
        :func:`building_occupancy.occupancy_data.building_total`).
    sensorid : int
        Environmental sensor to use.
    bin_minutes : int
        Bin width in minutes.
    required : sequence of str
        Feature columns that must be present; rows missing any are dropped. Other
        features may contain NaN and are imputed by the models that use them.

    Returns
    -------
    pandas.DataFrame
        One row per bin where the sensor and every building room have data, sorted by time.
    """
    feats = add_features(bin_environment(env_clean, sensorid, bin_minutes), bin_minutes)
    df = feats.merge(building, on="time_utc", how="inner")
    df = df.dropna(subset=[*required, config.TARGET]).sort_values("time_utc").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"sensor {sensorid} has no time overlap with the occupancy target")
    df.insert(0, "sensorid", sensorid)
    return df
