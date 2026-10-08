"""Tests for building_occupancy.occupancy_data."""
import numpy as np
import pandas as pd
import pytest

from building_occupancy import occupancy_data as od

OCC, NOT = od.OCCUPIED, od.NOT_OCCUPIED


def test_time_weighted_mean_within_bin():
    # 2 people for 30 min then empty for 30 min -> mean 1.0, occupied half the hour.
    out = od.bin_state_intervals([0, 1800, 7200], [OCC, NOT, OCC], [2, 0, 1], bin_s=3600)
    assert list(out["bin_start_s"]) == [0, 3600]          # last event has no end -> no bin 7200
    assert out.loc[0, "headcount"] == pytest.approx(1.0)
    assert out.loc[0, "occupied_share"] == pytest.approx(0.5)
    assert out.loc[0, "coverage"] == pytest.approx(1.0)
    assert out.loc[1, "headcount"] == 0 and out.loc[1, "coverage"] == pytest.approx(1.0)


def test_interval_split_across_bin_boundary():
    # State 3 people from 1800 to 5400 spans two hourly bins, half in each.
    out = od.bin_state_intervals([1800, 5400], [OCC, NOT], [3, 0], bin_s=3600)
    assert list(out["coverage"]) == pytest.approx([0.5, 0.5])
    assert list(out["headcount"]) == pytest.approx([3, 3])


def test_occupied_state_not_carried_beyond_hold_limit():
    out = od.bin_state_intervals([0, 3 * 3600], [OCC, NOT], [1, 0], bin_s=3600,
                                 hold_occupied_s=3600, hold_not_occupied_s=72 * 3600)
    assert list(out["bin_start_s"]) == [0]                # hours 1-2 are missing, not occupied


def test_empty_state_carried_through_quiet_night():
    out = od.bin_state_intervals([0, 8 * 3600], [NOT, OCC], [0, 1], bin_s=3600)
    assert len(out) == 8 and (out["coverage"] == 1).all() and (out["headcount"] == 0).all()


@pytest.mark.parametrize("kwargs", [dict(t_s=[10, 0], status=[OCC, NOT], headcount=[1, 0], bin_s=60),
                                    dict(t_s=[0, 10], status=[OCC, NOT], headcount=[1, 0], bin_s=0),
                                    dict(t_s=[0, 10], status=[OCC], headcount=[1, 0], bin_s=60)])
def test_invalid_inputs_rejected(kwargs):
    with pytest.raises(ValueError):
        od.bin_state_intervals(**kwargs)


def _binned():
    t = pd.to_datetime([0, 3600], unit="s", utc=True)
    return pd.DataFrame({
        "floorspaceid": ["a", "a", "b", "b"], "time_utc": list(t) * 2,
        "coverage": [1.0, 1.0, 1.0, 0.2], "headcount": [1.0, 2.0, 3.0, 4.0], "occupied_share": [1, 1, 0.5, 1],
    })


def test_building_total_sums_rooms_and_requires_full_coverage():
    out = od.building_total(_binned(), rooms=["a", "b"], min_coverage=0.5)
    assert len(out) == 1                                   # room b under-covered in second bin
    assert out.loc[0, "building_headcount"] == pytest.approx(4.0)
    assert out.loc[0, "rooms_occupied"] == pytest.approx(1.5)


def test_building_total_unknown_room():
    with pytest.raises(ValueError):
        od.building_total(_binned(), rooms=["a", "zzz"])


def test_bin_occupancy_rejects_misaligned_bins():
    with pytest.raises(ValueError):
        od.bin_occupancy(pd.DataFrame(), bin_minutes=7)


def test_bin_occupancy_end_to_end():
    t = pd.to_datetime([0, 1800, 3600], unit="s", utc=True)
    events = pd.DataFrame({"floorspaceid": ["a"] * 3, "occupancystatus": [OCC, NOT, OCC],
                           "headcount": [2.0, 0.0, 1.0], "t": t})
    out = od.bin_occupancy(events, 30)
    assert list(out["headcount"]) == [2.0, 0.0]
    assert out["time_utc"].iloc[1] == pd.Timestamp(1800, unit="s", tz="UTC")
    assert np.issubdtype(out["coverage"].dtype, np.floating)
