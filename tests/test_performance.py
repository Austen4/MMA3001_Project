"""Tests for the optimised timestamp parser and building_occupancy.performance."""
import numpy as np
import pandas as pd
import pytest

from building_occupancy import occupancy_data as od
from building_occupancy import performance

RNG = np.random.default_rng(11)


# ---------------------------------------------------------------- fixed-offset parser

def test_fixed_offset_parser_matches_pandas_general_parser():
    s = pd.Series(["2023-11-24 00:36:52+11", "2024-05-01 09:00:00+10", "2024-01-01 12:00:00-05"])
    fast = od.parse_offset_timestamps(s)
    ref = pd.to_datetime(s, utc=True, format="mixed")
    assert (fast.to_numpy() == ref.to_numpy()).all()
    assert fast.iloc[0] == pd.Timestamp("2023-11-23 13:36:52", tz="UTC")     # +11 h -> subtract
    assert fast.iloc[2] == pd.Timestamp("2024-01-01 17:00:00", tz="UTC")     # -05 h -> add


def test_fixed_offset_parser_falls_back_for_other_formats():
    s = pd.Series(["2024-05-01T09:00:00Z", "2024-05-01 09:00:00+10:00"])
    out = od.parse_offset_timestamps(s)
    assert list(out) == list(pd.to_datetime(s, utc=True, format="mixed"))


def test_fixed_offset_parser_dst_change_is_unambiguous():
    # 02:30 local occurs twice when daylight saving ends; the explicit offset disambiguates.
    s = pd.Series(["2024-04-07 02:30:00+11", "2024-04-07 02:30:00+10"])
    out = od.parse_offset_timestamps(s)
    assert out.iloc[1] - out.iloc[0] == pd.Timedelta(hours=1)


def test_loaders_agree_on_a_small_export(tmp_path):
    path = tmp_path / "occ.csv"
    pd.DataFrame({
        "id": [1, 2, 3, 4], "floorspaceid": ["a", "a", "b", "a"],
        "occupancystatus": ["CurrentlyOccupied", "NotOccupied", "CurrentlyOccupied", "NotOccupied"],
        "headcount": [2, 0, 1, 0],
        "collecteddate": ["2024-03-04 09:00:00+11", "2024-03-04 09:30:00+11", "2024-03-04 09:00:00+11",
                          "2024-03-04 09:30:00+11"],
    }).to_csv(path, index=False)
    pd.testing.assert_frame_equal(od.load_occupancy_events(path),
                                  performance.load_occupancy_events_categorical(path))


# ---------------------------------------------------------------- reference binning

def test_vectorised_binning_matches_python_loop():
    n = 2000
    t = np.cumsum(RNG.integers(1, 2400, size=n))
    status = RNG.choice([od.OCCUPIED, od.NOT_OCCUPIED, "RecentlyOccupied"], size=n)
    hc = np.where(status == od.OCCUPIED, RNG.integers(1, 10, size=n), 0).astype(float)
    for bin_s in (600, 3600):
        vec = od.bin_state_intervals(t, status, hc, bin_s)
        loop = performance.bin_state_intervals_loop(t, status, hc, bin_s)
        assert list(vec["bin_start_s"]) == list(loop["bin_start_s"])
        for col in ("coverage", "headcount", "occupied_share"):
            assert np.allclose(vec[col], loop[col])


# ---------------------------------------------------------------- measurement helpers

def test_profile_call_and_self_time():
    result, table, text = performance.profile_call(sorted, list(range(1000, 0, -1)), top=5)
    assert result[0] == 1
    assert {"function", "ncalls", "tottime_s", "cumtime_s"} <= set(table.columns)
    assert "function calls" in text
    assert performance.self_time(table, r"sorted") >= 0
    assert performance.self_time(table, r"no_such_function") == 0


def test_memory_readings_are_positive_and_ordered():
    assert performance.current_rss_bytes() > 0
    assert performance.peak_rss_bytes() >= performance.current_rss_bytes() * 0.5


def test_run_isolated_returns_time_memory_and_result():
    r = performance.run_isolated(performance.inner_fold_sizes, 60, 5)
    assert r["result"] == [10, 20, 30, 40, 50]
    assert r["seconds"] >= 0 and r["peak_rss_mb"] >= r["baseline_rss_mb"] > 0


# ---------------------------------------------------------------- operation counts

def test_flop_formulas():
    assert performance.lstsq_cholesky_flops(100, 3) == 100 * 9 + 9
    assert performance.thin_svd_flops(100, 3) == 6 * 100 * 9 + 20 * 27
    assert performance.inner_fold_sizes(60, 5) == [10, 20, 30, 40, 50]
    folds = [10, 20, 30, 40, 50]
    assert performance.ridge_fit_flops(60, 3, 4, 5) == pytest.approx(
        4 * sum(m * 9 + 9 for m in folds) + 60 * 9 + 9)
    assert performance.svd_path_fit_flops(60, 3, 5) == pytest.approx(
        sum(6 * m * 9 + 540 for m in folds) + 6 * 60 * 9 + 540)
    assert performance.gram_arithmetic_intensity(16) == 2
