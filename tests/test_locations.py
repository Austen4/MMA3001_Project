"""Tests for building_occupancy.locations."""
import pandas as pd
import pytest

from building_occupancy import locations


@pytest.mark.parametrize("raw, expected", [
    ("G52", "G.52"), ("G.52", "G.52"), ("g55a", "G.55A"), ("1.41A", "1.41A"), (" 1.40 ", "1.40"),
    ("G-PS2", "G-PS02"), ("1-PS01", "1-PS01"), ("GPS05", "G-PS05"),
    ("69", "69"),                      # unrecognised labels are returned unchanged
])
def test_standardise_room(raw, expected):
    assert locations.standardise_room(raw) == expected


def test_load_locations_adds_join_columns(tmp_path):
    path = tmp_path / "loc.xlsx"
    with pd.ExcelWriter(path) as xl:
        pd.DataFrame({"Building": ["69"], "Floor/Room": ["G.52"], "ID": ["1"]}).to_excel(
            xl, sheet_name="ENV", index=False)
        pd.DataFrame({"Building": ["69", "60"], "Floor/Room": ["G52", "x"], "ID": ["a", "b"]}).to_excel(
            xl, sheet_name="OCC", index=False)
    env, occ = locations.load_locations(path)
    assert list(env.columns[:4]) == ["room_key", "Building", "Floor/Room", "room"]
    assert env["room_key"].iloc[0] == occ["room_key"].iloc[0] == "69-G.52"
    assert occ["level"].iloc[0] == "G" and pd.isna(occ["level"].iloc[1])


def test_load_locations_requires_two_sheets(tmp_path):
    path = tmp_path / "one.xlsx"
    pd.DataFrame({"Building": ["1"], "Floor/Room": ["G.01"]}).to_excel(path, index=False)
    with pytest.raises(ValueError):
        locations.load_locations(path)
