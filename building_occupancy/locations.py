"""
Sensor locations: the supplied workbook -> tables with standardised room labels.

The workbook ``Sensor ID and Locations.xlsx`` has one sheet of environmental sensors and
one of occupancy sensors, which write room numbers differently (``G.52`` vs ``G52``).
Labels are standardised so the two can be joined on building and room. The occupancy
sheet is titled "Incorrect Data" by its authors, so its space ids are indicative only.
"""
import re

import pandas as pd

from . import config

_ROOM = re.compile(r"([G1])[.\s]?(\d{2}[A-Z]?)")   # G52, G.52, 1.40, 1.41A
_PLANT = re.compile(r"([G1])-?PS(\d+)")            # G-PS02, 1-PS01


def standardise_room(label):
    """Convert a room label to the form ``G.52`` / ``1.41A`` / ``G-PS02``.

    Parameters
    ----------
    label : object
        Room label as written in the workbook.

    Returns
    -------
    str
        Standardised label, or the upper-cased input unchanged if it is not recognised.
    """
    text = str(label).strip().upper()
    m = _ROOM.fullmatch(text)
    if m:
        return f"{m[1]}.{m[2]}"
    m = _PLANT.fullmatch(text)
    if m:
        return f"{m[1]}-PS{int(m[2]):02d}"
    return text


def load_locations(path=config.LOCATIONS_XLSX):
    """Read both sheets of the location workbook and add standardised room columns.

    Parameters
    ----------
    path : str or Path
        Workbook with the environmental sheet first and the occupancy sheet second.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        Environmental and occupancy sensor tables, all values as text, each with added
        ``room_key`` (``<building>-<room>``), ``room`` (standardised) and ``level``
        (``G`` or ``1``; empty if the room label is not recognised).
    """
    sheets = pd.read_excel(path, sheet_name=None, dtype=str)
    if len(sheets) < 2:
        raise ValueError("expected an environmental and an occupancy sheet")
    env, occ = (d.dropna(how="all") for d in list(sheets.values())[:2])
    for d in (env, occ):
        if not {"Building", "Floor/Room"} <= set(d.columns):
            raise ValueError("sheet is missing 'Building' or 'Floor/Room' columns")
        d.insert(d.columns.get_loc("Floor/Room") + 1, "room", d["Floor/Room"].map(standardise_room))
        d.insert(d.columns.get_loc("room") + 1, "level", d["room"].str[0].map({"G": "G", "1": "1"}))
        d.insert(0, "room_key", d["Building"] + "-" + d["room"])
    return env, occ
