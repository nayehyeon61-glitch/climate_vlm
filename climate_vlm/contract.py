"""Data contract shared with physical_guide_flowmatching (pg-flow).

pg-flow stores states in the KAI layout: one NetCDF file per year with
``fields (time, 67, lat, lon)`` in ERA5 units, 6-hourly snapshots, latitude S -> N,
equiangular grids without poles (5.625 deg 32x64, 2.8125 deg 64x128, ...).
u10 / v10 are not KAI channels; pg-flow keeps them in the WeatherBench1-style
``era5_data/<folder>/<folder>_<year>_<deg>deg.nc`` files next to the KAI files.

This module copies the parts of ``pgcfm/contract.py`` that this repository needs,
so that channel names, units and constants stay identical.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

LEVELS_HPA = [50, 100, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
UPPER_AIR_VARS = ["z", "q", "t", "u", "v"]
SURFACE_VARS = ["t2m", "sp", "tisr", "tcwv", "orog", "msl", "skt"]
KAI_CHANNELS: List[str] = [f"{v}{p}" for v in UPPER_AIR_VARS for p in LEVELS_HPA] + SURFACE_VARS
KAI_INDEX: Dict[str, int] = {n: i for i, n in enumerate(KAI_CHANNELS)}

# Fields that pg-flow keeps outside the KAI file (WeatherBench1 folders under era5_data).
EXTRA_FOLDERS = {"u10": "10m_u_component_of_wind", "v10": "10m_v_component_of_wind",
                 "z500": "geopotential_500", "t850": "temperature_850", "t2m": "2m_temperature"}

# pg-flow / ClimODE scoring variables, in ClimODE order (pg-flow scripts/run_step11_climode.py VARS).
CLIMODE_VARS = ["z500", "t850", "t2m", "u10", "v10"]

UNITS = {"z": "m2 s-2", "q": "kg kg-1", "t": "K", "u": "m s-1", "v": "m s-1", "t2m": "K", "sp": "Pa",
         "tisr": "J m-2", "tcwv": "kg m-2", "orog": "m2 s-2", "msl": "Pa", "skt": "K", "u10": "m s-1",
         "v10": "m s-1"}


def split_name(name: str):
    """'t850' -> ('t', 850); 't2m' -> ('t2m', None)."""
    if name in UNITS:
        return name, None
    head = name.rstrip("0123456789")
    return head, int(name[len(head):])


def unit_of(name: str) -> str:
    return UNITS[split_name(name)[0]]


@dataclass(frozen=True)
class Constants:
    earth_radius: float = 6.371e6  # m
    gravity: float = 9.80665  # m s-2
    r_dry: float = 287.05  # J kg-1 K-1
    eps_virtual: float = 0.61  # T_v = T (1 + 0.61 q)


CONST = Constants()


def grid_latitudes(n_lat: int) -> np.ndarray:
    """Cell-centre latitudes [deg, S -> N] of an equiangular grid without poles (pg-flow fine grids)."""
    d = 180.0 / n_lat
    return -90.0 + d * (np.arange(n_lat) + 0.5)


def area_weights(lat_deg: np.ndarray) -> np.ndarray:
    """cos(lat) normalised to mean 1 (ClimODE / pg-flow RMSE weight)."""
    w = np.cos(np.deg2rad(np.asarray(lat_deg, dtype=np.float64)))
    return w / w.mean()


def coarse_row_shares(n_lat_fine: int, factor: int) -> np.ndarray:
    """(n_lat_fine // factor, factor) area shares of fine rows inside each coarse row (no poles, S -> N).
    Same as pg-flow scripts/coarsen_ens.py cell_weights."""
    edges = np.deg2rad(np.linspace(-90.0, 90.0, n_lat_fine + 1))
    a = np.diff(np.sin(edges)).reshape(-1, factor)
    return a / a.sum(1, keepdims=True)
