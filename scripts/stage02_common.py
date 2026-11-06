"""Pieces shared by the stage-02 scripts: experts, periods, the routing domain and its sub-regions.

Splits follow the preprocessing guide (sec. 7): A predictor and experts, B router training, C probability
calibration, D final test. A window (history .. last lead) must lie inside its own period, with ``margin_h`` hours
kept free at both ends, so no sample crosses a split boundary.
"""

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from climate_vlm import build, data

EXPERTS = ["baseline", "advection", "moisture"]  # option names of the guide (sec. 8); expert 0 is the base forecast
G = 9.80665


def area_w(n_lat: int) -> np.ndarray:
    """(n_lat, 1) cos-latitude weights of a regular S -> N grid, mean 1."""
    lat = np.linspace(-90 + 90 / n_lat, 90 - 90 / n_lat, n_lat)
    w = np.cos(np.deg2rad(lat))
    return (w / w.mean())[:, None].astype(np.float32)


def period_dataset(base_cfg: Dict, spec: Dict, leads: Sequence[int]) -> data.ForecastDataset:
    """Issue times of one period whose whole window stays inside the period minus ``margin_h`` at both ends."""
    d, p = base_cfg["data"], build.paths(base_cfg)
    ds = data.ForecastDataset(p["cache"], spec["years"], p["stats"], d["history_h"], list(leads), d["factor"],
                              ics="all", hours=spec.get("hours", [0, 12]), stride=spec.get("stride", 1))
    hours = {y: np.load(os.path.join(p["cache"], f"{y}_hours.npy")) for y in spec["years"]}
    start, end = min(int(h[0]) for h in hours.values()), max(int(h[-1]) for h in hours.values())
    if "dates" in spec:  # optional [first day, last day] inside the years
        hr = lambda s: int(np.datetime64(str(s), "h").astype(np.int64))  # noqa: E731
        start, end = max(start, hr(spec["dates"][0])), min(end, hr(spec["dates"][1]) + 18)
    m = int(spec.get("margin_h", 0))
    lo, hi = min(d["history_h"]), max(leads)
    ds.index = [ix for ix in ds.index
                if int(hours[ix[2]][ix[3]]) + lo >= start + m and int(hours[ix[2]][ix[3]]) + hi <= end - m]
    if not ds.index:
        raise RuntimeError(f"no window of {spec} fits inside the period")
    return ds


def issue_hour(ds: data.ForecastDataset, ix) -> int:
    return int(np.load(os.path.join(ds.cache, f"{ix[2]}_hours.npy"), mmap_mode="r")[ix[3]])


def cells(centres: Sequence[float], lo: float, hi: float, periodic: bool = False) -> np.ndarray:
    """Indices of the cells whose centre lies in [lo, hi) (longitudes taken modulo 360)."""
    c = np.asarray(centres, np.float64)
    if periodic:
        c, lo, hi = c % 360, lo % 360, hi % 360 if hi % 360 else 360.0
    idx = np.where((c >= lo - 1e-9) & (c < hi - 1e-9))[0]
    if not len(idx):
        raise ValueError(f"no cell centre in [{lo}, {hi})")
    return idx


def lr_grid(meta: Dict, factor: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cell centres of the coarse (input / truth) grid that data.coarsen produces from the cache grid."""
    lat, lon = np.asarray(meta["lat"], np.float64), np.asarray(meta["lon"], np.float64)
    lon_lr = lon[::factor] if data.lon0_grid(lon) else lon.reshape(-1, factor).mean(1)
    return lat.reshape(-1, factor).mean(1), lon_lr


def domain(lat: Sequence[float], lon: Sequence[float], spec: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Row and column indices of the routing domain on a grid (lat S -> N)."""
    return cells(lat, *spec["lat"]), cells(lon, *spec["lon"], periodic=True)


def sub_regions(n_rows: int, n_cols: int, sub: Sequence[int]) -> List[Tuple[str, slice, slice]]:
    """Sub-regions of a domain crop in reading order from the north-west (rows S -> N), named NW, NE, SW, SE for 2 x 2."""
    a, b = sub
    if n_rows % a or n_cols % b:
        raise ValueError(f"a {n_rows} x {n_cols} domain does not split into {a} x {b} sub-regions")
    h, w = n_rows // a, n_cols // b
    names = {(2, 2): ["NW", "NE", "SW", "SE"]}.get((a, b))
    out = []
    for i in range(a):  # i = 0 is the northern band
        rows = slice(n_rows - (i + 1) * h, n_rows - i * h)
        for j in range(b):
            k = len(out)
            out.append((names[k] if names else f"s{k}", rows, slice(j * w, (j + 1) * w)))
    return out


def domain_label(spec: Dict) -> str:
    f = lambda x: f"{abs(x):.0f}{'N' if x >= 0 else 'S'}"  # noqa: E731
    g = lambda x: f"{x % 360:.0f}E"  # noqa: E731
    return f"{spec['name']} {f(spec['lat'][0])}-{f(spec['lat'][1])} {g(spec['lon'][0])}-{g(spec['lon'][1])}"


def wquantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """Area-weighted quantile of the finite values of x."""
    ok = np.isfinite(x)
    if not ok.any():
        return float("nan")
    x, w = x[ok], w[ok]
    o = np.argsort(x)
    c = np.cumsum(w[o]) - 0.5 * w[o]
    return float(np.interp(q * w.sum(), c, x[o]))
