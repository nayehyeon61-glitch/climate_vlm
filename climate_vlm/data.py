"""pg-flow input -> training samples.

Input (same files as pg-flow):
    <kai_dir>/<year>.nc            KAI layout, fields (time, 67, lat, lon), ERA5 units, 6-hourly
    <era5_data>/<folder>/...nc     WeatherBench1 files, only for fields outside the KAI file (u10, v10)

``prepare_cache`` copies the selected channels of the high-resolution (HR) grid into one
float32 ``.npy`` per year (memory-mapped during training). The low-resolution (LR) input is the
exact area-weighted average of the HR grid by ``factor`` (pg-flow's finer grids split the 5.625 deg
cells exactly, so factor 2 on 2.8125 deg data gives pg-flow's 5.625 deg input). On a WeatherBench2 grid
(a cell centred on 0 deg, scripts/wb2_to_cache.py) factor 2 averages 1/4, 1/2, 1/4 along longitude instead.

A sample at issue time t (index into a time line that may cross year files inside one split):
    x_lr  (S, C_state, h, w)  normalised LR history at t + history_h      model input
    x_hr  (S, C_state, H, W)  normalised HR history                       labels only (stage 1, current diagnostics)
    y     (L, C_out, H, W)    normalised HR states at t + leads_h         labels only
    x0    (C_out, H, W)       normalised HR analysis at t                 persistence / pg-flow X0 (scoring only)
Future states are never part of the model input.
"""

import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from . import contract

# ----------------------------------------------------------------------------------------------
# cache
# ----------------------------------------------------------------------------------------------


def _read_kai_year(path: str, channels: Sequence[str]):
    """Selected channels (T, C, H, W), orography, hours since 1970, lat, lon of one KAI year file."""
    import netCDF4 as nc
    with nc.Dataset(path) as d:
        d.set_auto_mask(False)
        ch = d.variables.get("channel")
        names = [str(n) for n in ch[:]] if ch is not None and ch.dtype == str else list(contract.KAI_CHANNELS)
        f = np.stack([np.asarray(d["fields"][:, names.index(c)], dtype=np.float32) for c in channels], 1)
        orog = np.asarray(d["fields"][0, names.index("orog")], dtype=np.float32)
        tv = d["time"]
        cal = getattr(tv, "calendar", "standard")
        hours = nc.date2num(nc.num2date(tv[:], tv.units, cal), "hours since 1970-01-01 00:00:00", cal)
        hours = np.rint(np.asarray(hours)).astype(np.int64)
        lat = np.asarray(d["latitude"][:], dtype=np.float64)
        lon = np.asarray(d["longitude"][:], dtype=np.float64)
    if np.any(np.diff(lat) < 0):
        raise ValueError(f"{path}: latitude must be S -> N (pg-flow contract)")
    return f, orog, hours, lat, lon


def _read_extra_year(era5_data: str, name: str, year: int, lat: np.ndarray) -> np.ndarray:
    import glob

    import netCDF4 as nc
    folder = contract.EXTRA_FOLDERS[name]
    files = sorted(glob.glob(os.path.join(era5_data, folder, f"{folder}_{year}_*deg.nc")))
    for p in files:
        with nc.Dataset(p) as d:
            if len(d["lat"]) == len(lat):
                var = [v for v in d.variables if v not in ("time", "lat", "lon", "level")][0]
                x = np.asarray(d[var][:], dtype=np.float32)
                if not np.allclose(np.asarray(d["lat"][:]), lat):
                    raise ValueError(f"{p}: latitudes differ from the KAI file")
                return x
    raise FileNotFoundError(f"no {folder} file for {year} on a {len(lat)}-row grid under {era5_data}")


def prepare_cache(kai_dir: str, out: str, years: Sequence[int], channels: Sequence[str],
                  extras: Sequence[str] = (), era5_data: Optional[str] = None) -> None:
    """KAI yearly NetCDF (+ era5_data extras) -> <out>/<year>.npy (T, C, H, W), <year>_hours.npy, meta.json."""
    os.makedirs(out, exist_ok=True)
    if extras and era5_data is None:
        raise ValueError(f"extras {list(extras)} live in pg-flow's era5_data folder: pass era5_data")
    meta_path = os.path.join(out, "meta.json")
    meta = load_meta(out) if os.path.exists(meta_path) else None
    if meta is not None and meta["channels"] != list(channels) + list(extras):
        raise ValueError(f"{out} holds other channels {meta['channels']}: use a new cache folder")
    for y in years:
        f, orog, hours, lat, lon = _read_kai_year(os.path.join(kai_dir, f"{y}.nc"), channels)
        if extras:
            ex = np.stack([_read_extra_year(era5_data, n, y, lat) for n in extras], 1)
            if len(ex) != len(f):
                raise ValueError(f"{y}: {len(ex)} extra steps vs {len(f)} KAI steps")
            f = np.concatenate([f, ex], 1)
        np.save(os.path.join(out, f"{y}.npy"), f)
        np.save(os.path.join(out, f"{y}_hours.npy"), hours)
        if meta is None:
            step = int(np.diff(hours[:2])[0]) if len(hours) > 1 else 6
            meta = {"channels": list(channels) + list(extras), "state_channels": list(channels),
                    "lat": lat.tolist(), "lon": lon.tolist(), "step_hours": step, "years": []}
            np.save(os.path.join(out, "orog.npy"), orog)
        meta["years"] = sorted(set(meta["years"]) | {int(y)})
    with open(meta_path, "w") as fh:
        json.dump(meta, fh)


def load_meta(cache: str) -> Dict:
    with open(os.path.join(cache, "meta.json")) as fh:
        return json.load(fh)


# ----------------------------------------------------------------------------------------------
# grids
# ----------------------------------------------------------------------------------------------


def lon0_grid(lon) -> bool:
    """True for a WeatherBench2-style grid with a cell centred on 0 deg longitude."""
    return abs(float(np.asarray(lon)[0])) < 1e-6


def coarsen(x: np.ndarray, factor: int, lon0: bool = False) -> np.ndarray:
    """(..., H, W) -> (..., H/f, W/f) area-weighted average.

    lon0=False: the fine cells split the coarse cells (pg-flow --grid files), exact.
    lon0=True: both grids centre a cell on 0 deg (WB2 128x64 -> 64x32), factor 2 only: a coarse cell holds one
    whole fine cell and half of each neighbour, weights 1/4, 1/2, 1/4 along longitude (pg-flow coarsen_ens.py).
    """
    if factor == 1:
        return x
    H, W = x.shape[-2:]
    w = contract.coarse_row_shares(H, factor).astype(np.float32).reshape(H // factor, factor, 1)
    if lon0:
        if factor != 2:
            raise ValueError("a grid centred on 0 deg longitude can only be coarsened by 2")
        xl = 0.25 * np.roll(x, 1, -1) + 0.5 * x + 0.25 * np.roll(x, -1, -1)
        y = xl[..., ::2].reshape(*x.shape[:-2], H // factor, factor, W // factor)
        return (y * w).sum(-2).astype(x.dtype, copy=False)
    y = x.reshape(*x.shape[:-2], H // factor, factor, W // factor, factor).mean(-1)
    return (y * w).sum(-2)


# ----------------------------------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------------------------------


@dataclass
class Normalizer:
    mean: np.ndarray  # (C,)
    std: np.ndarray  # (C,)

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        z = np.load(path)
        return cls(z["mean"].astype(np.float32), z["std"].astype(np.float32))

    def norm(self, x: np.ndarray, idx) -> np.ndarray:
        return (x - self.mean[idx, None, None]) / self.std[idx, None, None]

    def denorm_t(self, x: torch.Tensor, idx, dim: int = -3) -> torch.Tensor:
        shape = [1] * x.dim()
        shape[dim] = -1
        m = torch.as_tensor(self.mean[idx], dtype=x.dtype, device=x.device).view(shape)
        s = torch.as_tensor(self.std[idx], dtype=x.dtype, device=x.device).view(shape)
        return x * s + m


def segments(years: Sequence[int]) -> List[List[int]]:
    """Consecutive years form one time line; a window may cross years only inside one segment."""
    ys = sorted(int(y) for y in years)
    out = [[ys[0]]] if ys else []
    for y in ys[1:]:
        (out[-1].append(y) if y == out[-1][-1] + 1 else out.append([y]))
    return out


class ForecastDataset(Dataset):
    """Windows (history, issue time, leads) over the cache of one split.

    ics: "all" (every ``stride``-th step whose hour is in ``hours``) or "climode" (pg-flow / ClimODE test
    ICs: 6 h index 2 + 8 e of every year, 12 UTC every second day; history may reach into the previous
    year when that year is in the same split).
    context: cached years that may hold the HISTORY of a window (e.g. 2019 for the first 2020 ICs). They never
    give an issue time or a target.
    """

    def __init__(self, cache: str, years: Sequence[int], stats: str, history_h: Sequence[int],
                 leads_h: Sequence[int], factor: int, ics: str = "all", hours: Sequence[int] = (0, 12),
                 stride: int = 1, context: Sequence[int] = ()):
        self.meta = load_meta(cache)
        self.cache, self.factor = cache, factor
        step = self.meta["step_hours"]
        if any(h % step for h in list(history_h) + list(leads_h)):
            raise ValueError(f"history/leads must be multiples of the {step} h step")
        self.hist = [h // step for h in history_h]
        self.leads = [h // step for h in leads_h]
        self.history_h, self.leads_h = list(history_h), list(leads_h)
        self.channels = self.meta["channels"]
        self.state_idx = [self.channels.index(c) for c in self.meta["state_channels"]]
        self.lat = np.asarray(self.meta["lat"])
        self.lon = np.asarray(self.meta["lon"])
        H, W = len(self.lat), len(self.lon)
        if H % factor or W % factor:
            raise ValueError(f"grid {H}x{W} is not divisible by factor {factor}")
        self.lon0 = factor > 1 and lon0_grid(self.lon)
        if self.lon0:
            warnings.warn("HR grid has a cell centred on 0 deg (WeatherBench2 layout): the LR input averages 1/4, 1/2, "
                          "1/4 along longitude, close to but not exactly the 5.625 deg analysis. pg-flow's "
                          "local_era5_to_kai.py --grid writes split-cell grids, for which the average is exact.")
        self.normalizer = Normalizer.load(stats)
        missing = [y for y in list(years) + list(context) if y not in self.meta["years"]]
        if missing:
            raise FileNotFoundError(f"years {missing} are not in the cache {cache}")
        own = {int(y) for y in years}
        self._arr, self._hours = {}, {}
        self.index = []  # (segment years, global t)
        lo, hi = min(self.hist), max(self.leads)
        for seg in segments(sorted(own | {int(y) for y in context})):
            lens = [len(np.load(os.path.join(cache, f"{y}_hours.npy"), mmap_mode="r")) for y in seg]
            offs = np.cumsum([0] + lens)
            year_of = np.repeat(seg, lens)
            for k, y in enumerate(seg):
                if y not in own:
                    continue
                hrs = np.load(os.path.join(cache, f"{y}_hours.npy"))
                if ics == "climode":
                    local = list(range(2, lens[k], 8))
                else:
                    local = [t for t in range(0, lens[k], stride) if int(hrs[t] % 24) in hours]
                for t in local:
                    g = offs[k] + t
                    if g + lo >= 0 and g + hi < offs[-1] and int(year_of[g + hi]) in own:
                        self.index.append((tuple(seg), int(g), int(y), int(t)))
        if not self.index:
            raise RuntimeError("no valid windows: check years, history and leads")

    def _year(self, y):
        if y not in self._arr:
            self._arr[y] = np.load(os.path.join(self.cache, f"{y}.npy"), mmap_mode="r")
            self._hours[y] = np.load(os.path.join(self.cache, f"{y}_hours.npy"))
        return self._arr[y], self._hours[y]

    def _get(self, seg, g):
        for y in seg:
            a, h = self._year(y)
            if g < len(a):
                return a[g], h[g]
            g -= len(a)
        raise IndexError(g)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        seg, g, year, t = self.index[i]
        hist = [self._get(seg, g + o) for o in self.hist]
        fut = [self._get(seg, g + o) for o in self.leads]
        all_idx = list(range(len(self.channels)))
        x_hr = np.stack([self.normalizer.norm(np.asarray(a[self.state_idx]), self.state_idx) for a, _ in hist])
        y = np.stack([self.normalizer.norm(np.asarray(a), all_idx) for a, _ in fut])
        x_lr = coarsen(x_hr, self.factor, self.lon0)
        i0 = self.hist.index(0) if 0 in self.hist else len(self.hist) - 1
        hour0 = int(hist[i0][1])
        x0 = self.normalizer.norm(np.asarray(hist[i0][0]), all_idx)
        when = np.datetime64(hour0, "h").astype("datetime64[D]")
        return {"x_lr": torch.from_numpy(x_lr.astype(np.float32)), "x_hr": torch.from_numpy(x_hr.astype(np.float32)),
                "y": torch.from_numpy(y.astype(np.float32)), "x0": torch.from_numpy(x0.astype(np.float32)),
                "year": year, "t": t, "hour0": hour0,
                "issue": f"{str(when)} {hour0 % 24:02d} UTC"}


def static_fields(cache: str) -> torch.Tensor:
    """(3, H, W) HR static conditioning: normalised orography, sin(lat), cos(lat) cos(lon)."""
    meta = load_meta(cache)
    orog = np.load(os.path.join(cache, "orog.npy")).astype(np.float32)
    lat = np.deg2rad(np.asarray(meta["lat"], dtype=np.float32))
    lon = np.deg2rad(np.asarray(meta["lon"], dtype=np.float32))
    L, G = np.meshgrid(lat, lon, indexing="ij")
    o = (orog - 3800.0) / 8300.0  # KAI orog mean / std (m2 s-2), pg-flow contract
    return torch.from_numpy(np.stack([o, np.sin(L), np.cos(L) * np.cos(G)]).astype(np.float32))
