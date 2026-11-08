"""Public WeatherBench2 ERA5 -> training cache and ClimODE scoring truth, without pg-flow's KAI files.

For machines that cannot read pg-flow's local 0.25 deg ERA5 (e.g. a cloud container): the HR grid is WB2's
128x64 equiangular conservative regrid (2.8125 deg, no poles), written in the cache layout of
``data.prepare_cache`` (same channel names and ERA5 units as the KAI contract). Differences from the pg-flow
``local_era5_to_kai.py --grid 2.8125`` grid: WB2 centres a cell on 0 deg longitude, so an HR cell does not split
the 5.625 deg cells; ``data.coarsen`` then averages 1/4, 1/2, 1/4 along longitude (not exact; pg-flow measured
z500 8 m2/s2, t2m 0.18 K against the 5.625 deg analysis, scripts/coarsen_ens.py).

Scoring truth (``--truth-years``): ClimODE's five variables (z500, t850, t2m, u10, v10) on WB2's 64x32 conservative
regrid, the same zarr pg-flow's ``data/era5_5p625`` and its ClimODE comparison come from:
``<cache>/truth_5p625/<year>.npy`` (T, 5, 32, 64), ``lat.npy``, and ``minmax_2006_2018.json`` (ClimODE's
min-max CRPS units) when every year 2006-2018 is present.

    python scripts/wb2_to_cache.py --config configs/stage01_cpu_small.yaml
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from climate_vlm import build, contract  # noqa: E402

ROOT = "https://storage.googleapis.com/weatherbench2/datasets/era5/"
HR_URL = ROOT + "1959-2022-6h-128x64_equiangular_conservative.zarr"
LR_URL = ROOT + "1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
UPPER = {"z": "geopotential", "t": "temperature", "q": "specific_humidity", "u": "u_component_of_wind",
         "v": "v_component_of_wind"}
SURFACE = {"t2m": "2m_temperature", "msl": "mean_sea_level_pressure", "tcwv": "total_column_water_vapour",
           "sp": "surface_pressure", "u10": "10m_u_component_of_wind", "v10": "10m_v_component_of_wind"}
TRUTH_VARS = ["z500", "t850", "t2m", "u10", "v10"]  # ClimODE / pg-flow order (z, t, t2m, u10, v10)


def open_zarr(url):
    import xarray as xr
    return xr.open_zarr(url, storage_options={"client_kwargs": {"trust_env": True}})


def year_fields(ds, channels, year):
    """(T, C, lat, lon) float32 S -> N, hours since 1970."""
    sl = ds.sel(time=slice(f"{year}-01-01T00", f"{year}-12-31T18"))
    by_var = {}
    for c in channels:
        v, p = contract.split_name(c)
        by_var.setdefault(v, []).append(p)
    got = {}
    for v, levels in by_var.items():
        if v in UPPER:
            x = sl[UPPER[v]].sel(level=[int(p) for p in levels]).transpose("time", "level", "latitude", "longitude")
            x = np.asarray(x.values, dtype=np.float32)
            for k, p in enumerate(levels):
                got[f"{v}{p}"] = x[:, k]
        else:
            got[v] = np.asarray(sl[SURFACE[v]].transpose("time", "latitude", "longitude").values, dtype=np.float32)
    hours = ((sl.time.values - np.datetime64("1970-01-01T00")) // np.timedelta64(1, "h")).astype(np.int64)
    return np.stack([got[c] for c in channels], 1), hours


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--config", required=True)
    a.add_argument("--truth-years", default="2006-2018,2020", help="years of 5.625 deg ClimODE truth")
    a = a.parse_args(argv)
    cfg = build.load_config(a.config)
    d, p = cfg["data"], build.paths(cfg)
    out = p["cache"]
    os.makedirs(out, exist_ok=True)
    channels = list(d["state_channels"]) + list(d.get("extra_channels", []))
    years = sorted(set(build.years(d["train_years"]) + build.years(d["val_years"]) + build.years(d["test_years"])
                       + build.years(d.get("context_years", []))))
    meta_path = os.path.join(out, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else None
    if meta is not None and meta["channels"] != channels:
        raise ValueError(f"{out} holds other channels {meta['channels']}: use a new cache folder")
    ds = open_zarr(HR_URL)
    if meta is None:
        lat, lon = ds.latitude.values.astype(np.float64), ds.longitude.values.astype(np.float64)
        if np.any(np.diff(lat) < 0):
            raise ValueError("expected S -> N latitudes")
        orog = np.asarray(ds["geopotential_at_surface"].transpose("latitude", "longitude").values, np.float32)
        np.save(os.path.join(out, "orog.npy"), orog)
        meta = {"channels": channels, "state_channels": list(d["state_channels"]), "lat": lat.tolist(),
                "lon": lon.tolist(), "step_hours": 6, "years": [], "source": HR_URL}
    for y in years:
        if os.path.exists(os.path.join(out, f"{y}.npy")):
            continue
        t0 = time.time()
        f, hours = year_fields(ds, channels, y)
        if not np.isfinite(f).all():
            raise ValueError(f"{y}: non-finite values in the WB2 fields")
        np.save(os.path.join(out, f"{y}_hours.npy"), hours)
        np.save(os.path.join(out, f"{y}.npy.tmp.npy"), f)
        os.replace(os.path.join(out, f"{y}.npy.tmp.npy"), os.path.join(out, f"{y}.npy"))
        meta["years"] = sorted(set(meta["years"]) | {int(y)})
        with open(meta_path, "w") as fh:
            json.dump(meta, fh)
        print(f"HR {y}: {f.shape} in {time.time() - t0:.0f} s", flush=True)

    tdir = os.path.join(out, "truth_5p625")
    os.makedirs(tdir, exist_ok=True)
    lr = open_zarr(LR_URL)
    np.save(os.path.join(tdir, "lat.npy"), lr.latitude.values.astype(np.float64))
    for y in build.years([s.strip() for s in a.truth_years.split(",") if s.strip()]):
        f = os.path.join(tdir, f"{y}.npy")
        if os.path.exists(f):
            continue
        t0 = time.time()
        x, hours = year_fields(lr, TRUTH_VARS, y)
        np.save(os.path.join(tdir, f"{y}_hours.npy"), hours)
        np.save(f, x)
        print(f"truth 5.625 {y}: {x.shape} in {time.time() - t0:.0f} s", flush=True)
    have = [y for y in range(2006, 2019) if os.path.exists(os.path.join(tdir, f"{y}.npy"))]
    if len(have) == 13:
        lo, hi = np.full(5, np.inf), np.full(5, -np.inf)
        for y in have:
            x = np.load(os.path.join(tdir, f"{y}.npy"), mmap_mode="r")
            lo, hi = np.minimum(lo, x.min((0, 2, 3))), np.maximum(hi, x.max((0, 2, 3)))
        with open(os.path.join(tdir, "minmax_2006_2018.json"), "w") as fh:
            json.dump({"vars": TRUTH_VARS, "max": hi.tolist(), "min": lo.tolist()}, fh)
        print("ClimODE min-max 2006-2018:", dict(zip(TRUTH_VARS, zip(lo.round(2), hi.round(2)))))


if __name__ == "__main__":
    main()
