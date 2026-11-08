"""Small synthetic files in pg-flow's layout, for tests and smoke runs only (no skill can be learned from them).

    <root>/kai6h/<year>.nc         fields (time, 67, lat, lon), 6-hourly, KAI channel names, lat S -> N, no poles
    <root>/era5_data/10m_{u,v}_component_of_wind/..._<year>_<deg>deg.nc

Temperature waves travel east; geopotential is integrated hydrostatically from the virtual temperature, so the
hydrostatic residual of the truth is small, as in ERA5.

    python scripts/make_synthetic.py --out data/synthetic --years 2015 2016 2017 --days 20 --nlat 32
"""

import argparse
import os
import sys

import netCDF4 as nc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from climate_vlm import contract  # noqa: E402

G, RD = contract.CONST.gravity, contract.CONST.r_dry


def make_year(year, days, H, W, rng):
    lat = contract.grid_latitudes(H)
    lon = 360.0 / W * (np.arange(W) - 0.5)  # cells split the cells of the half-resolution grid (pg-flow --grid)
    T = days * 4
    t = np.arange(T)[:, None, None] * 0.25 + (year - 2000) * 365  # days
    la, lo = np.deg2rad(lat)[None, :, None], np.deg2rad(lon)[None, None, :]
    phase = rng.uniform(0, 2 * np.pi, 3)
    wave = (np.sin(3 * lo - 0.6 * t + phase[0]) * np.cos(la) ** 2
            + 0.5 * np.sin(5 * lo - 0.9 * t + phase[1]) * np.sin(2 * la) + 0.3 * np.cos(2 * lo + 0.2 * t + phase[2]))
    f = np.zeros((T, contract.KAI_INDEX["skt"] + 1, H, W), dtype=np.float64)
    p = np.array(contract.LEVELS_HPA, dtype=np.float64)
    temp = {k: 288.0 * (pk / 1000.0) ** 0.19 - 30 * np.sin(la) ** 2 + 4.0 * (pk / 1000.0) * wave
            for k, pk in enumerate(p)}
    hum = {k: 0.012 * (pk / 1000.0) ** 3 * np.cos(la) ** 2 * (1.2 + 0.3 * wave) for k, pk in enumerate(p)}
    phi = {len(p) - 1: G * 110.0 + 400.0 * wave}
    for k in range(len(p) - 2, -1, -1):  # upward from 1000 hPa
        tv = 0.5 * (temp[k] * (1 + 0.61 * hum[k]) + temp[k + 1] * (1 + 0.61 * hum[k + 1]))
        phi[k] = phi[k + 1] + RD * tv * np.log(p[k + 1] / p[k])
    for k, pk in enumerate(p):
        f[:, contract.KAI_INDEX[f"z{int(pk)}"]] = phi[k]
        f[:, contract.KAI_INDEX[f"t{int(pk)}"]] = temp[k]
        f[:, contract.KAI_INDEX[f"q{int(pk)}"]] = hum[k]
        f[:, contract.KAI_INDEX[f"u{int(pk)}"]] = 8.0 + 12.0 * np.cos(la) * (pk / 1000.0) ** -0.5 * (1 + 0.2 * wave)
        f[:, contract.KAI_INDEX[f"v{int(pk)}"]] = 6.0 * np.cos(3 * lo - 0.6 * t + phase[0]) * np.cos(la)
    f[:, contract.KAI_INDEX["t2m"]] = temp[len(p) - 1] - 2.0
    f[:, contract.KAI_INDEX["sp"]] = 1.0e5 - 500.0 * np.cos(2 * lo) * np.cos(la)
    f[:, contract.KAI_INDEX["tisr"]] = 1.2e6 * np.clip(np.cos(la - 0.4 * np.sin(2 * np.pi * t / 365)), 0, None)
    f[:, contract.KAI_INDEX["tcwv"]] = 40.0 * np.cos(la) ** 2 * (1 + 0.2 * wave)
    f[:, contract.KAI_INDEX["orog"]] = G * 500.0 * np.clip(np.cos(2 * lo) * np.cos(la), 0, None)
    f[:, contract.KAI_INDEX["msl"]] = 1.013e5 + 800.0 * wave
    f[:, contract.KAI_INDEX["skt"]] = temp[len(p) - 1] - 1.0
    u10 = 0.7 * f[:, contract.KAI_INDEX["u1000"]]
    v10 = 0.7 * f[:, contract.KAI_INDEX["v1000"]]
    return f.astype(np.float32), u10.astype(np.float32), v10.astype(np.float32), lat, lon


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--out", required=True)
    a.add_argument("--years", type=int, nargs="+", default=[2015, 2016, 2017])
    a.add_argument("--days", type=int, default=20)
    a.add_argument("--nlat", type=int, default=32)
    a.add_argument("--seed", type=int, default=0)
    a = a.parse_args(argv)
    H, W = a.nlat, 2 * a.nlat
    deg = f"{180.0 / H:g}"
    rng = np.random.default_rng(a.seed)
    os.makedirs(os.path.join(a.out, "kai6h"), exist_ok=True)
    for y in a.years:
        f, u10, v10, lat, lon = make_year(y, a.days, H, W, rng)
        with nc.Dataset(os.path.join(a.out, "kai6h", f"{y}.nc"), "w") as d:
            d.createDimension("time", len(f))
            d.createDimension("channel", f.shape[1])
            d.createDimension("latitude", H)
            d.createDimension("longitude", W)
            d.createVariable("fields", "f4", ("time", "channel", "latitude", "longitude"))[:] = f
            tv = d.createVariable("time", "i8", ("time",))
            tv.units, tv.calendar = f"hours since {y}-01-01", "proleptic_gregorian"
            tv[:] = 6 * np.arange(len(f))
            ch = d.createVariable("channel", str, ("channel",))
            for i, n in enumerate(contract.KAI_CHANNELS):
                ch[i] = n
            d.createVariable("latitude", "f8", ("latitude",))[:] = lat
            d.createVariable("longitude", "f8", ("longitude",))[:] = lon
        for name, x in (("u10", u10), ("v10", v10)):
            folder = contract.EXTRA_FOLDERS[name]
            os.makedirs(os.path.join(a.out, "era5_data", folder), exist_ok=True)
            with nc.Dataset(os.path.join(a.out, "era5_data", folder, f"{folder}_{y}_{deg}deg.nc"), "w") as d:
                d.createDimension("time", len(x))
                d.createDimension("lat", H)
                d.createDimension("lon", W)
                tv = d.createVariable("time", "i8", ("time",))
                tv.units = f"hours since {y}-01-01"
                tv[:] = 6 * np.arange(len(x))
                d.createVariable("lat", "f8", ("lat",))[:] = lat
                d.createVariable("lon", "f8", ("lon",))[:] = lon
                d.createVariable(name, "f4", ("time", "lat", "lon"))[:] = x
    print(f"wrote {len(a.years)} years of {H}x{W} synthetic data to {a.out}")


if __name__ == "__main__":
    main()
