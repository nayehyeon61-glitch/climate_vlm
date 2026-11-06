import math

import numpy as np
import torch

from climate_vlm import build, contract, data, physics


def test_coarsen_is_area_weighted_mean():
    H, W, f = 32, 64, 2
    lat = contract.grid_latitudes(H)
    x = np.random.default_rng(0).normal(size=(3, H, W)).astype(np.float32)
    y = data.coarsen(x, f)
    w = np.cos(np.deg2rad(lat))
    # global area mean is preserved by an exact conservative average
    lat_c = contract.grid_latitudes(H // f)
    edges = np.sin(np.deg2rad(np.linspace(-90, 90, H // f + 1)))
    wc = np.diff(edges)
    np.testing.assert_allclose((y.mean(-1) * wc).sum(-1) / wc.sum(), (x.mean(-1) * w).sum(-1) / w.sum(), rtol=2e-3)
    assert y.shape == (3, H // f, W // f) and len(lat_c) == H // f
    np.testing.assert_allclose(data.coarsen(np.ones((H, W), np.float32), f), 1.0, rtol=1e-6)


def test_windows_inputs_and_labels(cfg):
    ds = build.dataset(cfg, "train")
    s = ds[0]
    S, L = len(cfg["data"]["history_h"]), len(cfg["data"]["leads_h"])
    C_state = len(cfg["data"]["state_channels"])
    C_out = C_state + len(cfg["data"]["extra_channels"])
    assert s["x_lr"].shape == (S, C_state, 16, 32) and s["x_hr"].shape == (S, C_state, 32, 64)
    assert s["y"].shape == (L, C_out, 32, 64) and s["x0"].shape == (C_out, 32, 64)
    # the model input is the coarsened history, nothing else
    np.testing.assert_allclose(s["x_lr"].numpy(), data.coarsen(s["x_hr"].numpy(), 2), atol=1e-5)
    # labels sit at the right offsets of the 6-hourly time line
    seg, g, year, t = ds.index[0]
    arr = np.load(f"{cfg['data']['cache']}/{year}.npy")
    norm = ds.normalizer
    for k, h in enumerate(cfg["data"]["leads_h"]):
        np.testing.assert_allclose(s["y"][k].numpy(), norm.norm(arr[t + h // 6], list(range(C_out))), atol=1e-4)
    np.testing.assert_allclose(s["x0"].numpy(), norm.norm(arr[t], list(range(C_out))), atol=1e-4)


def test_test_ics_are_pgflow_climode_ics(cfg):
    ds = build.dataset(cfg, "test", ics="climode")
    ts = [r[3] for r in ds.index]
    assert all((t - 2) % 8 == 0 for t in ts)  # 12 UTC every second day
    assert all(ds[i]["issue"].endswith("12 UTC") for i in range(len(ds)))


def _sphere(H=32, W=64):
    lat = contract.grid_latitudes(H)
    lon = 360.0 / W * np.arange(W)
    return lat, lon, np.deg2rad(lat)[:, None], np.deg2rad(lon)[None, :]


def test_temperature_advection_matches_analytic():
    lat, lon, la, lo = _sphere(64, 128)
    a, U, A = contract.CONST.earth_radius, 10.0, 5.0
    T = 280 + A * np.sin(lo) * np.cos(la)
    u = U * np.cos(la) * np.ones_like(T)
    v = np.zeros_like(T)
    ch = ["u850", "v850", "t850"]
    x = torch.tensor(np.stack([u, v, T]), dtype=torch.float32)
    ph = physics.Physics(ch, lat, lon, np.zeros(3), np.ones(3), diags=["tadv850"])
    got = ph.diagnose(x)[0].numpy()
    want = -U * A * np.cos(lo) * np.cos(la) / a * physics.DAY  # -u/(a cos) dT/dlambda
    dl = np.deg2rad(lon[1] - lon[0])
    want *= math.sin(dl) / dl  # centred difference of a single wave
    np.testing.assert_allclose(got, np.broadcast_to(want, got.shape), atol=1e-3)


def test_hydrostatic_residual_zero_for_balanced_state(cfg):
    meta = data.load_meta(cfg["data"]["cache"])
    norm = data.Normalizer.load(build.paths(cfg)["stats"])
    _, po = build.physics_pair(cfg, meta, norm)
    s = build.dataset(cfg, "train")[0]
    r = po.hydrostatic_residual(s["y"])  # synthetic truth is integrated hydrostatically
    assert r.shape[-3] == len(po.layers) == 2  # 850-700, 700-500 hPa
    # metres; not exactly 0: the layer 700-500 hPa skips the 600 hPa level the data were integrated over
    assert float(r.abs().max()) < 5.0
    bad = s["y"].clone()
    bad[:, meta["channels"].index("z500")] += 1.0  # +1 std of z500
    assert float(po.hydrostatic_residual(bad).abs().max()) > 50.0


def test_diagnostics_supervision_targets(cfg):
    meta = data.load_meta(cfg["data"]["cache"])
    norm = data.Normalizer.load(build.paths(cfg)["stats"])
    obj = build.objective(cfg, meta, norm, "2")
    s = build.dataset(cfg, "train")[0]
    z = obj.targets(s["x_hr"][None], s["y"][None])
    assert z.shape == (1, 1 + len(cfg["data"]["leads_h"]), len(obj.po.diags), 4, 8)
    assert torch.isfinite(z).all()
