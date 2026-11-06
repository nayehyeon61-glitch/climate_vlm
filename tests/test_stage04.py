"""Stage 04 (neural-operator decomposition): band split, advection step, model variants, scripts."""

import copy
import os

import numpy as np
import pytest
import torch
import yaml

from climate_vlm import build, data
from climate_vlm.spectral import Advection, SphericalBands, legendre
from scripts import evaluate, train


def grid(n):
    return -90 + (np.arange(n) + 0.5) * 180 / n, np.arange(2 * n) * 360 / (2 * n)


def harmonic(lat, lon, l, m):
    p = legendre(l, m, np.sin(np.deg2rad(lat)))[m, l]
    return torch.tensor(p[:, None] * np.cos(m * np.deg2rad(lon))[None], dtype=torch.float32)


@pytest.mark.parametrize("n", [32, 64])
def test_bands_split_and_recombine(n):
    lat, lon = grid(n)
    sb = SphericalBands(lat, lon, (6, 15))
    for (l, m), k in (((3, 2), 0), ((10, 4), 1), ((14, 0), 1), ((25, 5), 2)):
        e = torch.stack([b.pow(2).mean() for b in sb(harmonic(lat, lon, l, m))])
        assert float(e[k] / e.sum()) > 0.999, (l, m, e)
    x = torch.randn(2, 3, n, 2 * n)
    b = sb(x)
    torch.testing.assert_close(sum(b), x)  # lossless
    torch.testing.assert_close(sb.combine(torch.stack(b, -3)), x, atol=1e-4, rtol=0)  # bands are projections
    torch.testing.assert_close(sb(torch.roll(x, 5, -1))[1], torch.roll(b[1], 5, -1), atol=1e-5, rtol=0)
    flip = sb(torch.flip(x, [-2]))[0]  # pole-symmetric grid: north-south mirror commutes with the split
    torch.testing.assert_close(flip, torch.flip(b[0], [-2]), atol=1e-5, rtol=0)


def test_advection_moves_and_keeps_mass():
    adv = Advection(dt_h=6, max_cells_per_h=1.0, compression=False)
    u0 = torch.zeros(1, 1, 16, 32)
    u0[0, 0, 6:9, 3:6] = 1.0
    vel = torch.zeros(1, 2, 1, 2, 16, 32)
    vel[:, :, :, 0] = 20.0  # tanh -> 1 cell/h eastward
    out = adv(u0, vel, torch.zeros(1, 2, 1, 16, 32), [6, 30])
    x = torch.arange(32.0)
    for j, h in enumerate([6, 30]):
        f = out[0, j, 0]
        cx = float((f.sum(0) * x).sum() / f.sum())
        assert abs(cx - (4 + h) % 32) < 1e-3 and abs(float(f.sum()) - 9) < 1e-3  # wraps around in longitude
    still = adv(u0, torch.zeros(1, 1, 1, 2, 16, 32), torch.ones(1, 1, 1, 16, 32), [24])
    torch.testing.assert_close(still[0, 0], u0[0] + 1.0)  # source per day


def variant(cfg, encoder="qwen", output="bands"):
    c = copy.deepcopy(cfg)
    c["model"]["spectral"] = {"edges": [6, 15], "split": "mean", "output": output,
                              "advect": {"channels": ["z500", "t850", "t2m"], "dt_h": 12}}
    v = c["model"]["vision"]
    v["types"] = ["band0", "band1", "band2", "tendency"]
    v["ranges"] = {**v["ranges"], "band0": 2.5, "band1": 2.0, "band2": 1.5}
    v["encoder"] = encoder
    v["sfno"] = {"hidden": 8, "layers": 2, "out": 16, "patch": 8}
    return c


@pytest.mark.parametrize("encoder,output", [("qwen", "bands"), ("sfno", "bands"), ("sfno", "advect")])
def test_variant_forward_and_gradients(cfg, encoder, output):
    m = build.model(variant(cfg, encoder, output))
    params = m.set_stage("2")
    ds = build.dataset(cfg, "train")
    b = torch.utils.data.default_collate([ds[0], ds[1]])
    out = m(b["x_lr"])
    C = len(data.load_meta(cfg["data"]["cache"])["channels"])
    H, W = b["x_hr"].shape[-2:]
    assert out["mean"].shape == (2, len(cfg["data"]["leads_h"]), C, H, W)
    out["mean"].pow(2).mean().backward()
    tok = m.tokenizer.operator if encoder == "sfno" else m.tokenizer.resampler
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in tok.parameters())
    assert all(id(p) in {id(q) for q in params} for p in m.tokenizer.parameters()) == (encoder == "sfno")


def test_zero_maps_give_the_base(cfg):
    m = build.model(variant(cfg, "sfno", "advect"))
    B, L, H, W = 2, len(cfg["data"]["leads_h"]), 32, 64
    base = torch.randn(B, len(m.out), H, W)
    out = m._fields(torch.zeros(B, L, m.F.n_maps, H, W), base)
    torch.testing.assert_close(out, base[:, None].expand_as(out), atol=1e-3, rtol=0)  # float32 resampling


def test_advect_refuses_lead_questions(cfg):
    c = variant(cfg, "sfno", "advect")
    c["model"]["question"] = {"enabled": True}
    with pytest.raises(ValueError):
        build.model(c)


def test_stage04_scripts(cfg, tmp_path):
    c = variant(cfg, "sfno", "advect")
    path = tmp_path / "s4.yaml"
    path.write_text(yaml.safe_dump({k: v for k, v in c.items() if k != "_path"}))
    run = str(tmp_path / "run")
    train.main(["--config", str(path), "--stage", "2", "--out", run, "--steps", "2"])
    evaluate.main(["--config", str(path), "--ckpt", f"{run}/stage2.pt", "--out", f"{run}/eval"])
    assert os.path.exists(f"{run}/eval/scores.json")
