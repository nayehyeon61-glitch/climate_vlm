"""Stage 07 (flow-matching ensemble generator after the decoder): the expert itself and the scripts end to end."""

import json
import os

import numpy as np
import pytest
import torch
import yaml

from climate_vlm.flow import FlowExpert, StructuredNoise
from scripts import stage07_analysis, stage07_cache, stage07_flow, train
from tests.test_wb2_compare import _truth_files


def test_patches_roundtrip_and_zero_start():
    m = FlowExpert(3, 2, 8, 16, patch=(4, 4), dim=16, depth=1, heads=2, mem_dim=8)
    x = torch.randn(2, 3, 2, 8, 16)
    torch.testing.assert_close(m._unpatch(m._patch(x)), x)
    v = m(x, torch.rand(2), mu=torch.randn_like(x), x0=torch.randn(2, 2, 8, 16), memory=torch.randn(2, 5, 8))
    assert v.shape == x.shape and float(v.abs().max()) == 0.0  # zero-initialised output layer


@pytest.mark.parametrize("use_mem", [True, False])
def test_learns_a_conditional_bimodal_target(use_mem):
    """r = sign(c) + small noise, where c is given only through the memory (or through x0): samples must follow."""
    torch.manual_seed(0)
    m = FlowExpert(1, 1, 4, 8, patch=(4, 4), dim=32, depth=2, heads=2, mem_dim=4, use_mem=use_mem, use_mu=False,
                   use_x0=not use_mem)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    w = torch.ones(4)

    def draw(n):
        c = torch.sign(torch.randn(n))
        r = c.view(n, 1, 1, 1, 1) + 0.1 * torch.randn(n, 1, 1, 4, 8)
        v = torch.tensor([1.0, -1.0, 0.5, 0.0])  # not constant: the memory goes through a layer norm
        cond = {"memory": (c.view(n, 1, 1) * v).expand(n, 3, 4).contiguous()} if use_mem else \
            {"x0": c.view(n, 1, 1, 1).expand(n, 1, 4, 8).contiguous()}
        return c, r, cond

    for _ in range(300):
        _, r, cond = draw(32)
        loss = m.loss(r, w, **cond)
        opt.zero_grad()
        loss.backward()
        opt.step()
    c, _, cond = draw(8)
    s = m.sample(6, steps=10, generator=torch.Generator().manual_seed(0), **cond)  # (8, 6, 1, 1, 4, 8)
    means = s.mean((1, 2, 3, 4, 5))
    assert torch.all(torch.sign(means) == c), (means, c)
    assert float((s.mean((-1, -2)) - c.view(8, 1, 1, 1)).abs().max()) < 0.5


def _lat(H):
    return -90 + (np.arange(H) + 0.5) * 180.0 / H


def _corr(a, b, w=None):
    w = torch.ones(a.shape[-2]) if w is None else w
    w = w[:, None]
    return float((w * a * b).sum() / torch.sqrt((w * a * a).sum() * (w * b * b).sum()))


def test_structured_noise_takes_the_targets_covariance():
    """Smooth targets correlated between leads: the fitted source has their spectrum, lead correlation and unit
    variance; on white targets it stays white."""
    torch.manual_seed(0)
    H, W, L = 16, 32, 3
    sn = StructuredNoise(L, 2, _lat(H), W)
    lp = torch.tensor([1.0 / (1 + l) ** 2 for l in range(H)])  # red spectrum
    sn.gain.copy_(lp.sqrt().expand(L, 2, H))
    smooth = sn._filter(torch.randn(400, L, 2, H, W))
    r = 0.8 * smooth[:, :1] + 0.6 * smooth  # leads correlated through a shared part
    r = r / r.std()
    fit = StructuredNoise(L, 2, _lat(H), W).fit(r, draws=128)
    x = fit(torch.Size((400, L, 2, H, W)), torch.Generator().manual_seed(1))
    v = fit._var(x)
    assert torch.allclose(v, torch.ones_like(v), atol=0.05), v
    for i, j in ((0, 1), (0, 2), (1, 2)):  # area-weighted, as fit() measures it
        assert abs(_corr(x[:, i], x[:, j], fit.wa) - _corr(r[:, i], r[:, j], fit.wa)) < 0.05
    rs = r / torch.sqrt(fit._var(r))[..., None, None]
    ratio = fit.power(x).mean(0)[..., :8] / fit.power(rs).mean(0)[..., :8]
    assert float((ratio - 1).abs().max()) < 0.2, ratio
    assert abs(_corr(x[..., :-1], x[..., 1:]) - _corr(r[..., :-1], r[..., 1:])) < 0.03  # neighbours move together
    white = StructuredNoise(L, 2, _lat(H), W).fit(torch.randn(400, L, 2, H, W), draws=128)
    y = white(torch.Size((200, L, 2, H, W)))
    assert abs(_corr(y[..., :-1], y[..., 1:])) < 0.05 and abs(_corr(y[:, 0], y[:, 1])) < 0.05


def test_structured_flow_expert_trains_samples_and_reloads():
    torch.manual_seed(0)
    m = FlowExpert(2, 1, 8, 16, patch=(4, 4), dim=16, depth=1, heads=2, mem_dim=4, use_mem=False, use_mu=True,
                   use_x0=False, noise="structured", lat_deg=_lat(8))
    r = torch.randn(32, 2, 1, 8, 16)
    m.noise.fit(r, draws=16)
    loss = m.loss(r[:4], torch.ones(8), mu=torch.zeros(4, 2, 1, 8, 16))
    loss.backward()
    m2 = FlowExpert(2, 1, 8, 16, patch=(4, 4), dim=16, depth=1, heads=2, mem_dim=4, use_mem=False, use_mu=True,
                    use_x0=False, noise="structured", lat_deg=_lat(8))
    m2.load_state_dict(m.state_dict())
    torch.testing.assert_close(m2.noise.chol, m.noise.chol)
    s = m2.sample(3, steps=2, generator=torch.Generator().manual_seed(0), mu=torch.zeros(2, 2, 1, 8, 16))
    assert s.shape == (2, 3, 2, 1, 8, 16) and torch.isfinite(s).all()
    with pytest.raises(ValueError):
        FlowExpert(2, 1, 8, 16, noise="structured")


def test_fair_crps_of_a_point_ensemble_is_the_absolute_error():
    E = np.full((4, 3), 2.0)
    np.testing.assert_allclose(stage07_flow.crps_fair(E, np.array([1.0, 2.0, 4.0])), [1.0, 0.0, 2.0])


def test_stage07_pipeline(cfg, tmp_path):
    _truth_files(cfg)
    c, run = cfg["_path"], str(tmp_path / "run")
    train.main(["--config", c, "--stage", "1", "--out", run, "--steps", "2"])
    train.main(["--config", c, "--stage", "2", "--out", run, "--init", f"{run}/stage1.pt", "--steps", "1"])
    with open(os.path.join(os.path.dirname(__file__), "..", "configs", "stage07_flow_ensemble.yaml")) as fh:
        s = yaml.safe_load(fh)
    s.update(base={"config": c, "ckpt": f"{run}/stage2.pt"}, out=str(tmp_path / "s07"), docs=str(tmp_path / "docs"),
             reference=None)
    s["flow"].update(dim=16, depth=1, heads=2, batch=4, val_every=2, val_windows=4, members=3, sample_steps=2)
    sc = tmp_path / "s07.yaml"
    sc.write_text(yaml.safe_dump(s))
    for split in ("train", "val", "test"):
        stage07_cache.main(["--config", str(sc), "--split", split, "--limit", "6"])
    info = json.load(open(tmp_path / "s07" / "cache" / "test" / "meta.json"))
    assert info["done"] == info["n"] > 0
    mu = np.load(tmp_path / "s07" / "cache" / "test" / "mu.npy")
    assert mu.shape[1:3] == (len(cfg["data"]["leads_h"]), 5) and np.isfinite(mu).all()
    assert os.path.exists(tmp_path / "s07" / "cache" / "test" / "sigma.npy")
    stage07_cache.main(["--config", str(sc), "--split", "test", "--limit", "6"])  # complete: nothing to do
    for arm in s["arms"]:
        stage07_flow.main(["--config", str(sc), "train", "--arm", arm, "--steps", "2"])
        stage07_flow.main(["--config", str(sc), "sample", "--arm", arm])
        z = np.load(tmp_path / "s07" / arm / "members_2020.npz")
        assert z["E_minus_x0"].shape == (info["n"], 3) + mu.shape[1:]
    res = stage07_flow.main(["--config", str(sc), "score"])
    h = str(cfg["data"]["leads_h"][0])
    for name in list(s["arms"]) + ["gaussian_head", "persistence"]:
        assert np.isfinite(res["scores"][name][h]["rmse"]["z"]["mean"]), name
    for arm in s["arms"]:
        d = res["scores"][arm][h]
        assert d["crps_ens"]["t"]["mean"] >= 0 and 0 <= d["outside"]["t"]["mean"] <= 1
    assert os.path.exists(tmp_path / "docs" / "table.md")
    an = stage07_analysis.main(["--config", str(sc)])
    for arm in s["arms"]:
        assert np.isfinite(an["pairs"][f"{arm} - gaussian_head"][h]["crps"]["z"]["mean"]), arm
    assert "fm_vla_x0 - fm_mu_x0" in an["pairs"] and "fm_mu_x0_sn - fm_mu_x0" in an["pairs"]
    for name in ("gaussian_head", "fm_mu_x0_sn"):
        assert np.isfinite(an["aggregates"][name]["lead_mean"]["z"]["crps"]), name
        assert np.isfinite(an["aggregates"][name]["box"][h]["t"]["spread_skill"]), name
    g = an["lead_corr"]["gaussian_head"]["t"]["members"]
    assert np.isfinite(g).all() and max(abs(c) for c in g) < 0.5  # independent draws per lead
    assert len(an["bands"]["fm_vla"][h]["z"]["spread"]) == len(an["band_edges"]) + 1
    assert os.path.exists(tmp_path / "docs" / "analysis.md")
