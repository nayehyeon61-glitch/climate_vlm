"""Stage 07 (flow-matching ensemble generator after the decoder): the expert itself and the scripts end to end."""

import json
import os

import numpy as np
import pytest
import torch
import yaml

from climate_vlm.flow import FlowExpert
from scripts import stage07_cache, stage07_flow, train
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
