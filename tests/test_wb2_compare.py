"""WeatherBench2 grid handling, history-only context years and the ClimODE-protocol scoring script."""

import json
import os

import numpy as np
import pytest
import torch

from climate_vlm import build, contract, data
from climate_vlm.downscaler import interp
from scripts import compare_climode, evaluate, train


def test_coarsen_wb2_grid_weights():
    H, W = 8, 16
    x = np.random.default_rng(0).normal(size=(2, H, W)).astype(np.float32)
    y = data.coarsen(x, 2, lon0=True)
    rows = contract.coarse_row_shares(H, 2).reshape(H // 2, 2)
    # coarse cell (0, 0) on a 0-deg-centred grid: fine columns -1, 0, 1 with weights 1/4, 1/2, 1/4
    lon_avg = 0.25 * x[:, :2, -1] + 0.5 * x[:, :2, 0] + 0.25 * x[:, :2, 1]
    np.testing.assert_allclose(y[:, 0, 0], (lon_avg * rows[0]).sum(-1), rtol=1e-5)
    np.testing.assert_allclose(data.coarsen(np.ones((H, W), np.float32), 2, lon0=True), 1.0, rtol=1e-6)
    with pytest.raises(ValueError):
        data.coarsen(np.ones((12, 12), np.float32), 3, lon0=True)


def test_interp_wb2_grid_hits_coarse_centres():
    h, w = 4, 8
    lon_c = np.arange(w) * 360.0 / w
    x = torch.tensor(np.cos(np.deg2rad(lon_c)), dtype=torch.float32).expand(1, 1, h, w).contiguous()
    y = interp(x, 2, lon0=True)
    assert y.shape == (1, 1, 2 * h, 2 * w)
    np.testing.assert_allclose(y[..., ::2].numpy(), np.repeat(x.numpy(), 2, -2), atol=1e-6)  # even columns = coarse
    np.testing.assert_allclose(y[..., 1::2].numpy(), 0.5 * (x + torch.roll(x, -1, -1)).repeat_interleave(2, -2).numpy(),
                               atol=1e-6)


def test_context_years_give_history_only(cfg):
    c = {**cfg, "data": {**cfg["data"], "test_years": [2017]}}
    plain = build.dataset(c, "test")
    c["data"]["context_years"] = [2016]
    ctx = build.dataset(c, "test")
    first = min(r[3] for r in ctx.index)
    assert first == 2 and 2 not in [r[3] for r in plain.index]  # 72 h history of t = 2 comes from 2016
    assert {r[2] for r in ctx.index} == {2017}  # issue times only in the split year
    hi = max(ctx.leads)
    for seg, g, y, t in ctx.index:
        lens = [len(np.load(os.path.join(cfg["data"]["cache"], f"{yy}_hours.npy"))) for yy in seg]
        assert np.repeat(seg, lens)[g + hi] == 2017  # targets never in a context year


def _truth_files(cfg):
    """WB2-style 5.625 truth for the synthetic cache: the coarsened HR fields of the five ClimODE variables."""
    cache = cfg["data"]["cache"]
    meta = data.load_meta(cache)
    k = [meta["channels"].index(v) for v in contract.CLIMODE_VARS]
    tdir = os.path.join(cache, "truth_5p625")
    os.makedirs(tdir, exist_ok=True)
    lo, hi = np.full(5, np.inf), np.full(5, -np.inf)
    for y in meta["years"]:
        x = data.coarsen(np.load(os.path.join(cache, f"{y}.npy"))[:, k], 2)
        np.save(os.path.join(tdir, f"{y}.npy"), x)
        lo, hi = np.minimum(lo, x.min((0, 2, 3))), np.maximum(hi, x.max((0, 2, 3)))
    lat = np.asarray(meta["lat"]).reshape(-1, 2).mean(1)
    np.save(os.path.join(tdir, "lat.npy"), lat)
    json.dump({"max": hi.tolist(), "min": lo.tolist()}, open(os.path.join(tdir, "minmax_2006_2018.json"), "w"))


def test_crps_matches_properscoring():
    ps = pytest.importorskip("properscoring")
    rng = np.random.default_rng(1)
    o, m, s = rng.normal(size=50), rng.normal(size=50), rng.uniform(0.1, 2, 50)
    np.testing.assert_allclose(compare_climode.crps_gaussian(o, m, s), ps.crps_gaussian(o, mu=m, sig=s), rtol=1e-6)


def test_compare_climode_end_to_end(cfg, tmp_path):
    _truth_files(cfg)
    c, out = cfg["_path"], str(tmp_path / "run")
    train.main(["--config", c, "--stage", "1", "--out", out, "--steps", "2"])
    train.main(["--config", c, "--stage", "2", "--out", out, "--init", f"{out}/stage1.pt", "--steps", "1"])
    evaluate.main(["--config", c, "--ckpt", f"{out}/stage2.pt", "--out", f"{out}/eval"])
    res = compare_climode.main(["--config", c, "--eval", f"{out}/eval", "--year", "2017"])
    leads = [str(h) for h in cfg["data"]["leads_h"]]
    assert np.isfinite(res["scores"]["ours"][leads[0]]["rmse"]["z"]["mean"])
    # truth coarsened from our HR file equals the 5.625 truth here (same source), so the grid error is ~0
    assert res["truth_coarsening_rmse"]["z"][leads[0]] < 1e-2
    # a reference file in pg-flow's layout: its persistence must be reproduced, its ClimODE rows carried over
    ref = {"ics": res["ics"][:-1], "leads": res["leads"], "vars": res["vars"],
           "scores": {"persistence": {h: {m: {v: {"mean": res["scores"]["persistence"][h][m][v]["mean"]}
                                              for v in res["vars"]} for m in ("rmse", "acc", "crps_gauss")}
                                      for h in leads},
                      "climode": {h: {m: {v: {"mean": 1.0} for v in res["vars"]} for m in ("rmse", "acc", "crps_gauss")}
                                  for h in leads}}}
    rp = tmp_path / "ref.json"
    rp.write_text(json.dumps(ref))
    r2 = compare_climode.main(["--config", c, "--eval", f"{out}/eval", "--year", "2017", "--reference", str(rp),
                               "--out", str(tmp_path / "cmp")])
    assert r2["n_ic"] == len(res["ics"]) - 1
    assert r2["scores"]["climode"][leads[0]]["rmse"]["z"]["mean"] == 1.0
    assert r2["checks"]["persistence_max_rel_diff_vs_reference"] > 0  # one IC fewer than the reference was built on
    assert os.path.exists(tmp_path / "cmp" / "table.md")
