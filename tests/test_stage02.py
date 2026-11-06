"""Stage 02 (Laya router) scripts end to end on the synthetic files; laya's decisions are stubbed by random arrays."""

import json
import os

import numpy as np
import yaml

from scripts import stage02_dataset, stage02_experts, stage02_forecasts, stage02_router, train
from scripts.stage02_common import EXPERTS, sub_regions
from tests.test_wb2_compare import _truth_files


def test_sub_regions_order():
    s = sub_regions(4, 6, [2, 2])
    assert [n for n, _, _ in s] == ["NW", "NE", "SW", "SE"]
    assert s[0][1] == slice(2, 4) and s[0][2] == slice(0, 3)  # north = the upper rows of a S -> N grid


def test_stage02_pipeline(cfg, tmp_path):
    _truth_files(cfg)
    c, run = cfg["_path"], str(tmp_path / "run")
    train.main(["--config", c, "--stage", "1", "--out", run, "--steps", "2"])
    train.main(["--config", c, "--stage", "3", "--out", run, "--init", f"{run}/stage1.pt", "--steps", "1"])
    out = str(tmp_path / "s02")
    per = {"experts": {"split": "A", "years": [2015], "hours": [0, 12], "stride": 1, "margin_h": 0},
           "router": {"split": "B", "years": [2016], "hours": [0, 6, 12, 18], "stride": 1, "margin_h": 0},
           "calib": {"split": "C", "years": [2017], "dates": ["2017-01-01", "2017-01-10"], "hours": [0, 12],
                     "stride": 1, "margin_h": 12},
           "test": {"split": "D", "years": [2017], "dates": ["2017-01-11", "2017-01-20"], "hours": [0, 12],
                    "stride": 1, "margin_h": 12}}
    s = {"base": {"config": c, "ckpt": f"{run}/stage3.pt"}, "out": out, "docs": str(tmp_path / "docs"),
         "schema": {"data": "t"}, "leads": [24], "periods": per,
         "domain": {"name": "Test", "key": "T", "lat": [0, 90], "lon": [0, 180], "sub": [2, 2]},
         "features": {"strong_quantile": 0.9, "near_tie": 0.05},
         "experts": {"hidden": 8, "layers": 2, "steps": 3, "batch": 4, "lr": 1e-3,
                     "advection": {"inputs": ["z500", "t850", "tadv850", "dz500_24h"], "weight": ["tadv850"]},
                     "moisture": {"inputs": ["q850", "qconv850", "dt850_24h"], "weight": ["qconv850", "qadv850"]}},
         "router": {"mlp": {"hidden": 8, "epochs": 2, "lr": 1e-3}, "head": {"hidden": 8, "epochs": 2, "lr": 1e-3},
                    "laya": {"base": "unused", "batch": 4, "instructions": "q",
                             "criteria": {"baseline": "a", "advection": "b", "moisture": "c"}}}}
    sc = tmp_path / "s02.yaml"
    sc.write_text(yaml.safe_dump(s))
    stage02_forecasts.main(["--config", str(sc), "--batch", "4"])
    z = np.load(f"{out}/fc_test.npz")
    n = len(z["issue"])
    assert z["P"].shape == (n, 1, 5, 16, 32) and z["F"].shape == (n, len(stage02_forecasts.FMAPS), 16, 32)
    assert z["R"].shape[:2] == (n, len(stage02_forecasts.RMAPS)) and np.isfinite(z["R"]).all()
    first = str(np.datetime64(int(z["issue"].min()), "h"))
    assert first >= "2017-01-11" and str(np.datetime64(int(z["issue"].max()) + 24, "h")) <= "2017-01-20T06"
    stage02_experts.main(["--config", str(sc)])
    fx = np.load(f"{out}/fx_test.npz")
    assert fx["E"].shape[:3] == (n, 3, 1)
    np.testing.assert_array_equal(fx["E"][:, 0], z["P"][..., z["rows5"][:, None], z["cols5"][None, :]])
    man = stage02_dataset.main(["--config", str(sc), "--no-laya-check"])
    assert man["leakage_checks"]["label_names_in_state"] == 0 and not man["leakage_checks"]["window_overlap"]
    rows = [json.loads(line) for line in open(f"{out}/router_test.jsonl")]
    assert len(rows) == n and set(rows[0]) == {"state", "questions", "expected"}
    assert rows[0]["expected"]["expert"] in EXPERTS and "lead=24h" in rows[0]["state"]
    assert "nan" not in open(f"{out}/router_train.jsonl").read().lower()
    rng = np.random.default_rng(0)
    for model, splits in (("base", "BCD"), ("tuned", "CD")):  # stand-ins for stage02_laya.py predict
        for sp in splits:
            m = sum(1 for _ in open(f"{out}/{stage02_dataset.JSONL[sp]}.jsonl"))
            res = {"probs": rng.dirichlet(np.ones(3), m).astype(np.float32), "sec_per_decision": 0.1, "rss_gb": 1.0}
            if model == "base":
                res["reps"] = rng.normal(size=(m, 3, 16)).astype(np.float16)
            np.savez(f"{out}/laya_{model}_{sp}.npz", **res)
    res = stage02_router.main(["--config", str(sc)])
    assert set(res["routers"]) == {"B_mlp", "C0_laya_zeroshot", "C1_laya_tuned", "C2_laya_cached_head"}
    for r in res["routers"].values():
        assert r["error_ratio_to_oracle"] >= 1.0 - 1e-9
    assert res["fields"]["oracle"]["E"] <= res["fields"]["A_baseline"]["E"] + 1e-12
    assert os.path.exists(tmp_path / "docs" / "table.md")
