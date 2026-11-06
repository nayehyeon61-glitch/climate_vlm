"""Stage 04: 2016 validation error per scored variable, to choose the stage-05 arm without the 2020 test year.

train.py's validation loss averages all output channels (18 state channels + u10, v10); the advection arm changes
only z500, t850 and t2m, so that average hides it. This scores every checkpoint on the 2016 validation windows
(all 179, every lead) per ClimODE variable: area-weighted RMSE in normalised units, and their mean.

    python scripts/stage04_val_vars.py --runs control=runs/stage01/stage2.pt:configs/stage01_cpu_small.yaml \
        bands=runs/stage04/bands/stage2.pt:configs/stage04_bands.yaml ... --out docs/stage04/val2016_vars.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from climate_vlm import build, data  # noqa: E402
from climate_vlm.contract import CLIMODE_VARS  # noqa: E402
from climate_vlm.llm import PromptBuilder  # noqa: E402
from climate_vlm.model import output_units  # noqa: E402
from climate_vlm.physics import area_weight  # noqa: E402


def score(cfg_path, ckpt, batch):
    cfg = build.load_config(cfg_path)
    p = build.paths(cfg)
    meta = data.load_meta(p["cache"])
    model = build.model(cfg)
    build.load_own(model, ckpt)
    model.set_stage("2")
    model.eval()
    pb = PromptBuilder(model.qwen_tokenizer, meta["state_channels"], output_units(meta["state_channels"]),
                       f"{180 / len(meta['lat']) * cfg['data']['factor']:g} deg", cfg["data"]["history_h"],
                       cfg["data"]["leads_h"], compact=cfg["model"].get("prompt") == "compact")
    use_prompt = cfg["model"].get("prompt", True) and model.qwen_tokenizer is not None
    ch = [meta["channels"].index(v) for v in CLIMODE_VARS]
    w = area_weight(meta["lat"]).view(-1, 1)
    se, n = 0.0, 0
    for b in DataLoader(build.dataset(cfg, "val"), batch_size=batch, shuffle=False, num_workers=0):
        with torch.no_grad():
            out = model(b["x_lr"], pb.prompt_ids(b["issue"]) if use_prompt else None)
        e = ((out["mean"].float()[:, :, ch] - b["y"][:, :, ch]) ** 2 * w).mean((-1, -2))  # (B, L, 5)
        se, n = se + e.sum(0), n + len(e)
    rmse = (se / n).sqrt()  # (L, 5)
    leads = cfg["data"]["leads_h"]
    return {"n_windows": n, "leads": leads,
            "rmse_norm": {v: {str(h): float(rmse[j, k]) for j, h in enumerate(leads)} for k, v in enumerate(CLIMODE_VARS)},
            "mean_over_vars_and_leads": float(rmse.mean())}


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--runs", nargs="+", required=True, help="name=checkpoint:config")
    a.add_argument("--batch", type=int, default=2)
    a.add_argument("--out", required=True)
    a = a.parse_args(argv)
    res = {}
    for r in a.runs:
        name, rest = r.split("=", 1)
        ckpt, cfg = rest.split(":", 1)
        res[name] = score(cfg, ckpt, a.batch)
        print(name, json.dumps({v: round(np.mean(list(d.values())), 4) for v, d in res[name]["rmse_norm"].items()}),
              round(res[name]["mean_over_vars_and_leads"], 4), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
