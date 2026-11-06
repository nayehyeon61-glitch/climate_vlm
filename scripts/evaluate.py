"""Forecasts on pg-flow's test ICs (6 h index 2 + 8 e: 12 UTC every second day), scores, pg-flow files.

    python scripts/evaluate.py --config C --ckpt runs/x/stage3.pt --out runs/x/eval [--years 2017 2018] [--members 8]

Writes per year <out>/<tag>_ens_<year>.npz (pg-flow member layout, ClimODE's 5 variables; score it with pg-flow's
score_step13.py / compare_step13_climode.py, or coarsen_ens.py first to score on 5.625 deg) and
<out>/<tag>_all_<year>.npz (every output channel), plus <out>/scores.json: area-weighted RMSE, ACC against the
train monthly climatology and CRPS per channel and lead, with persistence (issue-time analysis) as reference.
With a Gaussian decoder, --members draws per-grid-point members (marginal, not a spatial joint ensemble).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from climate_vlm import build, data, io, metrics  # noqa: E402
from climate_vlm.contract import CLIMODE_VARS  # noqa: E402
from climate_vlm.llm import PromptBuilder  # noqa: E402
from climate_vlm.model import output_units  # noqa: E402


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--config", required=True)
    a.add_argument("--ckpt", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--years", type=int, nargs="*")
    a.add_argument("--members", type=int, default=0)
    a.add_argument("--tag", default="climate_vlm")
    a.add_argument("--batch", type=int, default=4)
    a.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = a.parse_args(argv)
    cfg = build.load_config(a.config)
    dev = torch.device(a.device)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    p = build.paths(cfg)
    meta, norm = data.load_meta(p["cache"]), data.Normalizer.load(p["stats"])
    clim = np.load(p["clim"])
    model = build.model(cfg, lora=bool(ck.get("lora", False)), device=dev)
    build.load_own(model, a.ckpt)
    model.set_stage("2")
    model.eval()
    use_prompt = cfg["model"].get("prompt", True) and model.qwen_tokenizer is not None
    pb = PromptBuilder(model.qwen_tokenizer, meta["state_channels"], output_units(meta["state_channels"]),
                       f"{180 / len(meta['lat']) * cfg['data']['factor']:g} deg", cfg["data"]["history_h"],
                       cfg["data"]["leads_h"])
    ch, leads, lat = meta["channels"], cfg["data"]["leads_h"], meta["lat"]
    all_idx = list(range(len(ch)))
    years = a.years or build.years(cfg["data"]["test_years"])
    amp = dev.type == "cuda" and cfg["train"].get("amp", True)
    gen = torch.Generator(device="cpu").manual_seed(0)
    scores = {k: [] for k in ("rmse", "acc", "rmse_persistence", "crps")}
    for y in years:
        ds = build.dataset(cfg, "test", ics="climode")
        ds.index = [r for r in ds.index if r[2] == y]
        if not ds.index:
            print(f"{y}: no ICs (year not in test_years or not enough history)")
            continue
        dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=2)
        acc_y = {k: [] for k in ("mean", "truth", "x0", "ics", "scale", "members")}
        for b in dl:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out = model(b["x_lr"].to(dev), pb.prompt_ids(b["issue"]) if use_prompt else None)
            mu = norm.denorm_t(out["mean"].float().cpu(), all_idx).numpy()
            O = norm.denorm_t(b["y"], all_idx).numpy()
            X0 = norm.denorm_t(b["x0"], all_idx).numpy()
            months = [[int(str(np.datetime64(int(h0) + int(h), "h"))[5:7]) - 1 for h in leads] for h0 in b["hour0"]]
            C = clim[np.array(months)]  # (B, L, C, H, W)
            scores["rmse"].append(metrics.rmse(mu, O, lat))
            scores["acc"].append(metrics.acc(mu, O, C, lat))
            scores["rmse_persistence"].append(metrics.rmse(np.broadcast_to(X0[:, None], O.shape), O, lat))
            if out["scale"] is not None:
                sd = (out["scale"].float().cpu() * torch.tensor(norm.std).view(-1, 1, 1)).numpy()
                scores["crps"].append(metrics.crps_gaussian(mu, sd, O, lat))
                acc_y["scale"].append(sd)
                if a.members > 1:
                    eps = torch.randn((len(mu), a.members) + mu.shape[1:], generator=gen).numpy()
                    acc_y["members"].append(mu[:, None] + sd[:, None] * eps)
            acc_y["mean"].append(mu), acc_y["truth"].append(O), acc_y["x0"].append(X0)
            acc_y["ics"].extend(int(t) for t in b["t"])
        cat = lambda k: np.concatenate(acc_y[k]) if acc_y[k] else None  # noqa: E731
        files = io.write_year(a.out, a.tag, y, ch, cat("mean"), cat("truth"), cat("x0"), acc_y["ics"], leads,
                              members=cat("members"), scale=cat("scale"))
        print(f"{y}: {len(acc_y['ics'])} ICs -> {', '.join(files)}")
    res = {"leads_h": leads, "channels": ch, "checkpoint": a.ckpt}
    for k, v in scores.items():
        if v:
            r = np.concatenate(v).mean(0)  # (L, C)
            res[k] = {c: {str(h): float(r[j, i]) for j, h in enumerate(leads)} for i, c in enumerate(ch)}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "scores.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    if "rmse" in res:
        print("RMSE (persistence) " + " ".join(f"{h:>14d}h" for h in leads))
        for c in [c for c in CLIMODE_VARS if c in ch] + [c for c in ch if c not in CLIMODE_VARS]:
            print(f"{c:>18s} " + " ".join(f"{res['rmse'][c][str(h)]:9.3g} ({res['rmse_persistence'][c][str(h)]:.3g})"
                                          for h in leads))


if __name__ == "__main__":
    main()
