"""Stage 07, step 1: the frozen VLM's decoder memory and mean on every window of a split, for the flow expert.

    python scripts/stage07_cache.py --config configs/stage07_flow_ensemble.yaml --split train   # then val, test

<out>/cache/<split>/ (resumable: a re-run continues after the last finished batch)
    memory.npy  (n, N, D) f16     the tokens F reads (weather tokens, forecast queries, physics-head tokens)
    mu.npy      (n, L, 5, h, w)   decoder mean of ClimODE's variables, averaged to the WB2 64x32 grid
    y.npy       (n, L, 5, h, w)   WB2 64x32 truth at each lead (the scoring truth)
    x0.npy      (n, 5, h, w)      WB2 64x32 truth at the issue time
    sigma.npy   (n, L, 5, h, w)   Gaussian scale of the decoder, averaged like compare_climode.py (test split only)
    meta.json   hour0, ics (6 h index in the year), years, leads, vars, mean / std, done
Fields are in the model's normalised units of the 5 variables, (x - mean) / std, stored as float16.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from climate_vlm import build, data  # noqa: E402
from climate_vlm.contract import CLIMODE_VARS  # noqa: E402
from climate_vlm.llm import PromptBuilder  # noqa: E402
from climate_vlm.model import output_units  # noqa: E402
from scripts.stage02_forecasts import Truth  # noqa: E402


def write_json(path, obj):
    with open(path + ".tmp", "w") as fh:
        json.dump(obj, fh)
    os.replace(path + ".tmp", path)


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--config", required=True)
    a.add_argument("--split", required=True, choices=["train", "val", "test"])
    a.add_argument("--batch", type=int, default=4)
    a.add_argument("--limit", type=int, default=0, help="first n windows (tests)")
    a = a.parse_args(argv)
    cfg = build.load_config(a.config)
    bc = build.load_config(cfg["base"]["config"])
    p = build.paths(bc)
    meta, norm = data.load_meta(p["cache"]), data.Normalizer.load(p["stats"])
    ch, st = meta["channels"], meta["state_channels"]
    k5 = [ch.index(v) for v in CLIMODE_VARS]
    m5, s5 = norm.mean[k5].astype(np.float64), norm.std[k5].astype(np.float64)
    lon0 = data.lon0_grid(meta["lon"])
    f = bc["data"]["factor"]
    truth = Truth(os.path.join(p["cache"], "truth_5p625"), p["cache"])
    spec = cfg.get("cache", {}).get(a.split, {}) or {}
    ds = build.dataset(bc, a.split, ics=spec.get("ics"))
    ds.index = ds.index[::int(spec.get("stride", 1))]
    if a.limit:
        ds.index = ds.index[:a.limit]
    n, leads = len(ds), [int(h) for h in bc["data"]["leads_h"]]
    out = os.path.join(cfg["out"], "cache", a.split)
    os.makedirs(out, exist_ok=True)
    mpath = os.path.join(out, "meta.json")
    info = json.load(open(mpath)) if os.path.exists(mpath) else None
    if info is not None and (info["n"] != n or info["ckpt"] != cfg["base"]["ckpt"]):
        raise RuntimeError(f"{out} holds another cache ({info['n']} windows of {info['ckpt']}); remove it first")
    done = info["done"] if info else 0
    if done >= n:
        print(f"{a.split}: cache complete ({n} windows)")
        return out

    ck = torch.load(cfg["base"]["ckpt"], map_location="cpu", weights_only=False)
    model = build.model(bc, lora=bool(ck.get("lora", False)))
    build.load_own(model, cfg["base"]["ckpt"])
    model.set_stage("2")
    model.eval()
    use_prompt = bc["model"].get("prompt", True) and model.qwen_tokenizer is not None
    pb = PromptBuilder(model.qwen_tokenizer, st, output_units(st), f"{180 / len(meta['lat']) * f:g} deg",
                       bc["data"]["history_h"], leads, compact=bc["model"].get("prompt") == "compact")
    nz = lambda x: ((np.asarray(x, np.float64) - m5[:, None, None]) / s5[:, None, None]).astype(np.float16)  # noqa
    arrs = None
    dl = DataLoader(Subset(ds, range(done, n)), batch_size=a.batch, shuffle=False, num_workers=1)
    t0 = time.time()
    for b in dl:
        with torch.no_grad():
            o = model(b["x_lr"], pb.prompt_ids(b["issue"]) if use_prompt else None)
        mu = data.coarsen(o["mean"].float()[:, :, k5].numpy(), f, lon0)
        if arrs is None:
            H, W = mu.shape[-2:]
            shapes = {"memory": (n,) + tuple(o["memory"].shape[1:]), "mu": (n, len(leads), 5, H, W),
                      "y": (n, len(leads), 5, H, W), "x0": (n, 5, H, W)}
            if a.split == "test" and o["scale"] is not None:
                shapes["sigma"] = (n, len(leads), 5, H, W)
            mode = "r+" if done else "w+"
            arrs = {k: np.lib.format.open_memmap(os.path.join(out, f"{k}.npy"), mode=mode, dtype=np.float16,
                                                 shape=s) for k, s in shapes.items()}
            if not info:
                info = {"n": n, "ckpt": cfg["base"]["ckpt"], "leads": leads, "vars": CLIMODE_VARS,
                        "mean": m5.tolist(), "std": s5.tolist(), "hour0": [], "ics": [], "years": [], "done": 0}
        B = len(b["t"])
        sl = slice(done, done + B)
        arrs["memory"][sl] = o["memory"].float().numpy().astype(np.float16)
        arrs["mu"][sl] = mu.astype(np.float16)
        if "sigma" in arrs:
            arrs["sigma"][sl] = data.coarsen(o["scale"].float()[:, :, k5].numpy(), f, lon0).astype(np.float16)
        for i, h0 in enumerate(b["hour0"].tolist()):
            arrs["y"][done + i] = np.stack([nz(truth(h0 + h)) for h in leads])
            arrs["x0"][done + i] = nz(truth(h0))
            info["hour0"].append(int(h0))
            info["ics"].append(int(b["t"][i]))
            info["years"].append(int(b["year"][i]))
        done += B
        for v in arrs.values():
            v.flush()
        info["done"] = done
        write_json(mpath, info)
        if done % (10 * a.batch) == 0 or done == n:
            print(f"{a.split}: {done}/{n} windows, {time.time() - t0:.0f} s", flush=True)
    return out


if __name__ == "__main__":
    main()
