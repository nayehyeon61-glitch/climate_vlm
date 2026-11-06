"""pg-flow KAI files -> training cache, then train-only statistics.

    python scripts/prepare_data.py --config configs/qwen35_0p8b.yaml

1. cache   <cache>/<year>.npy (T, C, H, W) of data.state_channels + data.extra_channels for every split year
2. stats   from the TRAIN years only (plan p.9): channel mean / std (stats.npz), monthly climatology for ACC
           (clim.npy, (12, C, H, W)), mean / std of the patch-pooled diagnostics (diag_stats.npz)
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from climate_vlm import build, data, physics  # noqa: E402


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--config", required=True)
    a.add_argument("--stride", type=int, default=4, help="every n-th step for the statistics")
    a.add_argument("--diag-samples", type=int, default=2000)
    a = a.parse_args(argv)
    cfg = build.load_config(a.config)
    d, p = cfg["data"], build.paths(cfg)
    all_years = sorted(set(build.years(d["train_years"]) + build.years(d["val_years"]) + build.years(d["test_years"])))
    todo = [y for y in all_years if not os.path.exists(os.path.join(p["cache"], f"{y}.npy"))]
    if todo:
        data.prepare_cache(d["kai_dir"], p["cache"], todo, d["state_channels"], d.get("extra_channels", []),
                           d.get("era5_data"))
        print("cached", todo)
    meta = data.load_meta(p["cache"])
    C, H, W = len(meta["channels"]), len(meta["lat"]), len(meta["lon"])

    s1, s2, n = np.zeros(C), np.zeros(C), 0
    clim, cnt = np.zeros((12, C, H, W)), np.zeros(12)
    for y in build.years(d["train_years"]):
        arr = np.load(os.path.join(p["cache"], f"{y}.npy"), mmap_mode="r")
        hrs = np.load(os.path.join(p["cache"], f"{y}_hours.npy"))
        months = np.array([int(str(np.datetime64(int(h), "h"))[5:7]) for h in hrs])
        for t in range(0, len(arr), a.stride):
            x = np.asarray(arr[t], dtype=np.float64)
            s1 += x.mean((1, 2))
            s2 += (x ** 2).mean((1, 2))
            n += 1
            clim[months[t] - 1] += x
            cnt[months[t] - 1] += 1
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-30))
    np.savez(p["stats"], mean=mean, std=std, channels=np.array(meta["channels"]))
    filled = cnt > 0
    clim[filled] /= cnt[filled, None, None, None]
    clim[~filled] = clim[filled].mean(0) if filled.any() else 0.0  # months missing from short train sets
    np.save(p["clim"], clim.astype(np.float32))

    norm = data.Normalizer.load(p["stats"])
    _, po = build.physics_pair(cfg, meta, norm)
    rng = np.random.default_rng(0)
    batches, picked = [], 0
    ys = build.years(d["train_years"])
    while picked < a.diag_samples:
        y = ys[rng.integers(len(ys))]
        arr = np.load(os.path.join(p["cache"], f"{y}.npy"), mmap_mode="r")
        t = int(rng.integers(len(arr)))
        x = norm.norm(np.asarray(arr[t], dtype=np.float32), list(range(C)))
        batches.append(torch.from_numpy(x[None].astype(np.float32)))
        picked += 1
    patch = cfg["model"]["decoder"]["patch"]
    ds = physics.diag_stats(po, batches, patch)
    np.savez(p["diag"], patch=patch, **ds)
    print("stats:", {c: (round(float(m), 4), round(float(s), 4)) for c, m, s in zip(meta["channels"], mean, std)})
    print("diagnostics:", {k: (round(float(m), 4), round(float(s), 4)) for k, m, s in zip(ds["names"], ds["mean"], ds["std"])})


if __name__ == "__main__":
    main()
