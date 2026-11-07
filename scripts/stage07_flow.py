"""Stage 07, steps 2-4: train a flow-matching forecast expert per arm, draw members on 2020, score them.

    python scripts/stage07_flow.py --config configs/stage07_flow_ensemble.yaml train --arm fm_vla
    python scripts/stage07_flow.py --config configs/stage07_flow_ensemble.yaml sample --arm fm_vla
    python scripts/stage07_flow.py --config configs/stage07_flow_ensemble.yaml score

Arms (configs/stage07_flow_ensemble.yaml) differ by what the expert sees: the decoder mean mu, the issue-time state
x0, the VLM's decoder memory (cross-attention), and which field the residual is taken from (mu or x0). The Gaussian
head of the decoder (mu, sigma) and persistence are scored alongside, and ClimODE from pg-flow's reference file.

Members go to <out>/<arm>/members_2020.npz in pg-flow's layout (E_minus_x0 (n_ic, M, L, 5, h, w) f16, O, X0, ics,
leads), so pg-flow's scorers read them too. Scores (docs/stage07/scores.json, table.md) follow compare_climode.py:
RMSE of the ensemble mean, ACC, Gaussian CRPS from the member mean and spread (ClimODE's form), the fair ensemble
CRPS in the same min-max units, spread, spread / RMSE ((M + 1) / M corrected) and the share of truths outside the
member range (2 / (M + 1) for a calibrated ensemble).
"""

import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from climate_vlm import build  # noqa: E402
from climate_vlm.flow import FlowExpert  # noqa: E402
from scripts.compare_climode import REF_VARS, boot_ci, climode_scores  # noqa: E402


def load_cache(cfg, split):
    d = os.path.join(cfg["out"], "cache", split)
    info = json.load(open(os.path.join(d, "meta.json")))
    if info["done"] < info["n"]:
        raise RuntimeError(f"the {split} cache is not finished ({info['done']}/{info['n']})")
    arr = {k[:-4]: np.load(os.path.join(d, k), mmap_mode="r") for k in os.listdir(d) if k.endswith(".npy")}
    return arr, info


def lat_weights(cfg):
    bc = build.load_config(cfg["base"]["config"])
    lat = np.load(os.path.join(build.paths(bc)["cache"], "truth_5p625", "lat.npy"))
    w = np.cos(np.deg2rad(lat))
    return lat, w / w.mean()


def base_of(arm, arr, idx):
    return np.asarray(arr["mu"][idx] if arm["base"] == "mu" else arr["x0"][idx][:, None].repeat(len(arr["y"][0]), 1),
                      np.float32)


def residual_scale(arm, arr, w, n=2000):
    """(L, 5) area-weighted rms of y - base over (up to n) training windows: the target's unit."""
    idx = np.linspace(0, len(arr["y"]) - 1, min(n, len(arr["y"]))).astype(int)
    r = np.asarray(arr["y"][idx], np.float32) - base_of(arm, arr, idx)
    return np.sqrt((r ** 2 * w[:, None]).mean((0, -2, -1))).astype(np.float32)


def batch(arm, arr, idx, s):
    """Tensors for windows idx: scaled target r, and the conditions the arm uses."""
    y = np.asarray(arr["y"][idx], np.float32)
    r = (y - base_of(arm, arr, idx)) / s[None, :, :, None, None]
    cond = {"mu": torch.from_numpy(np.asarray(arr["mu"][idx], np.float32)) if arm.get("use_mu") else None,
            "x0": torch.from_numpy(np.asarray(arr["x0"][idx], np.float32)) if arm.get("use_x0") else None,
            "memory": torch.from_numpy(np.asarray(arr["memory"][idx], np.float32)) if arm.get("use_mem") else None}
    return torch.from_numpy(r), cond


def make_model(fc, arm, arr):
    L, V, H, W = arr["y"].shape[1:]
    return FlowExpert(L, V, H, W, patch=tuple(fc["patch"]), dim=fc["dim"], depth=fc["depth"], heads=fc["heads"],
                      mem_dim=arr["memory"].shape[-1], use_mem=bool(arm.get("use_mem")),
                      use_mu=bool(arm.get("use_mu")), use_x0=bool(arm.get("use_x0")))


def cmd_train(cfg, a):
    fc, arm = cfg["flow"], cfg["arms"][a.arm]
    torch.manual_seed(fc.get("seed", 0))
    rng = np.random.default_rng(fc.get("seed", 0))
    tr, _ = load_cache(cfg, "train")
    va, _ = load_cache(cfg, "val")
    lat, w = lat_weights(cfg)
    s = residual_scale(arm, tr, w)
    model = make_model(fc, arm, tr)
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=fc["lr"], weight_decay=fc.get("weight_decay", 0.0))
    steps, warm = int(a.steps or fc["steps"]), int(fc.get("warmup", 100))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(k, steps) / steps)))
    out = os.path.join(cfg["out"], a.arm)
    os.makedirs(out, exist_ok=True)
    wt = torch.from_numpy(w.astype(np.float32))
    nv = min(len(va["y"]), int(fc.get("val_windows", 64)))
    gv = torch.Generator().manual_seed(1)
    tv = torch.rand(nv, generator=gv)
    rv, cv = batch(arm, va, np.arange(nv), s)
    log, t0, start = [], time.time(), 1
    rpath = os.path.join(out, "train_resume.pt")
    if os.path.exists(rpath):  # written at every validation: a stopped container loses at most val_every steps
        r = torch.load(rpath, map_location="cpu", weights_only=False)
        model.load_state_dict(r["model"])
        ema.load_state_dict(r["ema"])
        opt.load_state_dict(r["opt"])
        sched.load_state_dict(r["sched"])
        torch.set_rng_state(r["torch_rng"])
        rng.bit_generator.state = r["np_rng"]
        log, start = r["log"], r["step"] + 1
        print(f"{a.arm}: resumed at step {r['step']}", flush=True)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{a.arm}: {n_par / 1e6:.2f} M parameters, {len(tr['y'])} training windows, {steps} steps", flush=True)

    def val_loss(m):
        torch.manual_seed(2)
        with torch.no_grad():
            return float(np.mean([m.loss(rv[i:i + 16], wt, t=tv[i:i + 16],
                                         **{k: None if v is None else v[i:i + 16] for k, v in cv.items()})
                                  for i in range(0, nv, 16)]))

    for step in range(start, steps + 1):
        idx = np.sort(rng.choice(len(tr["y"]), fc["batch"], replace=False))
        r, cond = batch(arm, tr, idx, s)
        loss = model.loss(r, wt, **cond)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            d = min(fc.get("ema", 0.999), (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        if step % fc.get("log_every", 100) == 0 or step == 1:
            print(f"{a.arm} step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        if step % fc.get("val_every", 500) == 0 or step == steps:
            v = val_loss(ema)
            log.append({"step": step, "train": float(loss.item()), "val_ema": v, "seconds": round(time.time() - t0)})
            print(f"{a.arm} step {step} val(ema) {v:.4f}", flush=True)
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "torch_rng": torch.get_rng_state(),
                        "np_rng": rng.bit_generator.state, "log": log, "step": step}, rpath + ".tmp")
            os.replace(rpath + ".tmp", rpath)
    torch.save({"state": ema.state_dict(), "scale": s, "arm": arm, "flow": fc, "log": log},
               os.path.join(out, "flow.pt"))
    with open(os.path.join(out, "train_log.json"), "w") as fh:
        json.dump({"arm": a.arm, "parameters": n_par, "log": log}, fh, indent=1)
    if os.path.exists(rpath):
        os.remove(rpath)
    return log


def cmd_sample(cfg, a):
    fc = cfg["flow"]
    te, info = load_cache(cfg, "test")
    ck = torch.load(os.path.join(cfg["out"], a.arm, "flow.pt"), map_location="cpu", weights_only=False)
    arm, s = ck["arm"], ck["scale"]
    model = make_model(fc, arm, te)
    model.load_state_dict(ck["state"])
    model.eval()
    M, n = int(fc.get("members", 8)), len(te["y"])
    m5, s5 = np.array(info["mean"])[:, None, None], np.array(info["std"])[:, None, None]
    phys = lambda x: np.asarray(x, np.float64) * s5 + m5  # noqa: E731
    gen = torch.Generator().manual_seed(fc.get("seed", 0))
    E, t0 = [], time.time()
    for i in range(0, n, a.batch):
        idx = np.arange(i, min(n, i + a.batch))
        _, cond = batch(arm, te, idx, s)
        r = model.sample(M, steps=int(fc.get("sample_steps", 10)), generator=gen, **cond).numpy()
        E.append(base_of(arm, te, idx)[:, None] + r * s[None, None, :, :, None, None])
        print(f"{a.arm}: sampled {idx[-1] + 1}/{n} windows, {time.time() - t0:.0f} s", flush=True)
    E = phys(np.concatenate(E))  # (n, M, L, 5, h, w)
    X0, O = phys(te["x0"]), phys(te["y"])
    path = os.path.join(cfg["out"], a.arm, "members_2020.npz")
    np.savez(path, E_minus_x0=(E - X0[:, None, None]).astype(np.float16), O=O.astype(np.float32),
             X0=X0.astype(np.float32), ics=np.array(info["ics"]), leads=np.array(info["leads"]),
             vars=np.array(REF_VARS))
    return path


def crps_fair(E, O):
    """(M, ...) members, (...) truth -> fair CRPS elementwise (Ferro 2014)."""
    M = E.shape[0]
    skill = np.abs(E - O[None]).mean(0)
    spread = np.abs(E[:, None] - E[None]).sum((0, 1)) / (2 * M * (M - 1))
    return skill - spread


def cmd_score(cfg, a):
    te, info = load_cache(cfg, "test")
    ref = json.load(open(cfg["reference"])) if cfg.get("reference") and os.path.exists(cfg["reference"]) else None
    ics = [int(i) for i in info["ics"]]
    use = [int(i) for i in ref["ics"]] if ref else ics
    rows = [ics.index(i) for i in use if i in ics]
    if len(rows) != len(use):
        raise ValueError("reference ICs missing from the test cache")
    bc = build.load_config(cfg["base"]["config"])
    tdir = os.path.join(build.paths(bc)["cache"], "truth_5p625")
    lat = np.load(os.path.join(tdir, "lat.npy"))
    mm = json.load(open(os.path.join(tdir, "minmax_2006_2018.json")))
    mx, mn = np.array(mm["max"]), np.array(mm["min"])
    year = info["years"][rows[0]]
    clim = np.asarray(np.load(os.path.join(tdir, f"{year}.npy"), mmap_mode="r"), np.float64).mean(0)
    m5, s5 = np.array(info["mean"])[:, None, None], np.array(info["std"])[:, None, None]
    phys = lambda x: np.asarray(x, np.float64) * s5 + m5  # noqa: E731
    leads = info["leads"]
    w = np.cos(np.deg2rad(lat))
    w = (w / w.mean())[:, None]
    sc = (mx - mn)[:, None, None]
    systems = {"gaussian_head": None, "persistence": None}
    for name in cfg["arms"]:
        f = os.path.join(cfg["out"], name, "members_2020.npz")
        if os.path.exists(f):
            systems[name] = f
    per = {}
    for name, f in systems.items():
        Ez = np.load(f)["E_minus_x0"] if f else None  # read once: an npz member is re-read at every access
        for e, r in enumerate(rows):
            O, X0 = phys(te["y"][r]), phys(te["x0"][r])
            if name == "gaussian_head":
                if "sigma" not in te:
                    continue
                P, S, E = phys(te["mu"][r]), np.asarray(te["sigma"][r], np.float64) * s5, None
            elif name == "persistence":
                P, S, E = np.broadcast_to(X0, O.shape), np.full(O.shape, 1e-6), None
            else:
                E = np.asarray(Ez[r], np.float64) + X0[None, None]  # (M, L, 5, h, w)
                P, S = E.mean(0), E.std(0, ddof=1)
            for j in range(len(leads)):
                sc_ = climode_scores(P[j], O[j], S[j], clim, lat, mx, mn)
                if E is not None:
                    M = E.shape[0]
                    sc_["crps_ens"] = (crps_fair((E[:, j] - mn[:, None, None]) / sc, (O[j] - mn[:, None, None]) / sc)
                                       ).mean((-1, -2))
                    lo, hi = E[:, j].min(0), E[:, j].max(0)
                    sc_["outside"] = ((O[j] < lo) | (O[j] > hi)).mean((-1, -2))
                    sc_["spread"] = np.sqrt((S[j] ** 2 * w).mean((-1, -2)) * (M + 1) / M)
                for met, v in sc_.items():
                    per.setdefault(name, {}).setdefault(met, np.zeros((len(rows), len(leads), 5)))[e, j] = v
    res = {"n_ic": len(rows), "ics": use, "leads": leads, "vars": REF_VARS, "members": int(cfg["flow"]["members"]),
           "scores": {}}
    for name, mets in per.items():
        d = res["scores"].setdefault(name, {})
        for met, arr in mets.items():
            for j, h in enumerate(leads):
                for k, v in enumerate(REF_VARS):
                    d.setdefault(str(h), {}).setdefault(met, {})[v] = {
                        "mean": float(arr[:, j, k].mean()),
                        "ci95": boot_ci(arr[:, j, k]) if name not in ("persistence",) else None}
        if "spread" in mets and name != "persistence":
            ratio = np.sqrt((mets["spread"] ** 2).mean(0)) / np.sqrt((mets["rmse"] ** 2).mean(0))
            for j, h in enumerate(leads):
                for k, v in enumerate(REF_VARS):
                    d[str(h)].setdefault("spread_skill", {})[v] = {"mean": float(ratio[j, k]), "ci95": None}
    if ref is not None:
        for name in ("climode",):
            if name in ref["scores"]:
                res["scores"]["climode"] = {str(h): {met: {v: {"mean": ref["scores"][name][str(h)][met][v]["mean"],
                                                               "ci95": None} for v in REF_VARS}
                                                     for met in ("rmse", "acc", "crps_gauss", "spread")
                                                     if met in ref["scores"][name][str(h)]}
                                            for h in leads if str(h) in ref["scores"][name]}
                for h, d in res["scores"]["climode"].items():  # ratio of the IC means (no per-IC values)
                    if "spread" in d:
                        d["spread_skill"] = {v: {"mean": d["spread"][v]["mean"] / d["rmse"][v]["mean"], "ci95": None}
                                             for v in REF_VARS}
        p_ref = ref["scores"]["persistence"]
        res["persistence_max_rel_diff_vs_reference"] = max(
            abs(p_ref[str(h)]["rmse"][v]["mean"] - res["scores"]["persistence"][str(h)]["rmse"][v]["mean"])
            / p_ref[str(h)]["rmse"][v]["mean"] for h in leads for v in REF_VARS if str(h) in p_ref)
    os.makedirs(cfg["docs"], exist_ok=True)
    with open(os.path.join(cfg["docs"], "scores.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    md = table(res)
    with open(os.path.join(cfg["docs"], "table.md"), "w") as fh:
        fh.write(md)
    print(md, flush=True)
    return res


def table(res):
    leads = res["leads"]
    show = [h for h in (24, 72, 144) if h in leads] or leads
    s = [f"# Stage 07 scores, 2020 (n = {res['n_ic']} ICs, {res['members']} members, WB2 64x32)", ""]
    for met, title, fmt in (("rmse", "RMSE of the ensemble mean", "{:.3g}"),
                            ("crps_gauss", "Gaussian CRPS (ClimODE min-max units)", "{:.4f}"),
                            ("crps_ens", "fair ensemble CRPS (same units)", "{:.4f}"),
                            ("spread_skill", "spread / RMSE", "{:.2f}"),
                            ("outside", "truth outside the member range (calibrated: 2 / (M + 1))", "{:.2f}")):
        names = [n for n, d in res["scores"].items() if str(leads[0]) in d and met in d[str(leads[0])]]
        if not names:
            continue
        s += [f"## {title}", "", "| system | " + " | ".join(f"{v} {h} h" for v in res["vars"] for h in show) + " |",
              "|---" * (1 + len(res["vars"]) * len(show)) + "|"]
        for n in names:
            d = res["scores"][n]
            s.append(f"| {n} | " + " | ".join(fmt.format(d[str(h)][met][v]["mean"]) if str(h) in d and met in d[str(h)]
                                              else "" for v in res["vars"] for h in show) + " |")
        s.append("")
    if "persistence_max_rel_diff_vs_reference" in res:
        s.append(f"Persistence against the reference file: max relative difference "
                 f"{res['persistence_max_rel_diff_vs_reference']:.1e}.")
    return "\n".join(s) + "\n"


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--config", required=True)
    a.add_argument("cmd", choices=["train", "sample", "score"])
    a.add_argument("--arm")
    a.add_argument("--steps", type=int, default=0, help="override flow.steps (tests)")
    a.add_argument("--batch", type=int, default=4, help="windows per sampling batch")
    a = a.parse_args(argv)
    cfg = build.load_config(a.config)
    if a.cmd in ("train", "sample") and a.arm not in cfg["arms"]:
        raise SystemExit(f"--arm must be one of {list(cfg['arms'])}")
    return {"train": cmd_train, "sample": cmd_sample, "score": cmd_score}[a.cmd](cfg, a)


if __name__ == "__main__":
    main()
