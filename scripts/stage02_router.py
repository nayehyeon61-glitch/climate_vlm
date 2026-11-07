"""Stage 02, step 5: routers A / B / C on the same information, calibrated on split C and scored on split D.

    python scripts/stage02_router.py --config configs/stage02_laya_router.yaml [--skip-laya]

    A   fixed path: each expert alone, and the one with the lowest split-B error (best fixed)
    B   numeric MLP router on the same rounded numbers that make Laya's text (plus month, issue hour, lead)
    C0  Laya base checkpoint, zero-shot
    C1  Laya fine-tuned on split B with the encoder frozen (stage02_laya.py train)
    C2  Laya base encoder as a frozen reader: a small head on its cached option states, trained on split B
    oracle  the label itself (upper bound of hard routing)

Every probabilistic router gets one temperature fitted on split C (NLL), apart from its training. On split D:
selection accuracy, Brier, NLL, ECE and reliability bins, error of the chosen expert relative to the oracle and
the base forecast, and RMSE / ACC of the routed fields on the domain (hard routing, and soft routing = probability
mixture of the experts), with 7-day block-bootstrap intervals of the change against the base forecast. ACC uses the
split-A monthly climatology (uncentred, WeatherBench2 style). Cost: time per decision and peak RAM.
-> <docs>/scores.json, <docs>/table.md
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from climate_vlm import build, data  # noqa: E402
from climate_vlm.contract import CLIMODE_VARS  # noqa: E402
from scripts.stage02_common import EXPERTS  # noqa: E402

SPLITS = {"B": "router", "C": "calib", "D": "test"}


def read_table(base):
    import pandas as pd
    return pd.read_parquet(base + ".parquet") if os.path.exists(base + ".parquet") else pd.read_csv(base + ".csv")


class MLPRouter(nn.Module):
    def __init__(self, d_in, hidden, k=3, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden),
                                 nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, k))

    def forward(self, x):
        return self.net(x)


class LayaHead(nn.Module):
    """Shared scorer over Laya's option states: logit_k = MLP(LayerNorm(rep_k))."""

    def __init__(self, d, hidden, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, reps):
        return self.net(self.norm(reps)).squeeze(-1)


def fit(model, x, y, epochs, lr, wd, batch=128, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(y), generator=g)
        for i in range(0, len(y), batch):
            b = perm[i:i + batch]
            loss = nn.functional.cross_entropy(model(x[b]), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def logits_of(model, x):
    with torch.no_grad():
        return model(x)


def temperature(logits, y):
    """Scalar temperature minimising the NLL on the calibration split."""
    lg, yt = torch.as_tensor(np.asarray(logits), dtype=torch.float32), torch.as_tensor(y)
    ts = np.exp(np.linspace(np.log(0.05), np.log(20), 120))
    nll = [nn.functional.cross_entropy(lg / float(t), yt).item() for t in ts]
    return float(ts[int(np.argmin(nll))])


def softmax_np(lg, t=1.0):
    z = np.asarray(lg, np.float64) / t
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def reliability(p, y, bins=10):
    conf, hit = p.max(1), (p.argmax(1) == y).astype(float)
    edges = np.linspace(1 / 3, 1, bins + 1)
    out, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) | (hi == 1))
        if m.any():
            ece += m.mean() * abs(hit[m].mean() - conf[m].mean())
            out.append({"bin": [round(lo, 3), round(hi, 3)], "n": int(m.sum()), "confidence": float(conf[m].mean()),
                        "accuracy": float(hit[m].mean())})
    return float(ece), out


def block_ci(d, groups, n=2000, seed=0):
    """Mean of d and its 95 % interval from a bootstrap over blocks (7-day groups of issue times)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    sums = np.array([d[groups == g].sum() for g in ug])
    cnts = np.array([(groups == g).sum() for g in ug])
    k = rng.integers(0, len(ug), (n, len(ug)))
    bs = sums[k].sum(1) / cnts[k].sum(1)
    return {"mean": float(d.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--config", required=True)
    a.add_argument("--skip-laya", action="store_true", help="numeric routers only")
    a = a.parse_args(argv)
    cfg = build.load_config(a.config)
    bc = build.load_config(cfg["base"]["config"])
    p = build.paths(bc)
    meta = data.load_meta(p["cache"])
    out, docs, rc = cfg["out"], cfg["docs"], cfg["router"]
    os.makedirs(docs, exist_ok=True)
    with open(os.path.join(out, "split_manifest.json")) as fh:
        man = json.load(fh)
    names = man["features"]["names"]
    feats, scores = read_table(os.path.join(out, "physics_features")), read_table(os.path.join(out, "expert_scores"))

    D = {}
    for s in "BCD":
        f, sc = feats[feats.split == s], scores[scores.split == s]
        if not (f.sample_id.values == sc.sample_id.values).all():
            raise ValueError(f"split {s}: features and labels are not aligned")
        num = f[names].to_numpy(np.float64)
        known = np.stack([np.sin(2 * np.pi * f.month / 12), np.cos(2 * np.pi * f.month / 12),
                          np.sin(2 * np.pi * f.issue_hour / 24), np.cos(2 * np.pi * f.issue_hour / 24),
                          f.lead_h / 144.0], 1)
        D[s] = {"num": num, "known": known, "null": ~np.isfinite(num),
                "E": sc[[f"E_{n}" for n in EXPERTS]].to_numpy(np.float64),
                "y": np.array([EXPERTS.index(v) for v in sc.label]), "issue": f.issue_time.values,
                "month": f.month.to_numpy()}
    mu, sd = np.nanmean(D["B"]["num"], 0), np.nanstd(D["B"]["num"], 0) + 1e-6  # split B only

    def x_mlp(s):
        z = (D[s]["num"] - mu) / sd
        z[D[s]["null"]] = 0.0
        return torch.from_numpy(np.concatenate([z, D[s]["null"].mean(1, keepdims=True), D[s]["known"]], 1)
                                .astype(np.float32))

    yt = {s: torch.from_numpy(D[s]["y"]) for s in "BCD"}
    logits, cost, temps = {}, {}, {}
    # B: numeric MLP
    t0 = time.time()
    mlp = fit(MLPRouter(x_mlp("B").shape[1], rc["mlp"]["hidden"]), x_mlp("B"), yt["B"], rc["mlp"]["epochs"],
              float(rc["mlp"]["lr"]), float(rc["mlp"].get("weight_decay", 1e-3)))
    train_sec = time.time() - t0
    t0 = time.time()
    logits["B_mlp"] = {s: logits_of(mlp, x_mlp(s)).numpy() for s in "BCD"}
    n_dec = sum(len(D[s]["y"]) for s in "BCD")
    cost["B_mlp"] = {"sec_per_decision": (time.time() - t0) / n_dec, "train_sec": train_sec,
                     "params": int(sum(q.numel() for q in mlp.parameters()))}
    if not a.skip_laya:
        L = {(m, s): np.load(os.path.join(out, f"laya_{m}_{s}.npz")) for m in ("base", "tuned") for s in "BCD"
             if os.path.exists(os.path.join(out, f"laya_{m}_{s}.npz"))}
        logits["C0_laya_zeroshot"] = {s: np.log(np.clip(L["base", s]["probs"], 1e-9, 1)) for s in "BCD"}
        cost["C0_laya_zeroshot"] = {"sec_per_decision": float(np.mean([L["base", s]["sec_per_decision"] for s in "BCD"])),
                                    "peak_rss_gb": float(max(L["base", s]["rss_gb"] for s in "BCD"))}
        if all(("tuned", s) in L for s in "CD"):
            logits["C1_laya_tuned"] = {s: np.log(np.clip(L["tuned", s]["probs"], 1e-9, 1)) for s in "CD"}
            tsum = os.path.join(out, "laya_tuned", "train_summary.json")
            ts = json.load(open(tsum)) if os.path.exists(tsum) else {}
            cost["C1_laya_tuned"] = {"sec_per_decision": float(np.mean([L["tuned", s]["sec_per_decision"] for s in "CD"])),
                                     "peak_rss_gb": float(max(L["tuned", s]["rss_gb"] for s in "CD")),
                                     "train_sec": ts.get("train_sec"), "train_peak_rss_gb": ts.get("peak_rss_gb")}
        reps = {s: torch.from_numpy(L["base", s]["reps"].astype(np.float32)) for s in "BCD"}
        t0 = time.time()
        head = fit(LayaHead(reps["B"].shape[-1], rc["head"]["hidden"]), reps["B"], yt["B"], rc["head"]["epochs"],
                   float(rc["head"]["lr"]), float(rc["head"].get("weight_decay", 1e-3)))
        cost["C2_laya_cached_head"] = {**cost["C0_laya_zeroshot"], "head_train_sec": time.time() - t0,
                                       "note": "decision cost = the frozen Laya pass that makes the option states"}
        logits["C2_laya_cached_head"] = {s: logits_of(head, reps[s]).numpy() for s in "BCD"}
    for k, lg in logits.items():
        temps[k] = temperature(lg["C"], D["C"]["y"])

    # ---- split D
    d = D["D"]
    n, y = len(d["y"]), d["y"]
    fx = np.load(os.path.join(out, f"fx_{SPLITS['D']}.npz"))
    E_f, O_f = fx["E"], fx["O"]  # (n_ic, 3, L, 5, h, w), (n_ic, L, 5, h, w)
    n_ic, n_lead = O_f.shape[:2]
    if n_ic * n_lead != n:
        raise ValueError("split D: expert fields and samples disagree")
    r5, c5 = fx["rows5"], fx["cols5"]
    lat5 = np.load(os.path.join(p["cache"], "truth_5p625", "lat.npy"))[r5]
    w5 = np.cos(np.deg2rad(lat5))[:, None] * np.ones((1, len(c5)))
    ch = meta["channels"]
    clim = np.asarray(np.load(p["clim"], mmap_mode="r")[:, [ch.index(v) for v in CLIMODE_VARS]], np.float64)
    clim = data.coarsen(clim, bc["data"]["factor"], data.lon0_grid(meta["lon"]))[..., r5[:, None], c5[None, :]]
    cl = clim[d["month"].reshape(n_ic, n_lead) - 1]  # (n_ic, L, 5, h, w), month of the issue time
    t_issue = np.array(d["issue"], "datetime64[h]")
    week = ((t_issue - t_issue.min()) / np.timedelta64(168, "h")).astype(int).reshape(n_ic, n_lead)[:, 0]

    def field_scores(wts):
        """wts (n, 3) expert weights per sample -> per-IC RMSE and ACC (n_ic, L, 5) of the routed field."""
        Y = np.einsum("nlk,nklvhw->nlvhw", wts.reshape(n_ic, n_lead, 3), E_f)
        rm = np.sqrt((((Y - O_f) ** 2) * w5).sum((-1, -2)) / w5.sum())
        af, ao = Y - cl, O_f - cl
        acc = (af * ao * w5).sum((-1, -2)) / np.sqrt((af ** 2 * w5).sum((-1, -2)) * (ao ** 2 * w5).sum((-1, -2)))
        return rm, acc

    onehot = lambda k: np.eye(3)[k]  # noqa: E731
    best_fixed = int(D["B"]["E"].mean(0).argmin())
    methods = {f"A_{e}": onehot(np.full(n, k)) for k, e in enumerate(EXPERTS)}
    methods["A_best_fixed"] = onehot(np.full(n, best_fixed))
    methods["oracle"] = onehot(y)
    res = {"domain": man["domain"]["label"], "test_samples": n, "leads": [int(h) for h in fx["leads"]],
           "vars": list(CLIMODE_VARS), "best_fixed": EXPERTS[best_fixed], "labels": man["labels"], "cost": cost,
           "temperature_split_C": temps, "routers": {}, "fields": {}}
    E_d = d["E"]
    for k, lg in logits.items():
        pr = softmax_np(lg["D"], temps[k])
        hard = pr.argmax(1)
        ece, bins = reliability(pr, y)
        pc = softmax_np(lg["C"], temps[k])
        res["routers"][k] = {
            "accuracy": float((hard == y).mean()), "accuracy_split_C": float((pc.argmax(1) == D["C"]["y"]).mean()),
            "brier": float(((pr - onehot(y)) ** 2).sum(1).mean()),
            "nll": float(-np.log(np.clip(pr[np.arange(n), y], 1e-12, 1)).mean()), "ece": ece, "reliability": bins,
            "error_ratio_to_oracle": float(E_d[np.arange(n), hard].mean() / E_d.min(1).mean()),
            "error_ratio_to_baseline": float(E_d[np.arange(n), hard].mean() / E_d[:, 0].mean()),
            "choice_share": {e: float(np.mean(hard == j)) for j, e in enumerate(EXPERTS)}}
        methods[f"{k}_hard"] = onehot(hard)
        methods[f"{k}_soft"] = pr
    prior = np.bincount(D["B"]["y"], minlength=3) / len(D["B"]["y"])
    res["majority_class_accuracy"] = float((y == prior.argmax()).mean())
    rm0, _ = field_scores(methods["A_baseline"])
    e_base = E_d[:, 0].reshape(n_ic, n_lead).mean(1)
    for name, wts in methods.items():
        rm, ac = field_scores(wts)
        e_sel = (wts * E_d).sum(1).reshape(n_ic, n_lead).mean(1)
        res["fields"][name] = {
            "E": float(e_sel.mean()),
            "rmse": {str(h): {v: float(rm[:, j, i].mean()) for i, v in enumerate(CLIMODE_VARS)}
                     for j, h in enumerate(res["leads"])},
            "acc": {str(h): {v: float(ac[:, j, i].mean()) for i, v in enumerate(CLIMODE_VARS)}
                    for j, h in enumerate(res["leads"])},
            "E_minus_baseline_ci95": block_ci(e_sel - e_base, week),
            "rmse_minus_baseline_ci95": {v: block_ci((rm - rm0)[:, :, i].mean(1), week)
                                         for i, v in enumerate(CLIMODE_VARS)}}
    with open(os.path.join(docs, "scores.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    md = table(res)
    with open(os.path.join(docs, "table.md"), "w") as fh:
        fh.write(md)
    print(md)
    return res


def table(res):
    h = str(res["leads"][0])
    L = [f"Domain: {res['domain']}; {res['test_samples']} test decisions (split D); lead {h} h.", "",
         "## Routers on split D", "",
         "| router | accuracy (D) | accuracy (C) | Brier | NLL | ECE | E / oracle | E / baseline | "
         "choices baseline / advection / moisture |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for k, r in res["routers"].items():
        L.append(f"| {k} | {r['accuracy']:.3f} | {r['accuracy_split_C']:.3f} | {r['brier']:.3f} | {r['nll']:.3f} | "
                 f"{r['ece']:.3f} | {r['error_ratio_to_oracle']:.3f} | {r['error_ratio_to_baseline']:.3f} | "
                 + " / ".join(f"{v:.2f}" for v in r["choice_share"].values()) + " |")
    L += ["", f"Majority class of split B on split D: accuracy {res['majority_class_accuracy']:.3f}. "
              f"Best fixed expert on split B: {res['best_fixed']}.", "",
          "## Routed fields on the domain (split D)", "",
          "| method | E | E - baseline [95 % CI] | " + " | ".join(f"RMSE {v}" for v in res["vars"]) + " | ACC z500 | "
          "ACC t850 | RMSE z500 - baseline [95 % CI] |", "|---|---:|---|" + "---:|" * (len(res["vars"]) + 2) + "---|"]
    for name, f in res["fields"].items():
        c, cz = f["E_minus_baseline_ci95"], f["rmse_minus_baseline_ci95"]["z500"]
        L.append(f"| {name} | {f['E']:.4f} | {c['mean']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] | "
                 + " | ".join(f"{f['rmse'][h][v]:.4g}" for v in res["vars"])
                 + f" | {f['acc'][h]['z500']:.4f} | {f['acc'][h]['t850']:.4f} | "
                 f"{cz['mean']:+.2f} [{cz['lo']:+.2f}, {cz['hi']:+.2f}] |")
    L += ["", "## Labels", "", "```", json.dumps({k: v for k, v in res["labels"].items() if k in "BCD"}, indent=1),
          "```", "", "## Temperatures (split C) and cost", "", "```",
          json.dumps({"temperature": res["temperature_split_C"], "cost": res["cost"]}, indent=1), "```", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
