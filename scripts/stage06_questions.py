"""Stage 06: typed questions about the forecast, answered by Laya's vega reader and by baselines.

The numeric model (stage 04's advection arm) forecasts first. A short text of what it has at the issue time and of
its forecast is the state; Laya answers typed questions about the domain at each lead with probabilities, a
conformal set and a route (auto / review). Labels come from the WB2 64x32 truth: stage02_forecasts.py's O at the lead
minus the truth at the issue time. Splits (stage 02): B (2016) fits the heads and the baselines, C (2019)
calibrates, D (2020) is scored.

    python scripts/stage06_questions.py --config configs/stage06_laya_questions.yaml rows       # states, labels
    python scripts/stage06_questions.py --config configs/stage06_laya_questions.yaml baselines  # climatology, rule, LR
    python scripts/stage06_questions.py --config configs/stage06_laya_questions.yaml laya       # vega: zero / cal / fit
    python scripts/stage06_questions.py --config configs/stage06_laya_questions.yaml report     # scores on D

Questions per lead L (thresholds from the 2016 truth, written into the question):
    drop_L     yes/no   the domain-mean T850 at +L h is at least tau_L K below now (tau_L: 20 % of 2016 cases)
    z500_L     score    domain-mean Z500 change to +L h: below / between / above the 2016 terciles
    coldest_L  choice   the quarter (NW, NE, SW, SE) with the lowest T850 change to +L h

Every number in a state is also a feature of the logistic baselines, rounded as written, so Laya and the baselines
see the same information. Laya's reads are cached in <out>/vega_cache.sqlite (resumable; a re-run costs nothing).
The engine reads all questions of a state in one pass; any subset it is asked later (vega's fit compares heads with
the engine one question at a time, and mode auto asks the engine only the questions without a head) is taken from
that pass.
"""

import argparse
import copy
import hashlib
import json
import os
import pickle
import sqlite3
import sys
import time

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from climate_vlm import build, data  # noqa: E402
from scripts.stage02_common import G, domain_label, lr_grid, sub_regions  # noqa: E402
from scripts.stage02_forecasts import Truth  # noqa: E402
from scripts.stage02_router import block_ci  # noqa: E402

PERIODS = {"B": "router", "C": "calib", "D": "test"}
METHODS = ["clim", "rule", "lr_fc", "lr_all", "lr_all_full", "laya_zero", "laya_cal", "laya_fit"]
LABEL = {"clim": "climatology (2016)", "rule": "forecast read as written", "lr_fc": "logistic, forecast number",
         "lr_all": "logistic, every number", "lr_all_full": "logistic, every number, all 2016",
         "laya_zero": "Laya zero-shot", "laya_cal": "Laya + calibration (2019)",
         "laya_fit": "Laya heads (2016) + calibration (2019)"}


# ----------------------------------------------------------------------------- questions and states
def labels_for(q):
    t = q["type"]
    return ["false", "true"] if t == "noul" else [str(i) for i in range(len(q["criteria"]))] if t == "score" \
        else list(q["criteria"])


def num(x, d, sign=False):
    v = round(float(x), d)
    v = 0.0 if v == 0 else v
    return v, (f"{v:+.{d}f}" if sign else f"{v:.{d}f}")


def quantities(z, truth, meta, norm, factor, spec):
    """Per issue time, on the domain: what the state says (model side, rounded as written) and the truth changes."""
    ch, fm, vs = meta["channels"], list(z["fmaps"]), list(z["vars"])
    r5, c5 = z["rows5"], z["cols5"]
    lat = lr_grid(meta, factor)[0]
    w = np.cos(np.deg2rad(lat[r5]))[:, None] * np.ones((1, len(c5)))
    subs = sub_regions(len(r5), len(c5), spec["sub"])
    crop = lambda a: np.asarray(a)[..., r5[:, None], c5[None, :]]  # noqa: E731

    def means(a):  # (..., h, w) -> (..., 1 + n_sub): domain mean, then each quarter
        out = [(a * w).sum((-1, -2)) / w.sum()]
        for _, rs, cs in subs:
            out.append((a[..., rs, cs] * w[rs, cs]).sum((-1, -2)) / w[rs, cs].sum())
        return np.stack(out, -1)

    phys = lambda a, c: a * norm.std[ch.index(c)] + norm.mean[ch.index(c)]  # noqa: E731
    std = lambda c: norm.std[ch.index(c)]  # noqa: E731
    F = z["F"]
    t0, z0 = phys(crop(F[:, fm.index("t850")]), "t850"), phys(crop(F[:, fm.index("z500")]), "z500")
    it, iz = vs.index("t850"), vs.index("z500")
    P, O = crop(z["P"]), crop(z["O"])  # (n, L, 5, h, w)
    T0 = np.stack([crop(truth(int(h)))[[it, iz]] for h in z["issue"]])  # truth at the issue time
    return {
        "issue": np.asarray(z["issue"], np.int64), "leads": [int(h) for h in z["leads"]],
        "subs": [n for n, _, _ in subs],
        "T": means(t0 - 273.15), "Z": means(z0 / G)[:, 0],
        "adv": means(crop(F[:, fm.index("tadv850")]))[:, 0],
        "dT24": means(crop(F[:, fm.index("dt850_24h")]) * std("t850"))[:, 0],
        "dZ24": means(crop(F[:, fm.index("dz500_24h")]) * std("z500") / G)[:, 0],
        "fdT": means(P[:, :, it] - t0[:, None]), "fdZ": means((P[:, :, iz] - z0[:, None]) / G)[..., 0],
        "tdT": means(O[:, :, it] - T0[:, None, 0]), "tdZ": means((O[:, :, iz] - T0[:, None, 1]) / G)[..., 0],
    }


def make_questions(qb, quantile=0.2):
    """Questions and their thresholds from the 2016 truth (split B)."""
    qs, th = {}, {}
    for j, L in enumerate(qb["leads"]):
        tau = max(0.1, num(-np.quantile(qb["tdT"][:, j, 0], quantile), 1)[0])
        q1, q2 = (num(v, 0)[0] for v in np.quantile(qb["tdZ"][:, j], [1 / 3, 2 / 3]))
        th[L] = {"tau_K": tau, "z500_terciles_m": [q1, q2]}
        qs[f"drop_{L}"] = {"type": "noul", "instructions":
                           f"Will the 850 hPa temperature averaged over the domain be at least {tau:.1f} K lower at "
                           f"+{L} h than now?"}
        qs[f"z500_{L}"] = {"type": "score", "instructions":
                           f"How will the 500 hPa height averaged over the domain change from now to +{L} h?",
                           "criteria": [f"changes by less than {q1:+.0f} m", f"changes by {q1:+.0f} to {q2:+.0f} m",
                                        f"changes by more than {q2:+.0f} m"]}
        qs[f"coldest_{L}"] = {"type": "choice", "instructions":
                              f"Which quarter of the domain will have the largest 850 hPa temperature fall (or the "
                              f"smallest rise) from now to +{L} h?",
                              "criteria": {"NW": "the north-west quarter", "NE": "the north-east quarter",
                                           "SW": "the south-west quarter", "SE": "the south-east quarter"}}
    return qs, th


def answer_from(dT, dZ, subs, th, L):
    """The answer the numbers give: truth changes -> label, forecast changes (as written) -> the rule."""
    tau, (q1, q2) = th[L]["tau_K"], th[L]["z500_terciles_m"]
    return {f"drop_{L}": "true" if dT[0] <= -tau else "false",
            f"z500_{L}": "0" if dZ < q1 else "2" if dZ > q2 else "1",
            f"coldest_{L}": subs[int(np.argmin(dT[1:]))]}


def rows_of(q, th, dom):
    """One row per issue time: state text, gold labels, the forecast rule and the features (as written)."""
    out, subs = [], q["subs"]
    for i, h in enumerate(q["issue"]):
        x, w = [], {}

        def put(key, v, d, sign=False):
            v, s = num(v, d, sign)
            x.append(v)
            w[key] = s
            return v

        doy = 2 * np.pi * (int(np.datetime64(int(h), "h").astype("datetime64[D]").astype(int)) % 365.25) / 365.25
        x += [round(float(np.sin(doy)), 3), round(float(np.cos(doy)), 3)]
        for k in range(len(subs) + 1):
            put(f"T{k}", q["T"][i, k], 1)
        put("Z", q["Z"][i], 0)
        put("adv", q["adv"][i], 1, True)
        put("dT24", q["dT24"][i], 1, True)
        put("dZ24", q["dZ24"][i], 0, True)
        date = str(np.datetime64(int(h), "h")).replace("T", " ")
        quarters = lambda p: ", ".join(f"{n} {w[f'{p}{k + 1}']}" for k, n in enumerate(subs))  # noqa: E731
        s = (f"{dom}, issued {date} UTC. Now: T850 {w['T0']} C ({quarters('T')}), Z500 {w['Z']} m, 850 hPa "
             f"temperature advection {w['adv']} K/day; past 24 h: T850 {w['dT24']} K, Z500 {w['dZ24']} m.")
        gold, rule = {}, {}
        for j, L in enumerate(q["leads"]):
            fdT = [put(f"f{L}T{k}", q["fdT"][i, j, k], 1, True) for k in range(len(subs) + 1)]
            fdZ = put(f"f{L}Z", q["fdZ"][i, j], 0, True)
            s += (f" Model forecast, change from now to +{L} h: T850 {w[f'f{L}T0']} K "
                  f"({quarters(f'f{L}T')}), Z500 {w[f'f{L}Z']} m.")
            gold.update(answer_from(q["tdT"][i, j], q["tdZ"][i, j], subs, th, L))
            rule.update(answer_from(np.array(fdT), fdZ, subs, th, L))
        out.append({"issue": int(h), "state": s, "gold": gold, "rule": rule, "x": x})
    return out


def fc_feature_idx(qid, leads, n_sub):
    """Features of `x` that hold the forecast number a question is about (the single-number logistic)."""
    L = int(qid.split("_")[1])
    base = 2 + (n_sub + 1) + 4 + leads.index(L) * (n_sub + 2)  # season, T now, Z / adv / dT24 / dZ24, earlier leads
    if qid.startswith("drop"):
        return [base]
    if qid.startswith("z500"):
        return [base + n_sub + 1]
    return list(range(base + 1, base + 1 + n_sub))


def cmd_rows(cfg, a):
    bc = build.load_config(cfg["base"]["config"])
    p = build.paths(bc)
    meta, norm = data.load_meta(p["cache"]), data.Normalizer.load(p["stats"])
    truth = Truth(os.path.join(p["cache"], "truth_5p625"), p["cache"])
    q = {}
    for sp, name in PERIODS.items():
        z = np.load(os.path.join(cfg["out"], f"fc_{name}.npz"))
        q[sp] = quantities(z, truth, meta, norm, bc["data"]["factor"], cfg["domain"])
    qs, th = make_questions(q["B"], cfg.get("drop_quantile", 0.2))
    dom = domain_label(cfg["domain"])
    res = {"questions": qs, "thresholds": {str(k): v for k, v in th.items()}, "leads": q["B"]["leads"],
           "subs": q["B"]["subs"], "sample": cfg.get("sample", {}),
           "rows": {sp: rows_of(q[sp], th, dom) for sp in PERIODS}}
    with open(os.path.join(cfg["out"], "rows.json"), "w") as fh:
        json.dump(res, fh)
    for sp, rows in res["rows"].items():
        base = {qid: {lab: round(float(np.mean([r["gold"][qid] == lab for r in rows])), 3) for lab in labels_for(qq)}
                for qid, qq in qs.items()}
        print(sp, len(rows), "rows; base rates", json.dumps(base), flush=True)
    print("example state:", res["rows"]["D"][0]["state"], flush=True)
    return res


def load_rows(cfg):
    with open(os.path.join(cfg["out"], "rows.json")) as fh:
        return json.load(fh)


def subset(res, sp):
    k = int(res["sample"].get(PERIODS[sp], 1) or 1)
    return res["rows"][sp][::k]


# ----------------------------------------------------------------------------- baselines
def fit_lr(X, y, k, l2=1e-2):
    """Multinomial logistic regression on standardised features (L-BFGS); returns a predict function."""
    X = np.asarray(X, np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt, yt = torch.tensor((X - mu) / sd), torch.tensor(np.asarray(y), dtype=torch.long)
    W = torch.zeros(X.shape[1], k, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(k, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = Fn.cross_entropy(Xt @ W + b, yt) + l2 * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    W, b = W.detach(), b.detach()
    return lambda Xn: torch.softmax(torch.tensor((np.asarray(Xn, np.float64) - mu) / sd) @ W + b, -1).numpy()


def cmd_baselines(cfg, a):
    res = load_rows(cfg)
    qs, leads, n_sub = res["questions"], res["leads"], len(res["subs"])
    B, Bfull, D = subset(res, "B"), res["rows"]["B"], subset(res, "D")
    out = {m: [{} for _ in D] for m in ("clim", "rule", "lr_fc", "lr_all", "lr_all_full")}
    for qid, q in qs.items():
        labs = labels_for(q)
        y = np.array([labs.index(r["gold"][qid]) for r in B])
        clim = (np.bincount(y, minlength=len(labs)) + 0.5) / (len(y) + 0.5 * len(labs))
        idx = fc_feature_idx(qid, leads, n_sub)
        X, XD = np.array([r["x"] for r in B]), np.array([r["x"] for r in D])
        p_fc = fit_lr(X[:, idx], y, len(labs))(XD[:, idx])
        p_all = fit_lr(X, y, len(labs))(XD)
        yf = np.array([labs.index(r["gold"][qid]) for r in Bfull])
        p_full = fit_lr(np.array([r["x"] for r in Bfull]), yf, len(labs))(XD)
        for i, r in enumerate(D):
            out["clim"][i][qid] = {"p": clim.tolist()}
            out["rule"][i][qid] = {"p": np.eye(len(labs))[labs.index(r["rule"][qid])].tolist()}
            out["lr_fc"][i][qid] = {"p": p_fc[i].tolist()}
            out["lr_all"][i][qid] = {"p": p_all[i].tolist()}
            out["lr_all_full"][i][qid] = {"p": p_full[i].tolist()}
    for m, v in out.items():
        with open(os.path.join(cfg["out"], f"pred_{m}.json"), "w") as fh:
            json.dump(v, fh)
    print("baselines written for", len(D), "test rows", flush=True)
    return out


# ----------------------------------------------------------------------------- Laya (vega)
def _key(*parts):
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


class ReadCache:
    """sqlite cache of vega's two kinds of reads: pooled features per (state, question) and engine passes."""

    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("create table if not exists c (k text primary key, v blob)")
        self.misses, self.t0 = 0, time.time()

    def get(self, k):
        r = self.db.execute("select v from c where k = ?", (k,)).fetchone()
        return None if r is None else pickle.loads(r[0])

    def put(self, k, v):
        self.db.execute("insert or replace into c values (?, ?)", (k, pickle.dumps(v)))
        self.db.commit()
        self.misses += 1
        if self.misses % 25 == 0:
            print(f"  vega reads: {self.misses} new, {time.time() - self.t0:.0f} s", flush=True)


def install_cache(agent, cache, sheet):
    """Route agent.vega's feature reads and engine passes through `cache`; engine subsets of `sheet` come from the
    full pass over `sheet`."""
    vega = getattr(agent, "vega", None)
    if vega is None or not hasattr(vega, "engine"):
        return
    feat, dec = vega.features, vega.engine.decide

    def features(state, question):
        k = _key("f", state, question)
        v = cache.get(k)
        if v is None:
            v = feat(state, question)
            cache.put(k, v)
        return v

    def decide(state, questions, alpha="0.1", risk=None, views=1):
        full = all(qid in sheet and questions[qid] == sheet[qid] for qid in questions)
        ask = sheet if full else questions
        k = _key("e", state, ask, str(float(alpha)), risk, views)
        v = cache.get(k)
        if v is None:
            v = dec(state, ask, alpha=alpha, risk=risk, views=views)
            cache.put(k, v)
        v = copy.deepcopy(v)
        v["answers"] = {qid: v["answers"][qid] for qid in questions}
        return v

    vega.features = features
    vega.engine.decide = decide


def make_agent():
    import laya
    return laya.Agent(vegaml=True)


def laya_rows(rows):
    return [{"state": r["state"], "gold": r["gold"]} for r in rows]


def predict_all(agent, rows, qs, tag):
    out, t0 = [], time.time()
    for i, r in enumerate(rows):
        ans = agent.predict(r["state"], questions=qs)["answers"]
        rec = {}
        for qid, q in qs.items():
            a, labs = ans[qid], labels_for(q)
            pr = a.get("probabilities") or a.get("probs") or {}
            p = np.array([float(pr.get(lab, 0.0)) for lab in labs])
            p = p / p.sum() if p.sum() > 0 else np.full(len(labs), 1 / len(labs))
            rec[qid] = {"p": p.tolist(), "set": a.get("prediction_set"), "abstain": a.get("abstain"),
                        "route": a.get("route"), "readout": a.get("readout")}
        out.append(rec)
        if (i + 1) % 20 == 0 or i + 1 == len(rows):
            print(f"{tag}: {i + 1}/{len(rows)} rows, {time.time() - t0:.0f} s", flush=True)
    return out


def cmd_laya(cfg, a, agent=None):
    res = load_rows(cfg)
    qs, out = res["questions"], cfg["out"]
    B, C, D = subset(res, "B"), subset(res, "C"), subset(res, "D")
    torch.set_num_threads(1)  # vega: one thread is the fastest CPU setting for the Qwen3.5 readers
    agent = agent or make_agent()
    try:
        from laya.vega import _to_vega_questions
        sheet = _to_vega_questions(qs)
    except ImportError:
        sheet = qs
    install_cache(agent, ReadCache(os.path.join(out, "vega_cache.sqlite")), sheet)
    rp = os.path.join(out, "laya_report.json")
    rep = json.load(open(rp)) if os.path.exists(rp) else {}  # arms may run one per call
    rep["rows"] = {"B": len(B), "C": len(C), "D": len(D)}
    path = lambda m: os.path.join(out, f"pred_{m}.json")  # noqa: E731
    if "zero" in a.arms:  # the engine as shipped, its own checkpoint calibration
        agent.calibration = None
        json.dump(predict_all(agent, D, qs, "zero-shot D"), open(path("laya_zero"), "w"))
    if "cal" in a.arms:  # a temperature and a conformal threshold per question from 2019, no heads
        agent.calibration = None
        rep["cal"] = agent.calibrate(laya_rows(C), qs)
        json.dump(predict_all(agent, D, qs, "calibrated D"), open(path("laya_cal"), "w"))
    if "fit" in a.arms:  # heads on 2016, calibration on 2019
        agent.calibration = None
        t0 = time.time()
        rep["fit"] = agent.fit(laya_rows(B), qs, calibration_rows=laya_rows(C))
        rep["fit_seconds"] = round(time.time() - t0)
        agent.save_calibration(os.path.join(out, "laya_fit"))
        json.dump(predict_all(agent, D, qs, "fitted D"), open(path("laya_fit"), "w"))
    with open(rp, "w") as fh:
        json.dump(rep, fh, indent=1, default=str)
    return rep


# ----------------------------------------------------------------------------- scores
def ece(P, y, bins=10):
    conf, hit = P.max(1), (P.argmax(1) == y).astype(float)
    e = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(e)


def per_row(P, y, ordinal):
    """Per-row scores: hit, Brier (sum over options), log loss, and the ranked probability score if ordinal."""
    n, k = P.shape
    oh = np.eye(k)[y]
    d = {"hit": (P.argmax(1) == y).astype(float), "brier": ((P - oh) ** 2).sum(1),
         "nll": -np.log(np.clip(P[np.arange(n), y], 1e-6, 1))}
    if ordinal:
        d["rps"] = ((np.cumsum(P, 1) - np.cumsum(oh, 1))[:, :-1] ** 2).sum(1) / (k - 1)
    return d


def cmd_report(cfg, a):
    res = load_rows(cfg)
    qs, D = res["questions"], subset(res, "D")
    groups = np.array([r["issue"] // (24 * 7) for r in D])
    preds = {}
    for m in METHODS:
        f = os.path.join(cfg["out"], f"pred_{m}.json")
        if os.path.exists(f):
            preds[m] = json.load(open(f))
    sc, rowsc = {}, {}
    for m, pr in preds.items():
        sc[m], rowsc[m] = {}, {}
        for qid, q in qs.items():
            labs = labels_for(q)
            y = np.array([labs.index(r["gold"][qid]) for r in D])
            P = np.array([p[qid]["p"] for p in pr])
            d = per_row(P, y, q["type"] == "score")
            rowsc[m][qid] = d
            s = {k: float(v.mean()) for k, v in d.items()}
            s["ece"] = ece(P, y)
            if m == "rule":
                s["nll"] = None
            if "set" in pr[0][qid] and pr[0][qid]["set"] is not None:
                sets = [p[qid]["set"] or [] for p in pr]
                s["set_coverage"] = float(np.mean([labs[t] in st for t, st in zip(y, sets)]))
                s["set_size"] = float(np.mean([len(st) for st in sets]))
                auto = np.array([p[qid].get("route") == "auto" for p in pr])
                s["abstain"] = float(np.mean([bool(p[qid].get("abstain")) for p in pr]))
                s["auto_share"] = float(auto.mean())
                s["auto_accuracy"] = float(d["hit"][auto].mean()) if auto.any() else None
                s["readout"] = sorted({str(p[qid].get("readout")) for p in pr})
            sc[m][qid] = s
    if "clim" in sc:
        for m in sc:
            for qid in qs:
                sc[m][qid]["bss"] = 1 - sc[m][qid]["brier"] / sc["clim"][qid]["brier"]
    comp = {}
    for m1, m2 in (("laya_fit", "lr_all"), ("laya_zero", "rule"), ("laya_cal", "laya_zero"), ("laya_fit", "laya_cal"),
                   ("lr_all", "rule"), ("lr_all", "clim")):
        if m1 in rowsc and m2 in rowsc:
            for metric in ("brier", "hit"):
                dd = np.mean([rowsc[m1][qid][metric] - rowsc[m2][qid][metric] for qid in qs], 0)
                comp[f"{m1} - {m2}: {metric}"] = block_ci(dd, groups)
    out = {"n_test": len(D), "questions": qs, "thresholds": res["thresholds"], "scores": sc, "differences": comp}
    os.makedirs(cfg["docs"], exist_ok=True)
    with open(os.path.join(cfg["docs"], "scores.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    lap = os.path.join(cfg["out"], "laya_report.json")
    if os.path.exists(lap):
        with open(lap) as fh, open(os.path.join(cfg["docs"], "laya_report.json"), "w") as g:
            g.write(fh.read())
    with open(os.path.join(cfg["docs"], "table.md"), "w") as fh:
        fh.write(table(out, D, preds))
    print(table(out, D, preds)[:3000], flush=True)
    return out


def table(out, D, preds):
    qs, sc = out["questions"], out["scores"]
    f = lambda v, d=3: "" if v is None else f"{v:.{d}f}"  # noqa: E731
    s = [f"# Stage 06 scores, 2020 (n = {out['n_test']} issue times)", "",
         "Accuracy / Brier skill score vs climatology (Brier summed over options) / ECE.", "",
         "| method | " + " | ".join(qs) + " | mean |", "|---" * (len(qs) + 2) + "|"]
    for m in METHODS:
        if m not in sc:
            continue
        cells = [f"{f(sc[m][q]['hit'], 2)} / {f(sc[m][q].get('bss'), 2)} / {f(sc[m][q]['ece'], 2)}" for q in qs]
        mean = [np.mean([sc[m][q][k] for q in qs]) for k in ("hit", "bss", "ece")]
        s.append(f"| {LABEL[m]} | " + " | ".join(cells) + f" | {mean[0]:.2f} / {mean[1]:.2f} / {mean[2]:.2f} |")
    s += ["", "Laya sets and routes (coverage of the 90 % set / mean set size / share routed auto / accuracy when auto "
              "/ readout):", "", "| method | " + " | ".join(qs) + " |", "|---" * (len(qs) + 1) + "|"]
    for m in ("laya_zero", "laya_cal", "laya_fit"):
        if m in sc:
            s.append(f"| {LABEL[m]} | " + " | ".join(
                f"{f(sc[m][q].get('set_coverage'), 2)} / {f(sc[m][q].get('set_size'), 2)} / "
                f"{f(sc[m][q].get('auto_share'), 2)} / {f(sc[m][q].get('auto_accuracy'), 2)} / "
                f"{','.join(sc[m][q].get('readout', []))}" for q in qs) + " |")
    s += ["", "Differences averaged over the questions (95 % interval, 7-day block bootstrap):", ""]
    for k, v in out["differences"].items():
        s.append(f"- {k}: {v['mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]")
    s += ["", "## Questions", ""]
    for qid, q in qs.items():
        crit = q.get("criteria")
        s.append(f"- `{qid}` ({q['type']}): {q['instructions']}" + (f" Options: {crit}" if crit else ""))
    s += ["", "## Examples", ""]
    for i in range(min(3, len(D))):
        r = D[i]
        s += [f"State: {r['state']}", "", "| question | truth | " + " | ".join(LABEL[m] for m in preds) + " |",
              "|---" * (len(preds) + 2) + "|"]
        for qid, q in qs.items():
            labs = labels_for(q)
            cell = lambda m: f"{labs[int(np.argmax(preds[m][i][qid]['p']))]} ({max(preds[m][i][qid]['p']):.2f})"  # noqa
            s.append(f"| {qid} | {r['gold'][qid]} | " + " | ".join(cell(m) for m in preds) + " |")
        s.append("")
    return "\n".join(s) + "\n"


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--config", required=True)
    a.add_argument("cmd", choices=["rows", "baselines", "laya", "report"])
    a.add_argument("--arms", nargs="+", default=["zero", "cal", "fit"])
    a = a.parse_args(argv)
    cfg = build.load_config(a.config)
    return {"rows": cmd_rows, "baselines": cmd_baselines, "laya": cmd_laya, "report": cmd_report}[a.cmd](cfg, a)


if __name__ == "__main__":
    main()
