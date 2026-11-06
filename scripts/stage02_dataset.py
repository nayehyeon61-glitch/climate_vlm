"""Stage 02, step 3: physics summaries, expert-selection labels and the Laya training files (preprocessing guide).

    python scripts/stage02_dataset.py --config configs/stage02_laya_router.yaml [--no-laya-check]

Per sample (issue time, domain, lead):
  features  from the router maps of stage02_forecasts (D_mu at the issue time, never the truth): temperature
            advection and moisture-flux convergence at 850 hPa (area-weighted mean, 10th / 90th percentile, area
            share above the split-B threshold, 2 x 2 sub-region means), 500 hPa height anomaly against the split-A
            monthly climatology (mean, sub-regions) and its gradient (mean, 90th percentile), 24 h changes of z500
            and T850 (mean, sub-regions), plus month, issue hour and lead. 850 hPa values use only cells where
            850 hPa is above the ground; a value with no such cell is null (never NaN in JSON).
            Values are rounded once to the precision of the text and the SAME rounded numbers feed the MLP router.
  label     the expert with the smallest E = mean over the 5 variables of the area-weighted domain MSE divided by
            the variable's split-A variance (temporal variance of the truth on the domain); near ties, shares and
            seasonal shares are recorded.

Files in <out>:
  router_train.jsonl (B), router_calib.jsonl (C), router_test.jsonl (D)   {state, questions, expected}, one per line
  physics_features.<parquet|csv>   sample keys, features, valid-area shares, missing-value causes
  expert_scores.<parquet|csv>      sample keys, E per expert and variable, label, margin, model versions
  split_manifest.json              periods, windows, gaps, sources of every statistic, schema versions, label and
                                   token statistics, leakage checks
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from climate_vlm import build, data  # noqa: E402
from scripts.stage02_common import EXPERTS, G, domain_label, sub_regions, wquantile  # noqa: E402

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA", 9: "SON",
          10: "SON", 11: "SON"}
JSONL = {"B": "router_train", "C": "router_calib", "D": "router_test"}
# (feature group, map, unit, decimals, statistics); "sub" = the 2 x 2 sub-region means in NW, NE, SW, SE order
GROUPS = [("T850_adv", "tadv850", "K/day", 1, ["mean", "p10", "p90", "strong", "sub"]),
          ("q850_conv", "qconv850", "g/kg/day", 2, ["mean", "p10", "p90", "strong", "sub"]),
          ("z500_anom", "z500_anom", "m", 0, ["mean", "sub"]),
          ("z500_grad", "z500_grad", "m/100km", 1, ["mean", "p90"]),
          ("z500_change_24h", "dz500_24h_m", "m", 0, ["mean", "sub"]),
          ("T850_change_24h", "dt850_24h", "K", 1, ["mean", "sub"])]
AT_850 = {"tadv850", "qconv850", "dt850_24h"}


def feature_names(subs):
    out = []
    for g, _, _, _, stats in GROUPS:
        for s in stats:
            out += [f"{g}_{n}" for n in subs] if s == "sub" else [f"{g}_{s}"]
    return out


def sha(path, n=12):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:n]


def hstr(h):
    return str(np.datetime64(int(h), "h"))


def summarise(R, rmaps, months, clim_z, w, subs, thr):
    """R (n, M, h, w) router maps -> (n, n_feat) rounded features, (n, 1 + n_sub) shares of valid 850 hPa area."""
    m = {k: R[:, rmaps.index(k)].astype(np.float64) for k in rmaps}
    valid = m["valid850"] > 0.5
    m["z500_anom"] = m["z500_m"] - clim_z[months - 1]
    for k in AT_850:
        m[k] = np.where(valid, m[k], np.nan)

    def wmean(x, ww):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nansum(x * ww, (-1, -2)) / np.where(np.isfinite(x), ww, 0).sum((-1, -2))

    feats = []
    for _, key, _, dec, stats in GROUPS:
        x = m[key]
        for s in stats:
            if s == "mean":
                v = [wmean(x, w)]
            elif s in ("p10", "p90"):
                v = [np.array([wquantile(xi, np.broadcast_to(w, xi.shape), int(s[1:]) / 100) for xi in x])]
            elif s == "strong":  # area share above the split-B threshold (|T adv|, q convergence)
                a = np.abs(x) if key == "tadv850" else x
                v = [wmean(np.where(np.isfinite(a), (a > thr[key]).astype(np.float64), np.nan), w)]
            else:  # sub-region means
                v = [wmean(x[:, rs, cs], w[rs, cs]) for _, rs, cs in subs]
            feats += [np.round(c, 2 if s == "strong" else dec) for c in v]
    vshare = np.stack([wmean(valid.astype(np.float64), w)]
                      + [wmean(valid[:, rs, cs].astype(np.float64), w[rs, cs]) for _, rs, cs in subs], 1)
    return np.stack(feats, 1), vshare


def fmt(v, dec):
    if not np.isfinite(v):
        return "null"
    return f"{v:.{dec}f}" if dec > 0 else f"{int(round(v))}"


def state_text(region, month, hour, lead, f, names, subs):
    d = dict(zip(names, f))
    parts = [f"region={region}", f"month={MONTHS[month - 1]}", f"issue={hour:02d}UTC", f"lead={lead}h",
             "sub-regions " + ",".join(n for n, _, _ in subs)]
    for g, _, unit, dec, stats in GROUPS:
        items = []
        for s in stats:
            if s == "sub":
                items.append("sub=" + ",".join(fmt(d[f"{g}_{n}"], dec) for n, _, _ in subs))
            else:
                items.append(f"{s}=" + fmt(d[f"{g}_{s}"], 2 if s == "strong" else dec))
        parts.append(f"{g} {unit}: " + " ".join(items))
    return "; ".join(parts)


def table_out(rows, path):
    """Write a list of dicts as parquet when pyarrow is there, else CSV; returns the path written."""
    import pandas as pd
    df = pd.DataFrame(rows)
    try:
        import pyarrow  # noqa: F401
        df.to_parquet(path + ".parquet", index=False)
        return path + ".parquet"
    except ImportError:
        df.to_csv(path + ".csv", index=False)
        return path + ".csv"


def tokenizer_check(lc, files, max_show=3):
    """Token counts of state + question + options with laya's tokenizer, and laya's own item build (skips)."""
    from laya import train as lt
    from transformers import AutoTokenizer
    d = lt.resolve_checkpoint_dir(lc["base"])
    tok = AutoTokenizer.from_pretrained(os.path.join(d, "tokenizer"))
    with open(os.path.join(d, "rl_agent_config.json")) as fh:
        c = json.load(fh)
    max_len, head_max_len = c.get("max_len", 512), c.get("head_max_len", 192)
    q = {k: lc["criteria"][k] for k in EXPERTS}
    opt_ids = {k: tuple(tok(f"{k}: {v}", add_special_tokens=False)["input_ids"]) for k, v in q.items()}
    rep = {"checkpoint": d, "max_len": max_len, "head_max_len": head_max_len,
           "option_tokens": {k: len(v) for k, v in opt_ids.items()},
           "options_distinct": len(set(opt_ids.values())) == len(opt_ids),
           "question_tokens": len(tok(lc["instructions"], add_special_tokens=False)["input_ids"]), "splits": {}}
    for split, path in files.items():
        rows = lt.read_jsonl(path)
        n = np.array([len(tok(r["state"], add_special_tokens=False)["input_ids"]) for r in rows])
        items, skipped = lt.items_from_rows(tok, rows, max_len, head_max_len)
        seq = [len(lt.encode_item(tok, it, max_len, head_max_len)["input_ids"]) for it in items[:50]]
        rep["splits"][split] = {"rows": len(rows), "items": len(items), "skipped": skipped,
                                "state_tokens": {"min": int(n.min()), "mean": float(n.mean()), "max": int(n.max())},
                                "state_truncated": int((n > max_len).sum()),
                                "full_input_tokens_first50": {"min": int(min(seq)), "max": int(max(seq))}}
    rep["examples"] = [r["state"] for r in lt.read_jsonl(files["B"])[:max_show]]
    return rep


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--config", required=True)
    a.add_argument("--no-laya-check", action="store_true", help="skip the tokenizer check (tests)")
    a = a.parse_args(argv)
    cfg = build.load_config(a.config)
    bc = build.load_config(cfg["base"]["config"])
    p = build.paths(bc)
    meta = data.load_meta(p["cache"])
    out = cfg["out"]
    by_split = {v["split"]: k for k, v in cfg["periods"].items()}
    if sorted(by_split) != ["A", "B", "C", "D"]:
        raise ValueError("periods must carry the splits A, B, C and D once each")
    dc, fc = cfg["domain"], cfg.get("features", {})
    strong_q, near = float(fc.get("strong_quantile", 0.9)), float(fc.get("near_tie", 0.05))
    region = domain_label(dc)
    lc = cfg["router"]["laya"]
    if set(lc["criteria"]) != set(EXPERTS):
        raise ValueError(f"router.laya.criteria must describe exactly the experts {EXPERTS}")
    question = {"expert": {"type": "choice", "instructions": lc["instructions"],
                           "criteria": {k: lc["criteria"][k] for k in EXPERTS}}}  # options in the experts' order

    fcs = {s: np.load(os.path.join(out, f"fc_{by_split[s]}.npz")) for s in "ABCD"}
    rmaps = [str(x) for x in fcs["B"]["rmaps"]]
    rows, cols = fcs["B"]["rows"], fcs["B"]["cols"]
    lat_hr = np.asarray(meta["lat"])[rows]
    w = np.cos(np.deg2rad(lat_hr))[:, None] * np.ones((1, len(cols)))
    subs = sub_regions(len(rows), len(cols), dc.get("sub", [2, 2]))
    sub_names = [n for n, _, _ in subs]
    names = feature_names(sub_names)

    # ---- statistics of the training splits only: climatology (A), label scale (A), "strong" thresholds (B)
    clim = np.load(p["clim"], mmap_mode="r")  # (12, C, H, W) monthly means of the stage-01 training years = split A
    clim_z = np.asarray(clim[:, meta["channels"].index("z500")], np.float64)[:, rows[:, None], cols[None, :]] / G
    r5, c5 = fcs["A"]["rows5"], fcs["A"]["cols5"]
    lat5 = np.load(os.path.join(p["cache"], "truth_5p625", "lat.npy"))[r5]
    w5 = np.cos(np.deg2rad(lat5))[:, None] * np.ones((1, len(c5)))
    OA = fcs["A"]["O"][..., r5[:, None], c5[None, :]].astype(np.float64)  # (n, L, 5, h, w)
    var_v = ((OA.var(0) * w5).sum((-1, -2)) / w5.sum()).mean(0)  # (5,) temporal variance, area mean, over leads
    sig_v = np.sqrt(var_v)
    RB = fcs["B"]["R"]
    vb = RB[:, rmaps.index("valid850")] > 0.5
    wb = np.broadcast_to(w, vb.shape)
    thr = {"tadv850": wquantile(np.abs(RB[:, rmaps.index("tadv850")])[vb], wb[vb], strong_q),
           "qconv850": wquantile(RB[:, rmaps.index("qconv850")][vb], wb[vb], strong_q)}
    print(f"split-A sigma per variable {dict(zip([str(v) for v in fcs['A']['vars']], sig_v.round(3)))}; "
          f"strong thresholds from split B (q={strong_q}): |T adv| > {thr['tadv850']:.2f} K/day, "
          f"q conv > {thr['qconv850']:.3f} g/kg/day", flush=True)

    base_sha, exp_sha = sha(cfg["base"]["ckpt"]), sha(os.path.join(out, "experts.pt"))
    feat_rows, score_rows, manifest_periods, labels = [], [], {}, {}
    files = {}
    leakage = {"label_names_in_state": 0}
    for s in "BCD":
        z, fx = fcs[s], np.load(os.path.join(out, f"fx_{by_split[s]}.npz"))
        if not np.array_equal(z["issue"], fx["issue"]):
            raise ValueError(f"split {s}: forecasts and experts disagree on the issue times")
        issue, leads = z["issue"].astype(np.int64), [int(h) for h in z["leads"]]
        months = np.array([int(hstr(h)[5:7]) for h in issue])
        hours = np.array([int(hstr(h)[11:13]) for h in issue])
        F, vshare = summarise(z["R"], rmaps, months, clim_z, w, subs, thr)
        # labels: (n, 3, L, 5) area-weighted domain MSE / split-A variance
        e = (((fx["E"] - fx["O"][:, None]) ** 2) * w5).sum((-1, -2)) / w5.sum() / var_v
        E = e.mean(-1).transpose(0, 2, 1)  # (n, L, 3)
        lab = E.argmin(-1)
        srt = np.sort(E, -1)
        margin = (srt[..., 1] - srt[..., 0]) / np.maximum(srt[..., 0], 1e-12)
        path = os.path.join(out, f"{JSONL[s]}.jsonl")
        files[s] = path
        n_lab = np.zeros(3, int)
        season_lab = {}
        with open(path, "w") as fh:
            for i in range(len(issue)):
                miss = [nm for nm, v in zip(names, F[i]) if not np.isfinite(v)]
                for j, h in enumerate(leads):
                    sid = f"{s}-{hstr(issue[i]).replace('-', '').replace('T', '')}-{dc.get('key', 'R0')}-L{h:03d}"
                    st = state_text(region, months[i], hours[i], h, F[i], names, subs)
                    leakage["label_names_in_state"] += sum(k in st.lower() for k in EXPERTS)
                    fh.write(json.dumps({"state": st, "questions": question,
                                         "expected": {"expert": EXPERTS[lab[i, j]]}}, separators=(",", ":")) + "\n")
                    key = {"sample_id": sid, "split": s, "period": by_split[s], "issue_time": hstr(issue[i]),
                           "year": int(z["years"][i]), "t_index": int(z["ics"][i]), "region": region,
                           "lead_h": h, "month": int(months[i]), "issue_hour": int(hours[i])}
                    feat_rows.append({**key, **dict(zip(names, F[i].tolist())),
                                      "valid850_share": float(vshare[i, 0]),
                                      **{f"valid850_share_{n}": float(vshare[i, k + 1]) for k, n in enumerate(sub_names)},
                                      "missing": ";".join(miss),
                                      "missing_cause": "850 hPa below the ground in every cell" if miss else ""})
                    score_rows.append({**key, **{f"E_{n}": float(E[i, j, k]) for k, n in enumerate(EXPERTS)},
                                       **{f"E_{n}_{v}": float(e[i, k, j, q]) for k, n in enumerate(EXPERTS)
                                          for q, v in enumerate([str(x) for x in z["vars"]])},
                                       "label": EXPERTS[lab[i, j]], "margin": float(margin[i, j]),
                                       "near_tie": bool(margin[i, j] < near), "base_ckpt_sha": base_sha,
                                       "experts_sha": exp_sha,
                                       "truth": f"{p['cache']}/truth_5p625 (WB2 64x32), labels and scores only"})
                    n_lab[lab[i, j]] += 1
                    se = SEASON[int(months[i])]
                    season_lab.setdefault(se, np.zeros(3, int))[lab[i, j]] += 1
        labels[s] = {"n": int(n_lab.sum()), "share": dict(zip(EXPERTS, (n_lab / n_lab.sum()).round(3).tolist())),
                     "near_tie_share": float(np.mean(margin < near)),
                     "by_season": {k: dict(zip(EXPERTS, (v / v.sum()).round(3).tolist())) for k, v in
                                   sorted(season_lab.items())},
                     "mean_E": dict(zip(EXPERTS, E.mean((0, 1)).round(4).tolist()))}
        spec = cfg["periods"][by_split[s]]
        lo, hi = min(bc["data"]["history_h"]), max(leads)
        manifest_periods[s] = {"period": by_split[s], **spec, "samples": int(n_lab.sum()), "issue_times": len(issue),
                               "first_issue": hstr(issue.min()), "last_issue": hstr(issue.max()),
                               "window_start": hstr(issue.min() + lo), "window_end": hstr(issue.max() + hi),
                               "valid850_share_mean": float(np.nanmean(vshare[:, 0])),
                               "samples_with_null": int(np.sum(~np.isfinite(F).all(1)))}
        print(f"{s} ({by_split[s]}): {n_lab.sum()} samples, labels {labels[s]['share']}, "
              f"near ties {labels[s]['near_tie_share']:.2f}, mean E {labels[s]['mean_E']}", flush=True)
    # split A: experts only, no routing samples
    zA = fcs["A"]
    lo, hi = min(bc["data"]["history_h"]), max(int(h) for h in zA["leads"])
    manifest_periods["A"] = {"period": by_split["A"], **cfg["periods"][by_split["A"]], "issue_times": len(zA["issue"]),
                             "window_start": hstr(zA["issue"].min() + lo), "window_end": hstr(zA["issue"].max() + hi),
                             "use": "experts (and the stage-01 predictor) only"}
    order = sorted("ABCD", key=lambda k: manifest_periods[k]["window_start"])
    gaps = {f"{x}->{y}": int((np.datetime64(manifest_periods[y]["window_start"]) -
                              np.datetime64(manifest_periods[x]["window_end"])) / np.timedelta64(1, "h"))
            for x, y in zip(order[:-1], order[1:])}
    leakage["window_overlap"] = any(v <= 0 for v in gaps.values())
    if leakage["label_names_in_state"] or leakage["window_overlap"]:
        raise RuntimeError(f"leakage check failed: {leakage}, gaps {gaps}")

    feat_path = table_out(feat_rows, os.path.join(out, "physics_features"))
    score_path = table_out(score_rows, os.path.join(out, "expert_scores"))
    manifest = {
        "schema": cfg.get("schema", {}), "created": time.strftime("%Y-%m-%d %H:%M:%S"), "cpu_count": os.cpu_count(),
        "base": {**cfg["base"], "ckpt_sha256_12": base_sha}, "experts_sha256_12": exp_sha,
        "domain": {**dc, "label": region, "rows_2p8125": rows.tolist(), "cols_2p8125": cols.tolist(),
                   "rows_5p625": r5.tolist(), "cols_5p625": c5.tolist(), "sub_regions": sub_names},
        "periods": manifest_periods, "gaps_h": gaps,
        "features": {"names": names, "source": "D_mu (stage-01 downscaler) of the 5.625 deg input at the issue time",
                     "z500_climatology": f"{p['clim']}: monthly means of the training years "
                                         f"{bc['data']['train_years']} (split A)",
                     "strong_thresholds": {k: float(v) for k, v in thr.items()}, "strong_quantile": strong_q,
                     "strong_threshold_source": "split B, area-weighted over valid domain cells",
                     "valid_mask": "850 hPa used where Phi_850 > surface geopotential",
                     "units": {g: u for g, _, u, _, _ in GROUPS}, "decimals": {g: d for g, _, _, d, _ in GROUPS}},
        "labels": {"metric": "mean over z500, t850, t2m, u10, v10 of area-weighted domain MSE / split-A variance",
                   "sigma_split_A": dict(zip([str(v) for v in fcs["A"]["vars"]], sig_v.tolist())),
                   "near_tie_rel_margin": near, "type": "hard", **labels},
        "question": question, "files": {"jsonl": files, "physics_features": feat_path, "expert_scores": score_path},
        "leakage_checks": {**leakage, "state_sources": "issue-time D_mu maps, split-A climatology, split-B thresholds, "
                                                       "calendar; no truth, no expert errors, no label names"},
    }
    if not a.no_laya_check:
        manifest["tokenizer"] = tokenizer_check(lc, files)
        tk = manifest["tokenizer"]
        print("tokens: " + json.dumps({k: v["state_tokens"] for k, v in tk["splits"].items()})
              + f", options distinct {tk['options_distinct']}, skipped "
              + json.dumps({k: v["skipped"] for k, v in tk["splits"].items()}), flush=True)
        for ex in tk["examples"]:
            print("example state:", ex)
    with open(os.path.join(out, "split_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"wrote {', '.join(files.values())}, {feat_path}, {score_path}, {out}/split_manifest.json", flush=True)
    return manifest


if __name__ == "__main__":
    main()
