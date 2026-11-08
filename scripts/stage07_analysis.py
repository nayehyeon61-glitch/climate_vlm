"""Stage 07, step 5: paired differences between the arms, and the structure of the members.

    python scripts/stage07_analysis.py --config configs/stage07_flow_ensemble.yaml

Paired differences over the 2020 ICs (95 % IC bootstrap, as compare_climode.py): RMSE of the ensemble mean and CRPS
(the fair ensemble CRPS of the members, the closed form for the Gaussian head; ClimODE's min-max units), for every arm
against the Gaussian head and for the pairs that isolate one input: the VLM memory (fm_vla_x0 - fm_mu_x0), the
issue-time state (fm_vla_x0 - fm_vla), the VLM mean (fm_mu_x0 - fm_x0) and the whole VLM (fm_vla_x0 - fm_x0).

Structure: a member's deviation from the ensemble mean should look like the error of the ensemble mean,
- across scales: spread and error per spherical-harmonic band (l <= 6, 7..15, > 15; (M + 1) / M corrected spread),
- across leads: correlation of the deviation (error) at consecutive leads,
- in aggregates, where it matters for the score: CRPS and spread / RMSE of 8 x 8-cell box means (2 x 2 tokens of the
  expert) and of the mean over the leads. Members that are independent between tokens or leads are too narrow there.
Members drawn from the Gaussian head independently per grid point and lead are the reference: their noise sits in the
smallest scales and is uncorrelated between leads. Writes <docs>/analysis.json and analysis.md.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from climate_vlm import build  # noqa: E402
from climate_vlm.spectral import SphericalBands  # noqa: E402
from scripts.compare_climode import REF_VARS, boot_ci, crps_gaussian  # noqa: E402
from scripts.stage07_flow import crps_fair, load_cache  # noqa: E402

PAIRS = [("fm_vla_x0", "fm_mu_x0", "VLM memory"), ("fm_vla_x0", "fm_vla", "issue-time state"),
         ("fm_mu_x0", "fm_x0", "VLM mean"), ("fm_vla_x0", "fm_x0", "whole VLM"),
         ("fm_mu_x0_sn", "fm_mu_x0", "structured source noise"),
         ("fm_vla_x0_sn", "fm_vla_x0", "structured source noise"),
         ("fm_vla_x0_sn", "fm_mu_x0_sn", "VLM memory, structured noise"),
         ("fm_mu_x0_sn", "fm_x0_sn", "VLM mean, structured noise"),
         ("fm_vla_x0_sn", "fm_x0_sn", "whole VLM, structured noise")]
BOX = 8  # cells per side of the box means (2 x 2 tokens of 4 x 4 cells)


def analyse(cfg, edges=(6, 15), seed=0):
    te, info = load_cache(cfg, "test")
    ref = json.load(open(cfg["reference"])) if cfg.get("reference") and os.path.exists(cfg["reference"]) else None
    ics = [int(i) for i in info["ics"]]
    rows = [ics.index(int(i)) for i in ref["ics"] if int(i) in ics] if ref else list(range(len(ics)))
    bc = build.load_config(cfg["base"]["config"])
    tdir = os.path.join(build.paths(bc)["cache"], "truth_5p625")
    lat = np.load(os.path.join(tdir, "lat.npy"))
    mm = json.load(open(os.path.join(tdir, "minmax_2006_2018.json")))
    mx, mn = np.array(mm["max"])[:, None, None], np.array(mm["min"])[:, None, None]
    m5, s5 = np.array(info["mean"])[:, None, None], np.array(info["std"])[:, None, None]
    phys = lambda x: np.asarray(x, np.float64) * s5 + m5  # noqa: E731
    leads, M = [int(h) for h in info["leads"]], int(cfg["flow"]["members"])
    H, W = te["y"].shape[-2:]
    edges = [e for e in edges if e < H - 1]
    sb = SphericalBands(lat, np.arange(W) * 360.0 / W, edges)
    w = np.cos(np.deg2rad(lat))
    w = (w / w.mean())[:, None]
    wt = torch.tensor(w)

    arms = {a: os.path.join(cfg["out"], a, "members_2020.npz") for a in cfg["arms"]}
    arms = {a: np.load(f)["E_minus_x0"] for a, f in arms.items() if os.path.exists(f)}  # read once (f16)
    names = ["gaussian_head"] + list(arms)
    gen = np.random.default_rng(seed)
    n, L, K = len(rows), len(leads), len(edges) + 1
    per = {k: {"rmse": np.zeros((n, L, 5)), "crps": np.zeros((n, L, 5)), "crps_box": np.zeros((n, L, 5)),
               "crps_lead": np.zeros((n, 1, 5))} for k in names}
    agg = {k: {a: np.zeros((2, L, 5)) for a in ("box", "lead")} for k in names}  # sums of spread^2, error^2
    box = BOX if H % BOX == 0 and W % BOX == 0 else 1
    wr = w[:, 0].reshape(H // box, box)
    wb = (wr.mean(1) / wr.mean())[:, None]  # area weight of a box row

    def boxmean(x):  # (..., H, W) -> (..., H / box, W / box), area-weighted
        xr = x.reshape(x.shape[:-2] + (H // box, box, W // box, box))
        return (xr * wr[:, :, None, None]).sum((-3, -1)) / (wr.sum(1)[:, None] * box)

    def agg_sums(E, O, wt_):  # members (M, ..., h, w), truth (..., h, w) -> (2, ...) weighted mean spread^2, error^2
        sp = E.var(0, ddof=1) * (M + 1) / M
        return np.stack([(sp * wt_).mean((-1, -2)), ((E.mean(0) - O) ** 2 * wt_).mean((-1, -2))])
    band = {k: {"spread": np.zeros((L, 5, K)), "error": np.zeros((L, 5, K))} for k in names}
    lc = {k: {"dev": np.zeros((3, L - 1, 5)), "err": np.zeros((3, L - 1, 5))} for k in names}  # sums ab, aa, bb

    def bands_ms(x):  # (..., h, w) -> (..., K) area-weighted mean square per band
        return torch.stack([(b ** 2 * wt).mean((-1, -2)) for b in sb(torch.tensor(x))], -1).numpy()

    def lead_sums(x):  # (..., L, 5, h, w) -> (3, L - 1, 5) weighted sums over everything but lead and variable
        a, b = x[..., :-1, :, :, :], x[..., 1:, :, :, :]
        red = tuple(range(x.ndim - 4)) + (-2, -1)
        return np.stack([(w * a * b).sum(red), (w * a * a).sum(red), (w * b * b).sum(red)])

    for e, r in enumerate(rows):
        O, X0 = phys(te["y"][r]), phys(te["x0"][r])
        mu, sig = phys(te["mu"][r]), np.asarray(te["sigma"][r], np.float64) * s5
        for k in names:
            if k == "gaussian_head":
                E = mu[None] + sig[None] * gen.standard_normal((M,) + mu.shape)
                per[k]["crps"][e] = crps_gaussian((O - mn) / (mx - mn), (mu - mn) / (mx - mn), sig / (mx - mn)
                                                  ).mean((-1, -2))
                P = mu
            else:
                E = np.asarray(arms[k][r], np.float64) + X0[None, None]  # (M, L, 5, h, w)
                per[k]["crps"][e] = crps_fair((E - mn) / (mx - mn), (O - mn) / (mx - mn)).mean((-1, -2))
                P = E.mean(0)
            per[k]["rmse"][e] = np.sqrt(((P - O) ** 2 * w).mean((-1, -2)))
            Eb, Ob = boxmean(E), boxmean(O)
            per[k]["crps_box"][e] = crps_fair((Eb - mn) / (mx - mn), (Ob - mn) / (mx - mn)).mean((-1, -2))
            agg[k]["box"] += agg_sums(Eb, Ob, wb)
            El, Ol = E.mean(1), O.mean(0)  # mean over the leads
            per[k]["crps_lead"][e, 0] = crps_fair((El - mn) / (mx - mn), (Ol - mn) / (mx - mn)).mean((-1, -2))
            agg[k]["lead"][:, 0] += agg_sums(El, Ol, w)
            dev, err = E - E.mean(0), E.mean(0) - O
            band[k]["spread"] += bands_ms(dev).sum(0) / (M - 1) * (M + 1) / M
            band[k]["error"] += bands_ms(err)
            lc[k]["dev"] += lead_sums(dev)
            lc[k]["err"] += lead_sums(err)

    hs = [str(h) for h in leads]
    res = {"n_ic": n, "leads": leads, "vars": REF_VARS, "members": M, "band_edges": edges, "box": box, "pairs": {},
           "bands": {}, "lead_corr": {}, "aggregates": {}}
    cmp = [(a, "gaussian_head", "flow matching vs Gaussian head") for a in arms] + \
          [p for p in PAIRS if p[0] in arms and p[1] in arms]
    for a, b, what in cmp:
        d = res["pairs"].setdefault(f"{a} - {b}", {"what": what})
        for met in ("rmse", "crps", "crps_box", "crps_lead"):
            x = per[a][met] - per[b][met]
            for j, h in enumerate(hs if met != "crps_lead" else ["lead_mean"]):
                d.setdefault(h, {})[met] = {v: {"mean": float(x[:, j, i].mean()), "ci95": boot_ci(x[:, j, i])}
                                            for i, v in enumerate(REF_VARS)}
    for k in names:
        res["bands"][k] = {h: {v: {"spread": np.sqrt(band[k]["spread"][j, i] / n).tolist(),
                                   "error": np.sqrt(band[k]["error"][j, i] / n).tolist()}
                               for i, v in enumerate(REF_VARS)} for j, h in enumerate(hs)}
        res["aggregates"][k] = {
            "box": {h: {v: {"crps": float(per[k]["crps_box"][:, j, i].mean()),
                            "spread_skill": float(np.sqrt(agg[k]["box"][0, j, i] / agg[k]["box"][1, j, i]))}
                        for i, v in enumerate(REF_VARS)} for j, h in enumerate(hs)},
            "lead_mean": {v: {"crps": float(per[k]["crps_lead"][:, 0, i].mean()),
                              "spread_skill": float(np.sqrt(agg[k]["lead"][0, 0, i] / agg[k]["lead"][1, 0, i]))}
                          for i, v in enumerate(REF_VARS)}}
        corr = {s: lc[k][s][0] / np.sqrt(lc[k][s][1] * lc[k][s][2]) for s in ("dev", "err")}
        res["lead_corr"][k] = {v: {"members": corr["dev"][:, i].tolist(), "error": corr["err"][:, i].tolist()}
                               for i, v in enumerate(REF_VARS)}
    return res


def table(res, show=("24", "72", "144")):
    v_unit = {"z": "z500", "t": "t850", "t2m": "t2m", "u10": "u10", "v10": "v10"}
    hs = [str(h) for h in res["leads"]]
    show = [h for h in show if h in hs] or hs[:3]
    hb = "72" if "72" in hs else hs[len(hs) // 2]  # lead of the band table
    e = res["band_edges"]
    names_b = [f"l<={e[0]}"] + [f"{a + 1}<=l<={b}" for a, b in zip(e[:-1], e[1:])] + [f"l>{e[-1]}"]
    out = [f"# Stage 07 analysis, 2020 ({res['n_ic']} ICs, {res['members']} members)", ""]
    for met, title in (("crps", "CRPS (min-max units, fair ensemble CRPS; closed form for the Gaussian head)"),
                       ("rmse", "RMSE of the ensemble mean")):
        out += [f"## Paired differences: {title}", "",
                "Mean [95 % IC bootstrap] at " + " / ".join(f"{h} h" for h in show) + "; negative = first is better.",
                "", "| pair | " + " | ".join(v_unit[v] for v in res["vars"]) + " |",
                "|---|" + "---|" * len(res["vars"])]
        sc = 1e4 if met == "crps" else 1.0
        for p, d in res["pairs"].items():
            cells = []
            for v in res["vars"]:
                f = "{:+.1f}" if (met == "crps" or v == "z") else "{:+.2f}"
                f = f + " [" + f + ", " + f + "]"
                cells.append("<br>".join(f.format(sc * d[h][met][v]["mean"], *(sc * c for c in d[h][met][v]["ci95"]))
                                         for h in show))
            out.append(f"| {p} ({d['what']}) | " + " | ".join(cells) + " |")
        if met == "crps":
            out += ["", "CRPS differences are multiplied by 10^4."]
        out.append("")
    out += ["## Spread / error per spherical-harmonic band of the deviation (" + ", ".join(names_b) + f"), {hb} h", "",
            "Spread: rms deviation of the members from their mean ((M + 1) / M corrected); error: rms error of the "
            "ensemble mean. A calibrated ensemble has spread = error in every band.", "",
            "| system | " + " | ".join(f"{v_unit[v]} spread / error" for v in ("z", "t")) + " |", "|---|---|---|"]
    for k, d in res["bands"].items():
        cells = []
        for v in ("z", "t"):
            f = "{:.0f}" if v == "z" else "{:.2f}"
            cells.append(", ".join((f + " / " + f).format(s, x) for s, x in zip(d[hb][v]["spread"], d[hb][v]["error"])))
        out.append(f"| {k} | " + " | ".join(cells) + " |")
    bx = res.get("box", 1)
    out += ["", f"## Aggregates: {bx} x {bx}-cell box means at {hb} h, and the mean over all leads", "",
            "CRPS x 10^4 (fair, min-max units) / spread / RMSE ((M + 1) / M corrected).", "",
            "| system | " + " | ".join(f"{v_unit[v]} box" for v in res["vars"]) + " | "
            + " | ".join(f"{v_unit[v]} lead mean" for v in res["vars"]) + " |",
            "|---|" + "---|" * (2 * len(res["vars"]))]
    for k, d in res["aggregates"].items():
        cells = [f"{1e4 * d['box'][hb][v]['crps']:.1f} / {d['box'][hb][v]['spread_skill']:.2f}" for v in res["vars"]]
        cells += [f"{1e4 * d['lead_mean'][v]['crps']:.1f} / {d['lead_mean'][v]['spread_skill']:.2f}"
                  for v in res["vars"]]
        out.append(f"| {k} | " + " | ".join(cells) + " |")
    out += ["", f"Paired differences of these CRPS (x 10^4, box means at {hb} h / lead mean):", "",
            "| pair | " + " | ".join(v_unit[v] for v in res["vars"]) + " |", "|---|" + "---|" * len(res["vars"])]
    for p, d in res["pairs"].items():
        cells = []
        for v in res["vars"]:
            q = [d[hb]["crps_box"][v], d["lead_mean"]["crps_lead"][v]]
            cells.append("<br>".join("{:+.1f} [{:+.1f}, {:+.1f}]".format(1e4 * x["mean"], *(1e4 * c for c in x["ci95"]))
                                     for x in q))
        out.append(f"| {p} ({d['what']}) | " + " | ".join(cells) + " |")
    out += ["", "## Correlation between consecutive leads (" + ", ".join(f"{a}-{b}" for a, b in zip(hs[:-1], hs[1:]))
            + " h)", "",
            "Members: deviation from the ensemble mean; error: ensemble mean minus truth.", "",
            "| system | z500 members | z500 error | t850 members | t850 error |", "|---|---|---|---|---|"]
    for k, d in res["lead_corr"].items():
        cells = [" ".join(f"{c:.2f}" for c in d[v][s]) for v in ("z", "t") for s in ("members", "error")]
        out.append(f"| {k} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--config", required=True)
    a.add_argument("--bands", default="6,15", help="band edges in total wavenumber l")
    a = a.parse_args(argv)
    cfg = build.load_config(a.config)
    res = analyse(cfg, [int(e) for e in a.bands.split(",")])
    os.makedirs(cfg["docs"], exist_ok=True)
    json.dump(res, open(os.path.join(cfg["docs"], "analysis.json"), "w"), indent=1)
    with open(os.path.join(cfg["docs"], "analysis.md"), "w") as fh:
        fh.write(table(res))
    print(table(res))
    return res


if __name__ == "__main__":
    main()
