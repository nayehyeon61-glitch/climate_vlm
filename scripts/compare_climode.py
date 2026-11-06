"""Score evaluate.py forecasts like pg-flow's ClimODE comparison (WeatherBench2 5.625 deg, leads 24..144 h).

Same ICs, truth and score definitions as physical_guide_flowmatching scripts/compare_step13_climode.py, whose 2020
result (docs/results_2026_11/scores/raw/blend_ngcm_2020_climode_paired.json) holds the ClimODE public checkpoint and
persistence on the 179 ClimODE-convention ICs of 2020:

  grid   forecasts (2.8125 deg) are averaged to WB2's 64x32 grid (data.coarsen; the Gaussian scale is averaged too,
         i.e. errors assumed fully correlated inside a 5.625 deg cell)
  truth  WB2 ERA5 64x32 conservative (scripts/wb2_to_cache.py truth_5p625/<year>.npy), as for ClimODE
  rmse   sqrt(mean_hw(alpha (P - O)^2)), alpha = cos(lat) / mean cos(lat)
  acc    anomalies against the test year's time mean, field mean removed, alpha-weighted correlation
  crps   Gaussian closed form in ClimODE's min-max units (2006-2018 min / max), unweighted spatial mean
  spread sqrt(alpha-weighted mean of sigma^2)

Persistence is recomputed here and checked against the reference file, which validates truth, ICs and scores.

    python scripts/compare_climode.py --config configs/stage01_cpu_small.yaml --eval runs/stage01/eval_stage3 \
        --reference ../physical_guide_flowmatching/docs/results_2026_11/scores/raw/blend_ngcm_2020_climode_paired.json
    # paired intervals against ClimODE need its per-IC forecasts (pg-flow scripts/run_climode_2020.py output):
    #   --climode-pred runs/blend_ngcm/ngcm2020/climode2020/climode_pred_2020_valid.npz
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from climate_vlm import build, data  # noqa: E402
from climate_vlm.contract import CLIMODE_VARS  # noqa: E402

REF_VARS = ["z", "t", "t2m", "u10", "v10"]  # pg-flow's names of CLIMODE_VARS
UNITS = {"z": "m2/s2", "t": "K", "t2m": "K", "u10": "m/s", "v10": "m/s"}


def crps_gaussian(obs, mu, sig):
    """properscoring.crps_gaussian, elementwise."""
    z = (obs - mu) / sig
    erf = torch.erf(torch.from_numpy(np.asarray(z / math.sqrt(2), dtype=np.float64))).numpy()
    pdf = np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    return sig * (z * erf + 2 * pdf - 1 / math.sqrt(math.pi))


def climode_scores(P, O, S, clim, lat, mx, mn):
    """(..., 5, H, W) physical -> dict of (..., 5): pg-flow run_step11_climode.climode_scores."""
    w = np.cos(np.deg2rad(lat))
    w = (w / w.mean())[:, None]
    rmse = np.sqrt(((P - O) ** 2 * w).mean((-1, -2)))
    pa, oa = P - clim, O - clim
    pa = pa - pa.mean((-1, -2), keepdims=True)
    oa = oa - oa.mean((-1, -2), keepdims=True)
    acc = (w * pa * oa).sum((-1, -2)) / np.sqrt((w * pa ** 2).sum((-1, -2)) * (w * oa ** 2).sum((-1, -2)))
    sc = (mx - mn)[:, None, None]
    crps = crps_gaussian((O - mn[:, None, None]) / sc, (P - mn[:, None, None]) / sc, S / sc).mean((-1, -2))
    return {"rmse": rmse, "acc": acc, "crps_gauss": crps, "spread": np.sqrt((S ** 2 * w).mean((-1, -2)))}


def boot_ci(d, n=2000, seed=0):
    """pg-flow scripts/report_step11.boot_ci: 95 % interval of the mean over ICs."""
    rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def load_forecast(path, lon0, n_lat):
    """<tag>_all_<year>.npz -> ics, leads, P, S (n, L, 5, h, w) on the truth grid (n_lat rows), HR truth coarsened."""
    z = np.load(path)
    ch = [str(c) for c in z["channels"]]
    k = [ch.index(v) for v in CLIMODE_VARS]
    f = z["mean"].shape[-2] // n_lat
    P = data.coarsen(z["mean"][:, :, k], f, lon0)
    S = data.coarsen(z["scale"][:, :, k], f, lon0) if "scale" in z.files else np.full_like(P, 1e-6)
    O_hr = data.coarsen(z["truth"][:, :, k], f, lon0)
    return [int(i) for i in z["ics"]], [int(h) for h in z["leads"]], P, S, O_hr


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--config", required=True)
    a.add_argument("--eval", required=True, help="evaluate.py output folder")
    a.add_argument("--tag", default="climate_vlm")
    a.add_argument("--year", type=int, default=2020)
    a.add_argument("--reference", help="pg-flow climode_paired JSON of the same year (ClimODE, persistence, ...)")
    a.add_argument("--climode-pred", help="pg-flow ClimODE per-IC forecasts (npz with mean, std, ics, lead_hours)")
    a.add_argument("--out", help="default: <eval>/climode_compare")
    a = a.parse_args(argv)
    cfg = build.load_config(a.config)
    cache = build.paths(cfg)["cache"]
    meta = data.load_meta(cache)
    lon0 = data.lon0_grid(meta["lon"])
    tdir = os.path.join(cache, "truth_5p625")
    truth = np.load(os.path.join(tdir, f"{a.year}.npy"), mmap_mode="r")
    lat = np.load(os.path.join(tdir, "lat.npy"))
    with open(os.path.join(tdir, "minmax_2006_2018.json")) as fh:
        mm = json.load(fh)
    mx, mn = np.array(mm["max"]), np.array(mm["min"])
    clim = np.asarray(truth, np.float64).mean(0)

    ics, leads, P, S, O_hr = load_forecast(os.path.join(a.eval, f"{a.tag}_all_{a.year}.npz"), lon0, len(lat))
    ref = json.load(open(a.reference)) if a.reference else None
    use = ics
    if ref is not None:
        missing = sorted(set(ref["ics"]) - set(ics))
        if missing:
            raise ValueError(f"reference ICs {missing} are not in the forecast file")
        use = [int(i) for i in ref["ics"]]
        bad = [h for h in leads if h not in ref["leads"]]
        if bad:
            raise ValueError(f"forecast leads {bad} are not in the reference")  # a subset of its leads is fine
    rows = [ics.index(i) for i in use]
    cp = None
    if a.climode_pred:
        cp = dict(np.load(a.climode_pred))
        yi = [int(y) for y in cp["years"]].index(a.year) if "years" in cp else 0
        cpl = [int(h) for h in cp["lead_hours"]]
        cpi = [int(i) for i in cp["ics"]]
        use = [i for i in use if i in cpi]
        rows = [ics.index(i) for i in use]

    per = {}  # name -> metric -> (n_ic, L, 5)
    grid_err = np.zeros((len(leads), 5))
    for e, (r, i0) in enumerate(zip(rows, use)):
        for j, h in enumerate(leads):
            O = np.asarray(truth[i0 + h // 6], np.float64)
            grid_err[j] += ((O_hr[r, j] - O) ** 2 * (np.cos(np.deg2rad(lat)) / np.cos(np.deg2rad(lat)).mean())[:, None]
                            ).mean((-1, -2))
            items = {"ours": (P[r, j], S[r, j]),
                     "persistence": (np.asarray(truth[i0], np.float64), np.full_like(O, 1e-6))}
            if cp is not None:
                c = cpi.index(i0)
                items["climode"] = (cp["mean"][yi, c, cpl.index(h)], cp["std"][yi, c, cpl.index(h)])
            for name, (p, s) in items.items():
                for met, v in climode_scores(np.asarray(p, np.float64), O, np.asarray(s, np.float64), clim, lat,
                                             mx, mn).items():
                    per.setdefault(name, {}).setdefault(met, np.zeros((len(use), len(leads), 5)))[e, j] = v
    grid_err = np.sqrt(grid_err / len(use))

    res = {"year": a.year, "n_ic": len(use), "ics": use, "leads": leads, "vars": REF_VARS, "grid": "WB2 64x32",
           "truth_coarsening_rmse": {v: {str(h): float(grid_err[j, k]) for j, h in enumerate(leads)}
                                     for k, v in enumerate(REF_VARS)},
           "scores": {}}

    def put(name, met, arr, ci=False):
        d = res["scores"].setdefault(name, {})
        for j, h in enumerate(leads):
            for k, v in enumerate(REF_VARS):
                x = arr[:, j, k]
                d.setdefault(str(h), {}).setdefault(met, {})[v] = {"mean": float(x.mean()),
                                                                    "ci95": boot_ci(x) if ci else None}

    for name, mets in per.items():
        for met, arr in mets.items():
            if name == "persistence" and met == "spread":
                continue
            put(name, met, arr, ci=name == "ours")
            if name != "ours" and met != "spread":
                put(f"ours-{name}", met, per["ours"][met] - arr, ci=True)

    check = {}
    if ref is not None:
        for name, rs in ref["scores"].items():
            if name.endswith("-climode") or (cp is not None and name == "climode"):
                continue
            key = name if name in ("climode", "persistence") else f"pgflow:{name}"
            if key == "persistence":  # recomputed: compare
                diff = max(abs(rs[str(h)][met][v]["mean"] - res["scores"]["persistence"][str(h)][met][v]["mean"])
                           / max(abs(rs[str(h)][met][v]["mean"]), 1e-12)
                           for h in leads for met in ("rmse", "acc", "crps_gauss") for v in REF_VARS)
                check["persistence_max_rel_diff_vs_reference"] = float(diff)
                continue
            res["scores"][key] = {str(h): {met: {v: {"mean": rs[str(h)][met][v]["mean"], "ci95": None}
                                                 for v in REF_VARS}
                                           for met in rs[str(h)] if met in ("rmse", "acc", "crps_gauss", "spread")}
                                  for h in leads if str(h) in rs}
        res["reference"] = a.reference
    res["checks"] = check

    out = a.out or os.path.join(a.eval, "climode_compare")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "scores.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    md = table(res)
    with open(os.path.join(out, "table.md"), "w") as fh:
        fh.write(md)
    print(md)
    return res


def table(res):
    leads = res["leads"]
    names = [n for n in res["scores"] if "-" not in n or n.startswith("pgflow:")]
    label = {"ours": "climate_vlm", "climode": "ClimODE", "persistence": "persistence"}
    lines = [f"# {res['year']}: climate_vlm vs ClimODE (WeatherBench2 64x32, {res['n_ic']} ICs)", ""]
    for met, title, fmt in (("rmse", "RMSE", "{:.3g}"), ("acc", "ACC", "{:.3f}"),
                            ("crps_gauss", "Gaussian CRPS (ClimODE min-max units)", "{:.4f}")):
        lines += [f"## {title}", "", "| var | model | " + " | ".join(f"{h // 24} d" for h in leads) + " |",
                  "|---|---|" + "---:|" * len(leads)]
        for v in res["vars"]:
            for n in names:
                d = res["scores"][n]
                if met not in d.get(str(leads[0]), {}):
                    continue
                vals = [d[str(h)][met][v]["mean"] if str(h) in d else float("nan") for h in leads]
                lines.append(f"| {v} | {label.get(n, n)} | " + " | ".join(fmt.format(x) for x in vals) + " |")
        lines.append("")
    if "ours-climode" in res["scores"]:
        lines += ["## Paired difference climate_vlm - ClimODE (95 % bootstrap over ICs)", ""]
        d = res["scores"]["ours-climode"]
        for met in ("rmse", "crps_gauss"):
            for v in res["vars"]:
                lines.append(f"- {met} {v}: " + ", ".join(
                    f"{h} h {d[str(h)][met][v]['mean']:.3g} [{d[str(h)][met][v]['ci95'][0]:.3g}, "
                    f"{d[str(h)][met][v]['ci95'][1]:.3g}]" for h in leads))
        lines.append("")
    lines += [f"Units: {', '.join(f'{v} {u}' for v, u in UNITS.items())}.",
              "truth coarsening RMSE (our HR truth averaged to 64x32 vs WB2 64x32), z500 / t2m at 24 h: "
              f"{res['truth_coarsening_rmse']['z']['24']:.3g} / {res['truth_coarsening_rmse']['t2m']['24']:.3g}."]
    if res.get("checks"):
        lines.append(f"Checks: {json.dumps(res['checks'])}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
