"""Stage 02, step 1: forecasts of the frozen stage-01 model and the maps it has at the issue time, per period.

    python scripts/stage02_forecasts.py --config configs/stage02_laya_router.yaml [--periods experts router]

Every input of the experts and of the routers comes from what the model has at inference (guide sec. 5): the
5.625 deg history x_lr, downscaled by the stage-01 D_mu to 2.8125 deg, the same path as the VLM's own diagnostics.
The HR analysis is never an input. Diagnostics are computed in physical units on the full sphere and cropped after.

<out>/fc_<period>.npz:
    years, ics, issue (n,)      year, 6 h index in that year, issue time (hours since 1970); leads (L,) hours
    P (n, L, 5, 32, 64)         base forecast of ClimODE's variables (physical units, WB2 64x32 by data.coarsen)
    O (n, L, 5, 32, 64)         WB2 64x32 truth: labels and scores only
    F (n, len(FMAPS), 32, 64)   expert inputs: D_mu state fields (model normalisation), diagnostics (K/day,
                                g/kg/day), 24 h changes (model normalisation)
    R (n, len(RMAPS), h, w)     router maps on the 2.8125 deg domain in physical units (RMAPS), with the 850 hPa
                                validity mask (1 where Phi_850 > surface geopotential)
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from climate_vlm import build, data  # noqa: E402
from climate_vlm.contract import CLIMODE_VARS  # noqa: E402
from climate_vlm.llm import PromptBuilder  # noqa: E402
from climate_vlm.model import output_units  # noqa: E402
from scripts.stage02_common import G, domain, lr_grid, period_dataset  # noqa: E402

STATE_MAPS = ["z500", "t850", "q850", "u850", "v850", "t2m", "tcwv"]  # model inputs only (u10 / v10 are outputs)
DIAG_MAPS = ["tadv850", "qadv850", "qconv850"]
CHANGE_MAPS = ["dz500_24h", "dt850_24h"]
FMAPS = STATE_MAPS + DIAG_MAPS + CHANGE_MAPS
RMAPS = ["tadv850", "qconv850", "z500_m", "z500_grad", "dz500_24h_m", "dt850_24h", "valid850"]
RUNITS = ["K/day", "g/kg/day", "m", "m/100km", "m", "K", "1"]


class Truth:
    """WB2 64x32 truth by hour since 1970, across year files."""

    def __init__(self, tdir: str, cache: str):
        self.tdir, self.cache, self.arr, self.hours = tdir, cache, {}, {}

    def __call__(self, hour: int) -> np.ndarray:
        y = int(str(np.datetime64(int(hour), "h"))[:4])
        if y not in self.arr:
            self.arr[y] = np.load(os.path.join(self.tdir, f"{y}.npy"), mmap_mode="r")
            hf = os.path.join(self.tdir, f"{y}_hours.npy")
            self.hours[y] = np.load(hf if os.path.exists(hf) else os.path.join(self.cache, f"{y}_hours.npy"))
        k = int(np.searchsorted(self.hours[y], hour))
        if k >= len(self.hours[y]) or int(self.hours[y][k]) != hour:
            raise KeyError(f"no truth at {np.datetime64(int(hour), 'h')}")
        return np.asarray(self.arr[y][k])


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--config", required=True)
    a.add_argument("--periods", nargs="*")
    a.add_argument("--batch", type=int, default=8)
    a.add_argument("--limit", type=int, default=0, help="first n ICs per period (tests)")
    a = a.parse_args(argv)
    cfg = build.load_config(a.config)
    bc = build.load_config(cfg["base"]["config"])
    p = build.paths(bc)
    meta, norm = data.load_meta(p["cache"]), data.Normalizer.load(p["stats"])
    ch, st = meta["channels"], meta["state_channels"]
    lon0 = data.lon0_grid(meta["lon"])
    ps, _ = build.physics_pair(bc, meta, norm)  # diagnostics on the downscaled state channels
    missing = [d for d in DIAG_MAPS if d not in ps.diags]
    if missing:
        raise ValueError(f"diagnostics {missing} need channels the base model does not have")
    kd = [ps.diags.index(d) for d in DIAG_MAPS]
    hist = bc["data"]["history_h"]
    if -24 not in hist or 0 not in hist:
        raise ValueError("the 24 h changes need history_h to hold -24 and 0")
    i24, i0 = hist.index(-24), hist.index(0)
    k5 = [ch.index(v) for v in CLIMODE_VARS]
    ks = [st.index(v) for v in STATE_MAPS]
    kz, kt = st.index("z500"), st.index("t850")
    rows, cols = domain(meta["lat"], meta["lon"], cfg["domain"])  # router maps, 2.8125 deg
    rows5, cols5 = domain(*lr_grid(meta, bc["data"]["factor"]), cfg["domain"])  # routing and scores, 5.625 deg
    orog = np.load(os.path.join(p["cache"], "orog.npy"))  # surface geopotential, m2 s-2
    truth = Truth(os.path.join(p["cache"], "truth_5p625"), p["cache"])

    model = build.model(bc, lora=True)
    build.load_own(model, cfg["base"]["ckpt"])
    model.set_stage("2")
    model.eval()
    pb = PromptBuilder(model.qwen_tokenizer, st, output_units(st), f"{180 / len(meta['lat']) * bc['data']['factor']:g} deg",
                       hist, bc["data"]["leads_h"], compact=bc["model"].get("prompt") == "compact")
    use_prompt = bc["model"].get("prompt", True) and model.qwen_tokenizer is not None
    leads = [int(h) for h in cfg.get("leads", bc["data"]["leads_h"])]
    li = [bc["data"]["leads_h"].index(h) for h in leads]
    out_dir = cfg["out"]
    os.makedirs(out_dir, exist_ok=True)
    phys = lambda m, c: m[:, st.index(c)] * norm.std[ch.index(c)] + norm.mean[ch.index(c)]  # noqa: E731
    for name in a.periods or list(cfg["periods"]):
        ds = period_dataset(bc, cfg["periods"][name], leads)
        if a.limit:
            ds.index = ds.index[:a.limit]
        dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=2)
        acc = {k: [] for k in ("P", "O", "F", "R", "years", "ics", "issue")}
        t0 = time.time()
        for b in dl:
            with torch.no_grad():
                out = model(b["x_lr"], pb.prompt_ids(b["issue"]) if use_prompt else None)
            mu = norm.denorm_t(out["mean"].float(), list(range(len(ch)))).numpy()[:, li][:, :, k5]
            acc["P"].append(data.coarsen(mu, 2, lon0).astype(np.float32))
            m = out["maps"]["mean"].float()  # (B, n_hist, C_state, H, W) D_mu, model normalisation
            m0, m24 = m[:, i0], m[:, i24]
            diag = ps.diagnose(m0)[:, kd]
            dz = (m0[:, kz] - m24[:, kz])[:, None]
            dt = (m0[:, kt] - m24[:, kt])[:, None]
            acc["F"].append(data.coarsen(torch.cat([m0[:, ks], diag, dz, dt], 1).numpy(), 2, lon0).astype(np.float32))
            m0n, m24n = m0.numpy(), m24.numpy()
            z, z24 = phys(m0n, "z500") / G, phys(m24n, "z500") / G
            zt = torch.from_numpy(z)
            grad = (torch.sqrt(ps.ops.ddx(zt) ** 2 + ps.ops.ddy(zt) ** 2) * 1e5).numpy()  # m per 100 km
            valid = (phys(m0n, "z850") > orog).astype(np.float32)  # 850 hPa above the ground
            R = np.stack([diag[:, 0].numpy(), diag[:, 2].numpy(), z, grad, z - z24,
                          phys(m0n, "t850") - phys(m24n, "t850"), valid], 1)
            acc["R"].append(R[..., rows[:, None], cols[None, :]].astype(np.float32))
            for y, t, h0 in zip(b["year"].tolist(), b["t"].tolist(), b["hour0"].tolist()):
                acc["O"].append(np.stack([truth(h0 + h) for h in leads]))
                acc["years"].append(y)
                acc["ics"].append(t)
                acc["issue"].append(h0)
            print(f"{name}: {len(acc['ics'])}/{len(ds)} ICs, {time.time() - t0:.0f} s", flush=True)
        np.savez(os.path.join(out_dir, f"fc_{name}.npz"), P=np.concatenate(acc["P"]),
                 O=np.stack(acc["O"]).astype(np.float32), F=np.concatenate(acc["F"]), R=np.concatenate(acc["R"]),
                 years=np.array(acc["years"]), ics=np.array(acc["ics"]), issue=np.array(acc["issue"], np.int64),
                 leads=np.array(leads), fmaps=np.array(FMAPS), rmaps=np.array(RMAPS), runits=np.array(RUNITS),
                 vars=np.array(CLIMODE_VARS), rows=rows, cols=cols, rows5=rows5, cols5=cols5)
        print(f"{name}: wrote {out_dir}/fc_{name}.npz ({len(ds)} ICs, {time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
