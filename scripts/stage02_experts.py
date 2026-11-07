"""Stage 02, step 2: forecast-correction experts on top of the frozen stage-01 forecast (plan p.2: base / advection
correction / moisture correction, all with the same variables, grid and leads).

    python scripts/stage02_experts.py --config configs/stage02_laya_router.yaml

Expert 0 (baseline) is the base forecast itself. Experts 1 (advection) and 2 (moisture) add a correction from a small
periodic CNN that sees the base forecast of one lead, the lead, and the issue-time maps of its regime
(experts.<name>.inputs; all from the downscaled input, see stage02_forecasts). Both are trained on split A (the
stage-01 training years) with an area- and regime-weighted MSE: pixel weight 1 + sum |diagnostic| / std, so the
advection expert is fitted mostly where temperature advection is strong and the moisture expert where moisture
transport and convergence are. Normalisation statistics come from split A only.

<out>/experts.pt (weights, normalisation) and, for the router periods, <out>/fx_<period>.npz:
    E (n, 3, L, 5, h, w)  the three experts on the routing domain (physical units), O (n, L, 5, h, w) its truth
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from climate_vlm import build  # noqa: E402
from climate_vlm.downscaler import Conv  # noqa: E402
from scripts.stage02_common import EXPERTS, area_w  # noqa: E402


class CorrectionCNN(nn.Module):
    def __init__(self, c_in: int, c_out: int, hidden: int, layers: int):
        super().__init__()
        mods, c = [], c_in
        for _ in range(layers - 1):
            mods += [Conv(c, hidden), nn.GELU()]
            c = hidden
        mods.append(Conv(c, c_out))
        self.net = nn.Sequential(*mods)
        nn.init.zeros_(self.net[-1].conv.weight)
        nn.init.zeros_(self.net[-1].conv.bias)

    def forward(self, x):
        return self.net(x)


class Expert:
    """Normalisation + inputs of one expert; ``apply`` gives physical forecasts (n, L, 5, H, W)."""

    def __init__(self, spec, fmaps, stats, leads, hidden, layers):
        self.fi = [fmaps.index(m) for m in spec["inputs"]]
        self.wi = [fmaps.index(m) for m in spec["weight"]]
        self.stats, self.leads = stats, np.asarray(leads, np.float32)
        self.net = CorrectionCNN(5 + len(self.fi) + 1, 5, hidden, layers)

    def inputs(self, P, F, j):
        """P (n, 5, H, W) physical forecast of lead j, F (n, M, H, W) -> network input, normalised."""
        s = self.stats
        p = (P - s["v_mean"][:, None, None]) / s["v_std"][:, None, None]
        f = (F[:, self.fi] - s["f_mean"][self.fi, None, None]) / s["f_std"][self.fi, None, None]
        lead = np.full((len(P), 1) + P.shape[-2:], self.leads[j] / 144.0, np.float32)
        return torch.from_numpy(np.concatenate([p, f, lead], 1).astype(np.float32))

    def weight(self, F):
        s = self.stats
        return 1.0 + sum(np.abs(F[:, k]) / s["f_std"][k] for k in self.wi)

    def apply(self, P, F, batch=64):
        out = np.empty_like(P)
        sd = self.stats["v_std"][:, None, None]
        with torch.no_grad():
            for j in range(P.shape[1]):
                for i in range(0, len(P), batch):
                    d = self.net(self.inputs(P[i:i + batch, j], F[i:i + batch], j)).numpy()
                    out[i:i + batch, j] = P[i:i + batch, j] + d * sd
        return out


def rmse(E, O, wa):
    """(n, K, L, 5, H, W), (n, L, 5, H, W) -> (K, L, 5) area-weighted RMSE per IC, averaged over ICs."""
    return np.sqrt((((E - O[:, None]) ** 2) * wa).mean((-1, -2)) / wa.mean()).mean(0)


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--config", required=True)
    a.add_argument("--steps", type=int)
    a = a.parse_args(argv)
    cfg = build.load_config(a.config)
    ec, out = cfg["experts"], cfg["out"]
    train_period = [k for k, v in cfg["periods"].items() if v.get("split") == "A"]
    if len(train_period) != 1:
        raise ValueError("exactly one period must have split A (expert training)")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    tr = np.load(os.path.join(out, f"fc_{train_period[0]}.npz"))
    fmaps, leads = [str(m) for m in tr["fmaps"]], [int(h) for h in tr["leads"]]
    P, O, F = tr["P"], tr["O"], tr["F"]
    stats = {"v_mean": O.mean((0, 1, 3, 4)), "v_std": O.std((0, 1, 3, 4)),
             "f_mean": F.mean((0, 2, 3)), "f_std": F.std((0, 2, 3)) + 1e-12}
    wa = area_w(P.shape[-2])
    w_area = torch.from_numpy(wa)
    experts = {}
    for name in EXPERTS[1:]:
        ex = Expert(ec[name], fmaps, stats, leads, ec["hidden"], ec["layers"])
        opt = torch.optim.AdamW(ex.net.parameters(), lr=float(ec["lr"]))
        steps = a.steps or ec["steps"]
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        t0 = time.time()
        for step in range(1, steps + 1):
            i = rng.integers(0, len(P), ec["batch"])
            j = int(rng.integers(0, len(leads)))
            x = ex.inputs(P[i, j], F[i], j)
            r = torch.from_numpy(((O[i, j] - P[i, j]) / stats["v_std"][:, None, None]).astype(np.float32))
            w = torch.from_numpy(ex.weight(F[i]).astype(np.float32))[:, None] * w_area
            loss = ((ex.net(x) - r) ** 2 * w).mean() / w.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if step % 250 == 0 or step == steps:
                base = (r ** 2 * w).mean().item() / w.mean().item()
                print(f"{name} step {step}: weighted MSE {loss.item():.4f} (base {base:.4f}) {time.time() - t0:.0f} s",
                      flush=True)
        experts[name] = ex
    torch.save({"stats": stats, "nets": {k: v.net.state_dict() for k, v in experts.items()}, "config": ec,
                "trained_on": train_period[0]}, os.path.join(out, "experts.pt"))
    for period in cfg["periods"]:
        f = os.path.join(out, f"fc_{period}.npz")
        if not os.path.exists(f):
            print(f"{period}: {f} missing, skipped")
            continue
        z = np.load(f)
        E = np.stack([z["P"]] + [experts[n].apply(z["P"], z["F"]) for n in EXPERTS[1:]], 1)
        r5, c5 = z["rows5"], z["cols5"]
        Ed, Od = E[..., r5[:, None], c5[None, :]], z["O"][..., r5[:, None], c5[None, :]]
        rg, rd = rmse(E, z["O"], wa), rmse(Ed, Od, wa[r5])
        print(f"{period} ({cfg['periods'][period].get('split')}): RMSE z500 / t850 / t2m at {leads[0]} h, "
              "global | domain: " + ", ".join(
                  f"{n} {rg[k, 0, 0]:.1f} / {rg[k, 0, 1]:.2f} / {rg[k, 0, 2]:.2f} | "
                  f"{rd[k, 0, 0]:.1f} / {rd[k, 0, 1]:.2f} / {rd[k, 0, 2]:.2f}" for k, n in enumerate(EXPERTS)),
              flush=True)
        if cfg["periods"][period].get("split") != "A":
            np.savez(os.path.join(out, f"fx_{period}.npz"), E=Ed.astype(np.float32), O=Od.astype(np.float32),
                     rows5=r5, cols5=c5, issue=z["issue"], leads=z["leads"], experts=np.array(EXPERTS))


if __name__ == "__main__":
    main()
