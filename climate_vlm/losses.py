"""Training objective (plan section 07):

    L = L_forecast + lam_d L_diag + lam_p L_phys + lam_c L_cons + lam_t L_text

L_forecast  area-weighted MSE of the normalised future fields, or Gaussian CRPS when F outputs a scale
L_diag      G's z^ against z*_t = Diag(X_<=t) (issue time, from the HR analysis) and z*_{t+h} = Diag(Y_{t+h})
            (future truth, label only, never an input)
L_phys      weak hydrostatic constraint: mean square layer residual of Y^ in excess of the residual of the
            analysis itself (the discrete operator is not exact on ERA5 either: plan p.6 "reference residual"),
            relative to that reference + floor^2 (m^2); plus q >= 0
L_cons      || z^_{t+h} - Diag(Y^_{t+h}) ||^2: the process quantities G states must match the predicted fields
L_text      token cross-entropy of the template explanation (optional)

Diagnostics are patch-pooled (area weighted) and normalised by train statistics so the terms are unit-free.
"""

import math
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from .physics import Physics, area_weight, patch_pool


def gaussian_crps(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = (y - mu) / sigma
    pdf = torch.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


class Objective:
    def __init__(self, phys_state: Physics, phys_out: Physics, diag_stats: Dict[str, np.ndarray], patch: int,
                 weights: Dict[str, float], issue_slot: int, lat, phys_floor_m: float = 1.0):
        self.ps, self.po, self.patch = phys_state, phys_out, patch
        names = list(diag_stats["names"])
        if phys_state.diags != phys_out.diags or phys_out.diags != names:
            raise ValueError(f"diagnostics differ: state {phys_state.diags}, output {phys_out.diags}, stats {names}")
        self.dm = torch.tensor(diag_stats["mean"], dtype=torch.float32).view(-1, 1, 1)
        self.ds = torch.tensor(diag_stats["std"], dtype=torch.float32).view(-1, 1, 1)
        self.lam = {k: float(weights.get(k, 0.0)) for k in ("diag", "phys", "cons", "text")}
        self.issue_slot, self.floor = issue_slot, phys_floor_m
        self.w = area_weight(lat)  # forecast loss: every row, cos(lat)
        wp = phys_out.w.view(-1, patch).mean(1)  # diagnostics: polar cap excluded
        self.wp = wp / wp.mean().clamp_min(1e-12)

    def zdiag(self, phys: Physics, x: torch.Tensor) -> torch.Tensor:
        """Normalised (..., C, H, W) -> normalised patch diagnostics (..., n_diag, Hp, Wp)."""
        z = patch_pool(phys.diagnose(x), self.patch, phys.w)
        return (z - self.dm.to(z)) / self.ds.to(z)

    def targets(self, x_hr: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """(B, 1 + L, n_diag, Hp, Wp): issue-time and future diagnostic labels."""
        now = self.zdiag(self.ps, x_hr[:, self.issue_slot])
        return torch.cat([now[:, None], self.zdiag(self.po, y)], 1)

    def __call__(self, out: Dict, batch: Dict) -> (torch.Tensor, Dict[str, float]):
        y, mu, sd = batch["y"], out["mean"], out["scale"]
        w = self.w.to(y).view(-1, 1)
        if sd is None:
            l_f = ((mu - y) ** 2 * w).mean()
        else:
            l_f = (gaussian_crps(mu, sd, y) * w).mean()
        parts = {"forecast": l_f}
        wp = self.wp.to(y).view(-1, 1)
        if self.lam["diag"] or self.lam["cons"]:
            with torch.no_grad():
                z_star = self.targets(batch["x_hr"], y)
            parts["diag"] = ((out["z_hat"] - z_star) ** 2 * wp).mean()
        if self.lam["cons"]:
            parts["cons"] = ((out["z_hat"][:, 1:] - self.zdiag(self.po, mu)) ** 2 * wp).mean()
        if self.lam["phys"]:
            r_hat, r_true = self.po.hydrostatic_residual(mu), self.po.hydrostatic_residual(y)
            l_p = torch.zeros((), device=y.device)
            if r_hat is not None:  # weak constraint: only residual energy beyond the analysis' own residual
                e_hat, e_ref = self.po.wmean(r_hat ** 2), self.po.wmean(r_true ** 2).detach()
                l_p = l_p + (torch.relu(e_hat - e_ref) / (e_ref + self.floor ** 2)).mean()
            neg = self.po.negative_q(mu)
            if neg is not None:
                l_p = l_p + self.po.wmean(neg ** 2).mean()
            parts["phys"] = l_p
        if self.lam["text"] and "text_logits" in out:
            lg, m = out["text_logits"], out["text_mask"]
            ids = batch["text_ids"].to(lg.device)
            parts["text"] = F.cross_entropy(lg.flatten(0, 1), ids.masked_fill(m == 0, -100).flatten(), ignore_index=-100)
        total = l_f + sum(self.lam[k] * v for k, v in parts.items() if k != "forecast")
        return total, {k: float(v.detach()) for k, v in parts.items()}


def downscale_loss(elementwise: torch.Tensor, w_lat: torch.Tensor) -> torch.Tensor:
    """Area-weighted mean of a physicsnemo elementwise loss (B, C, H, W)."""
    return (elementwise * w_lat.to(elementwise).view(-1, 1)).mean()
