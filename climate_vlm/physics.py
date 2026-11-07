"""Quantitative physics diagnostics Diag(X) and residuals on de-normalised fields.

All derivatives use the real spherical grid spacing (equiangular, no poles, lat S -> N,
periodic longitude). Plan, section 06:

    advection      A_T = -v_h . grad_p T            (850 hPa, K/day)
                   A_q = -v_h . grad_p q            (850 hPa, g/kg/day)
    moisture       C_q = -div_p(q v_h)              (850 hPa flux convergence, g/kg/day)
    geopotential   thickness (Phi_500 - Phi_850) / g  (m)
    hydrostatic    R_Phi = dPhi/dln p + R_d T_v,  T_v = T (1 + 0.61 q)   (layer form between output levels)

The moisture equation R_q = d_t q + v.grad q + omega d_p q - S_q is NOT used as a residual:
the data have no omega and no source term S_q, and the plan rules out setting them to zero.
Moisture enters only as supervised diagnostics (A_q, C_q) and as q >= 0.
"""

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .contract import CONST, split_name

DAY = 86400.0
DIAGS = {  # name: (needed channels, unit)
    "tadv850": (("u850", "v850", "t850"), "K day-1"),
    "qadv850": (("u850", "v850", "q850"), "g kg-1 day-1"),
    "qconv850": (("u850", "v850", "q850"), "g kg-1 day-1"),
    "thick500_850": (("z500", "z850"), "m"),
}


class SphereOps:
    """d/dx = 1/(a cos phi) d/dlambda (centred, periodic), d/dy = 1/a d/dphi (centred, one-sided at the edges)."""

    def __init__(self, lat_deg: Sequence[float], lon_deg: Sequence[float]):
        self.lat = torch.tensor(np.deg2rad(np.asarray(lat_deg, dtype=np.float64)), dtype=torch.float32)
        lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
        self.dlon = float(lon[1] - lon[0])
        self.a = CONST.earth_radius

    def _cos(self, x):
        return torch.cos(self.lat).to(x).view(-1, 1)

    def ddx(self, f: torch.Tensor) -> torch.Tensor:
        d = (torch.roll(f, -1, -1) - torch.roll(f, 1, -1)) / (2 * self.dlon)
        return d / (self.a * self._cos(f))

    def ddy(self, f: torch.Tensor) -> torch.Tensor:
        return torch.gradient(f, spacing=(self.lat.to(f),), dim=-2)[0] / self.a

    def div(self, fu: torch.Tensor, fv: torch.Tensor) -> torch.Tensor:
        c = self._cos(fu)
        dlam = (torch.roll(fu, -1, -1) - torch.roll(fu, 1, -1)) / (2 * self.dlon)
        dphi = torch.gradient(fv * c, spacing=(self.lat.to(fv),), dim=-2)[0]
        return (dlam + dphi) / (self.a * c)


def area_weight(lat_deg: Sequence[float], pole_cap_deg: Optional[float] = None) -> torch.Tensor:
    lat = np.asarray(lat_deg, dtype=np.float64)
    w = np.cos(np.deg2rad(lat))
    if pole_cap_deg is not None:
        w = np.where(np.abs(lat) <= pole_cap_deg, w, 0.0)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def patch_pool(x: torch.Tensor, p: int, w_lat: torch.Tensor) -> torch.Tensor:
    """(..., H, W) -> (..., H/p, W/p) area-weighted mean over p x p patches."""
    H, W = x.shape[-2:]
    w = w_lat.to(x).view(H, 1)
    num = (x * w).reshape(*x.shape[:-2], H // p, p, W // p, p).sum((-3, -1))
    den = w.reshape(H // p, p).sum(-1, keepdim=True) * p
    return num / den.clamp_min(1e-12)


class Physics:
    """Diagnostics and residuals for one channel list (state channels or decoder output channels)."""

    def __init__(self, channels: Sequence[str], lat, lon, mean: np.ndarray, std: np.ndarray,
                 diags: Sequence[str] = tuple(DIAGS), pole_cap_deg: float = 85.0):
        self.channels = list(channels)
        self.ix = {c: i for i, c in enumerate(self.channels)}
        self.ops = SphereOps(lat, lon)
        self.w = area_weight(lat, pole_cap_deg)
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.diags = [d for d in diags if all(c in self.ix for c in DIAGS[d][0])]
        # hydrostatic layers between consecutive output levels that carry both z and t
        levels = sorted({split_name(c)[1] for c in self.channels if split_name(c)[0] == "z"} &
                        {split_name(c)[1] for c in self.channels if split_name(c)[0] == "t"}, reverse=True)
        self.layers = list(zip(levels[:-1], levels[1:]))  # (p_low, p_up) in hPa, p_low > p_up

    def phys(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """Channel ``name`` of a normalised (..., C, H, W) tensor in ERA5 units."""
        i = self.ix[name]
        return x[..., i, :, :] * self.std[i].to(x) + self.mean[i].to(x)

    def diagnose(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised (..., C, H, W) -> (..., n_diag, H, W) diagnostics in the units of DIAGS."""
        f = lambda n: self.phys(x, n)  # noqa: E731
        out = []
        for d in self.diags:
            if d == "tadv850":
                u, v, t = f("u850"), f("v850"), f("t850")
                out.append(-(u * self.ops.ddx(t) + v * self.ops.ddy(t)) * DAY)
            elif d == "qadv850":
                u, v, q = f("u850"), f("v850"), f("q850")
                out.append(-(u * self.ops.ddx(q) + v * self.ops.ddy(q)) * DAY * 1e3)
            elif d == "qconv850":
                u, v, q = f("u850"), f("v850"), f("q850")
                out.append(-self.ops.div(q * u, q * v) * DAY * 1e3)
            elif d == "thick500_850":
                out.append((f("z500") - f("z850")) / CONST.gravity)
        return torch.stack(out, -3)

    def hydrostatic_residual(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """(..., n_layers, H, W) layer residual (Phi_up - Phi_low) - R_d Tv_mean ln(p_low / p_up), in metres."""
        if not self.layers:
            return None
        out = []
        for lo, up in self.layers:
            tv = []
            for p in (lo, up):
                t = self.phys(x, f"t{p}")
                q = self.phys(x, f"q{p}") if f"q{p}" in self.ix else torch.zeros_like(t)
                tv.append(t * (1.0 + CONST.eps_virtual * q))
            thick = self.phys(x, f"z{up}") - self.phys(x, f"z{lo}")
            out.append((thick - CONST.r_dry * 0.5 * (tv[0] + tv[1]) * math.log(lo / up)) / CONST.gravity)
        return torch.stack(out, -3)

    def negative_q(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """(..., n_q, H, W) negative part of specific humidity in units of its std."""
        qs = [c for c in self.channels if split_name(c)[0] == "q"]
        if not qs:
            return None
        return torch.stack([torch.relu(-self.phys(x, c)) / self.std[self.ix[c]].to(x) for c in qs], -3)

    def wmean(self, e: torch.Tensor) -> torch.Tensor:
        """Area-weighted mean over the last two dims."""
        return (e * self.w.to(e).view(-1, 1)).mean((-1, -2))


def diag_stats(phys: Physics, batches: List[torch.Tensor], patch: int) -> Dict[str, np.ndarray]:
    """Mean / std of the patch-pooled diagnostics over normalised state batches (..., C, H, W)."""
    n = len(phys.diags)
    v = torch.cat([patch_pool(phys.diagnose(b), patch, phys.w).movedim(-3, 0).reshape(n, -1) for b in batches], 1)
    v = v.double()
    return {"names": np.array(phys.diags), "mean": v.mean(1).numpy(), "std": v.std(1).clamp_min(1e-12).numpy()}
