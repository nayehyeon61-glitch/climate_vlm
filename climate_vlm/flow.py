"""Stage 07: a flow-matching ensemble generator after the decoder (the "forecast expert").

Like the action expert of a VLA (pi0: a small transformer that attends to the VLM's prefix and turns noise into an
action chunk by flow matching), FlowExpert attends to the VLM's decoder memory (weather tokens + forecast queries +
physics-head tokens, the tokens F reads) and turns noise into a forecast chunk: every lead of the scored variables at
once, so a member is one trajectory, coherent in space and across leads.

    target   r = (y - base) / s        y truth, base the decoder mean mu (or the issue-time state x0 for the
                                       control without the VLM), s the training std per (lead, variable)
    path     x_t = (1 - (1 - sigma_min) t) x_0 + t r,  x_0 ~ N(0, I)   (Lipman et al. 2023, eq. 22, OT path)
    loss     || v(x_t, t | cond) - (r - (1 - sigma_min) x_0) ||^2, cos-latitude weighted           (eq. 23)
    sampling Euler from t = 0 to 1 (pi0 uses 10 steps); members are independent noise draws

noise="structured": x_0 is drawn from StructuredNoise instead of N(0, I), a Gaussian fitted to the training targets
(spherical-harmonic power per lead and variable, correlation between leads). The pilot's members were coherent inside
a token but independent between tokens and leads; with this source they inherit the targets' covariance.

cond: the patches of mu and / or x0 (inputs of each token) and, with use_mem, cross-attention to the memory.
Tokens: one per (lead, lat patch, lon patch); embeddings for the patch position and the lead; the flow time enters
through adaptive layer norm (DiT's adaLN-Zero).
"""

import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from climate_vlm.spectral import lat_weights, legendre


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """(B,) in [0, 1] -> (B, dim) sinusoidal features."""
    half = dim // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = 1000.0 * t.float()[:, None] * f[None]
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mem_dim: int = 0, mlp: int = 4):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross = None
        if mem_dim:
            self.nc = nn.LayerNorm(dim, eps=1e-6)
            self.cross = nn.MultiheadAttention(dim, heads, kdim=mem_dim, vdim=mem_dim, batch_first=True)
            self.gc = nn.Parameter(torch.zeros(dim))  # cross-attention starts closed
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp * dim), nn.GELU(approximate="tanh"), nn.Linear(mlp * dim, dim))
        self.ada = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor, mem: Optional[torch.Tensor]) -> torch.Tensor:
        s1, b1, g1, s2, b2, g2 = self.ada(F.silu(c))[:, None].chunk(6, -1)
        h = self.n1(x) * (1 + s1) + b1
        x = x + g1 * self.attn(h, h, h, need_weights=False)[0]
        if self.cross is not None and mem is not None:
            x = x + self.gc * self.cross(self.nc(x), mem, mem, need_weights=False)[0]
        h = self.n2(x) * (1 + s2) + b2
        return x + g2 * self.mlp(h)


class StructuredNoise(nn.Module):
    """Gaussian source noise (B, L, V, H, W) with roughly the covariance of the flow's targets.

    Spatial: white noise filtered with a gain per total wavenumber l (lead and variable), so that its spherical-harmonic
    power spectrum matches the targets'; the targets' variance outside the transform's range (l or m above the
    grid's limit) is added as white noise projected out of that range. Leads: mixed per variable with the Cholesky
    factor of the targets' correlation between leads. Each (lead, variable) is scaled to unit area-weighted variance.
    fit() sets all of it from targets.
    """

    def __init__(self, n_leads: int, n_vars: int, lat_deg: Sequence[float], W: int):
        super().__init__()
        lat = np.asarray(lat_deg, np.float64)
        H = len(lat)
        self.L, self.V, self.H, self.W = n_leads, n_vars, H, W
        self.lmax, self.M = H - 1, min(H - 1, W // 2)
        P = legendre(self.lmax, self.M, np.sin(np.deg2rad(lat)))  # (M + 1, lmax + 1, H)
        self.register_buffer("P", torch.tensor(P, dtype=torch.float32))
        self.register_buffer("w", torch.tensor(lat_weights(lat), dtype=torch.float32))
        wa = np.cos(np.deg2rad(lat))
        self.register_buffer("wa", torch.tensor(wa / wa.mean(), dtype=torch.float32))  # area weights, mean 1
        self.register_buffer("gain", torch.ones(n_leads, n_vars, self.lmax + 1))
        self.register_buffer("white", torch.zeros(n_leads, n_vars))
        self.register_buffer("chol", torch.eye(n_leads).repeat(n_vars, 1, 1))
        self.register_buffer("norm", torch.ones(n_leads, n_vars))

    def _coef(self, x: torch.Tensor) -> torch.Tensor:
        """(..., H, W) -> (..., lmax + 1, M + 1) spherical-harmonic coefficients (complex)."""
        f = torch.fft.rfft(x.float(), dim=-1)[..., :self.M + 1]
        return torch.einsum("mlh,...hm->...lm", (self.P * self.w).to(f.dtype), f)

    def power(self, x: torch.Tensor) -> torch.Tensor:
        """(..., H, W) -> (..., lmax + 1) power per total wavenumber."""
        a = self._coef(x).abs() ** 2
        return a[..., 0] + 2 * a[..., 1:].sum(-1)

    def _filter(self, x: torch.Tensor, gain: bool = True) -> torch.Tensor:
        """Synthesis of x's coefficients times the gain (without it: the projection on the transform's range)."""
        a = self._coef(x)
        if gain:
            a = a * self.gain[..., None].to(a.dtype)
        g = torch.einsum("mlh,...lm->...hm", self.P.to(a.dtype), a)
        g = F.pad(g, (0, self.W // 2 + 1 - g.shape[-1]))
        return torch.fft.irfft(g, n=self.W, dim=-1)

    def _rest(self, shape, generator=None) -> torch.Tensor:
        """White noise outside the transform's range (what the gains cannot shape)."""
        e = torch.randn(shape, generator=generator)
        return e - self._filter(e, gain=False)

    def _draw(self, shape, generator=None) -> torch.Tensor:
        x = self._filter(torch.randn(shape, generator=generator))
        x = x + self.white[..., None, None] * self._rest(shape, generator)
        return torch.einsum("vij,bjvhw->bivhw", self.chol, x)

    def forward(self, shape, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        return self._draw(shape, generator) * self.norm[..., None, None]

    def _var(self, x: torch.Tensor) -> torch.Tensor:  # (N, L, V, H, W) -> (L, V) area-weighted variance
        return (x ** 2 * self.wa[:, None]).mean((0, -2, -1))

    @torch.no_grad()
    def fit(self, r: torch.Tensor, draws: int = 64, seed: int = 0):
        """r (N, L, V, H, W): targets of the flow (training windows)."""
        g = torch.Generator().manual_seed(seed)
        r = r.float()
        r = r / torch.sqrt(self._var(r))[..., None, None]  # each (lead, variable) to unit variance
        e = torch.randn((draws,) + tuple(r.shape[1:]), generator=g)
        self.gain.copy_(torch.sqrt(self.power(r).mean(0) / self.power(e).mean(0).clamp_min(1e-12)))
        rest = self._rest(e.shape, g)
        self.white.copy_(torch.sqrt((1.0 - self._var(self._filter(e))).clamp_min(0.0) / self._var(rest)))
        pre = torch.sqrt(self._var(self._filter(e) + self.white[..., None, None] * rest))
        self.gain.div_(pre[..., None])  # unit variance per (lead, variable) before the leads are mixed
        self.white.div_(pre)
        c = torch.einsum("nivhw,njvhw,h->vij", r, r, self.wa) / (r.shape[0] * self.H * self.W)
        c = (c + 1e-4 * torch.eye(self.L)) / (1 + 1e-4)  # unit diagonal: r is standardised
        self.chol.copy_(torch.linalg.cholesky(c))
        self.norm.fill_(1.0)
        self.norm.copy_(1.0 / torch.sqrt(self._var(self._draw(e.shape, g))))
        return self


class FlowExpert(nn.Module):
    """v(x_t, t | mu, x0, memory) on (B, L, V, H, W) chunks; see the module docstring."""

    def __init__(self, n_leads: int, n_vars: int, H: int, W: int, patch=(4, 4), dim: int = 128, depth: int = 4,
                 heads: int = 4, mem_dim: int = 256, use_mem: bool = True, use_mu: bool = True, use_x0: bool = True,
                 sigma_min: float = 1e-4, noise: str = "white", lat_deg: Optional[Sequence[float]] = None):
        super().__init__()
        if noise not in ("white", "structured"):
            raise ValueError(f"noise must be white or structured, not {noise}")
        if noise == "structured" and lat_deg is None:
            raise ValueError("structured noise needs the grid's latitudes")
        ph, pw = patch
        if H % ph or W % pw:
            raise ValueError(f"grid {H}x{W} does not split into {ph}x{pw} patches")
        self.L, self.V, self.H, self.W, self.ph, self.pw = n_leads, n_vars, H, W, ph, pw
        self.use_mem, self.use_mu, self.use_x0, self.sigma_min = use_mem, use_mu, use_x0, sigma_min
        self.n_h, self.n_w = H // ph, W // pw
        k = ph * pw * n_vars
        self.inp = nn.Linear(k * (1 + use_mu + use_x0), dim)
        self.pos = nn.Parameter(torch.randn(1, 1, self.n_h * self.n_w, dim) * 0.02)
        self.lead = nn.Parameter(torch.randn(1, n_leads, 1, dim) * 0.02)
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.mem_proj = nn.Sequential(nn.LayerNorm(mem_dim), nn.Linear(mem_dim, dim)) if use_mem else None
        self.blocks = nn.ModuleList(Block(dim, heads, dim if use_mem else 0) for _ in range(depth))
        self.n_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, k)
        for m in (self.ada_out, self.out):  # the velocity starts at zero
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.noise = StructuredNoise(n_leads, n_vars, lat_deg, W) if noise == "structured" else None

    def source(self, shape, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """x_0: N(0, I), or the fitted StructuredNoise."""
        return torch.randn(shape, generator=generator) if self.noise is None else self.noise(shape, generator)

    # (B, L, V, H, W) <-> (B, L, n_h * n_w, ph * pw * V)
    def _patch(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape[:2]
        x = x.reshape(B, L, self.V, self.n_h, self.ph, self.n_w, self.pw)
        return x.permute(0, 1, 3, 5, 4, 6, 2).reshape(B, L, self.n_h * self.n_w, -1)

    def _unpatch(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape[:2]
        x = x.reshape(B, L, self.n_h, self.n_w, self.ph, self.pw, self.V)
        return x.permute(0, 1, 6, 2, 4, 3, 5).reshape(B, L, self.V, self.H, self.W)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, mu: Optional[torch.Tensor] = None,
                x0: Optional[torch.Tensor] = None, memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x_t, mu (B, L, V, H, W); x0 (B, V, H, W); memory (B, N, mem_dim); t (B,) -> velocity like x_t."""
        parts = [self._patch(x_t)]
        if self.use_mu:
            parts.append(self._patch(mu))
        if self.use_x0:
            parts.append(self._patch(x0[:, None].expand(-1, self.L, -1, -1, -1)))
        h = self.inp(torch.cat(parts, -1)) + self.pos + self.lead  # (B, L, P, dim)
        B, L, P, D = h.shape
        h = h.reshape(B, L * P, D)
        c = self.t_mlp(time_embedding(t, D))
        mem = self.mem_proj(memory.float()) if self.use_mem else None
        for blk in self.blocks:
            h = blk(h, c, mem)
        s, b = self.ada_out(F.silu(c))[:, None].chunk(2, -1)
        h = self.out(self.n_out(h) * (1 + s) + b)
        return self._unpatch(h.reshape(B, L, P, -1))

    def loss(self, r: torch.Tensor, w_lat: torch.Tensor, t: Optional[torch.Tensor] = None, **cond) -> torch.Tensor:
        """Conditional flow matching on the OT path; r (B, L, V, H, W) the scaled target, w_lat (H,) mean 1."""
        B = r.shape[0]
        t = torch.rand(B, device=r.device) if t is None else t
        x0 = self.source(r.shape).to(r)
        tt = t.view(B, 1, 1, 1, 1)
        x_t = (1 - (1 - self.sigma_min) * tt) * x0 + tt * r
        u = r - (1 - self.sigma_min) * x0
        err = (self(x_t, t, **cond) - u) ** 2
        return (err * w_lat.view(1, 1, 1, -1, 1)).mean()

    @torch.no_grad()
    def sample(self, n: int, steps: int = 10, generator: Optional[torch.Generator] = None, **cond) -> torch.Tensor:
        """n members per condition: cond tensors (B, ...) -> (B, n, L, V, H, W), Euler on [0, 1]."""
        B = next(v for v in cond.values() if v is not None).shape[0]
        rep = {k: (None if v is None else v.repeat_interleave(n, 0)) for k, v in cond.items()}
        x = self.source((B * n, self.L, self.V, self.H, self.W), generator)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((B * n,), i * dt)
            x = x + dt * self(x, t, **rep)
        return x.view(B, n, self.L, self.V, self.H, self.W)
