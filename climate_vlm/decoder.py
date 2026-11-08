"""Numeric heads on the shared LLM hidden states (plan sections 05 and 06).

PhysicsHead G      slot (issue time / lead) x patch queries read the LLM memory and predict the patch-pooled
                   diagnostics z_t, z_{t+h}; its tokens Z_phys join the decoder memory.
ForecastDecoder F  q_hij = E_lead(h) + E_geo(lat_i, lon_j)
                   U = CrossAttention(Q, [H_LLM, Z_phys])
                   Y^ = CNN(Unpatchify(MLP(U)))      (+ base field, optional Gaussian scale)

Both read only the hidden states of the input and query positions (never teacher-forced explanation text).
"""

import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .downscaler import Conv


def fourier(x: torch.Tensor, n: int) -> torch.Tensor:
    """(..., d) -> (..., d * 2n) sin/cos features at octave frequencies."""
    f = (2.0 ** torch.arange(n, device=x.device, dtype=x.dtype)) * math.pi
    a = x[..., None] * f
    return torch.cat([torch.sin(a), torch.cos(a)], -1).flatten(-2)


class GeoEmbedding(nn.Module):
    """E_geo(lat, lon) of patch centres from unit-sphere coordinates (continuous across the date line)."""

    def __init__(self, lat_deg: Sequence[float], lon_deg: Sequence[float], patch: int, dim: int, n_freq: int = 6):
        super().__init__()
        la = np.deg2rad(np.asarray(lat_deg, dtype=np.float64)).reshape(-1, patch).mean(1)
        lo = np.asarray(lon_deg, dtype=np.float64).reshape(-1, patch)
        lo = np.deg2rad(lo[:, 0] + (lo[:, -1] - lo[:, 0]) / 2)
        L, G = np.meshgrid(la, lo, indexing="ij")
        xyz = np.stack([np.cos(L) * np.cos(G), np.cos(L) * np.sin(G), np.sin(L)], -1).reshape(-1, 3)
        self.register_buffer("xyz", torch.tensor(xyz, dtype=torch.float32), persistent=False)
        self.mlp = nn.Sequential(nn.Linear(3 * 2 * n_freq + 3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.n_freq, self.grid = n_freq, L.shape

    def forward(self) -> torch.Tensor:
        return self.mlp(torch.cat([self.xyz, fourier(self.xyz, self.n_freq)], -1))  # (P, dim)


class LeadEmbedding(nn.Module):
    """E_lead(h): continuous in hours (issue time h = 0 included), so unseen leads stay meaningful."""

    def __init__(self, dim: int, n_freq: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * n_freq + 1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.n_freq = n_freq

    def forward(self, hours: torch.Tensor) -> torch.Tensor:
        x = (hours.float() / 240.0)[:, None]
        return self.mlp(torch.cat([x, fourier(x, self.n_freq)], -1))


class CrossBlock(nn.Module):
    """Pre-norm cross-attention (+ optional self-attention) + MLP."""

    def __init__(self, dim: int, heads: int, self_attn: bool):
        super().__init__()
        self.nq, self.nm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ns = nn.LayerNorm(dim) if self_attn else None
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True) if self_attn else None
        self.nf = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, q, mem, mem_mask: Optional[torch.Tensor] = None):
        m = self.nm(mem)
        q = q + self.cross(self.nq(q), m, m, key_padding_mask=mem_mask, need_weights=False)[0]
        if self.self_attn is not None:
            s = self.ns(q)
            q = q + self.self_attn(s, s, s, need_weights=False)[0]
        return q + self.ff(self.nf(q))


class PhysicsHead(nn.Module):
    """G: (B, N, dim) memory -> z^ (B, 1 + L, n_diag, Hp, Wp) normalised diagnostics and Z_phys tokens."""

    def __init__(self, geo: GeoEmbedding, lead: LeadEmbedding, slots_h: Sequence[int], n_diag: int, dim: int,
                 heads: int, layers: int):
        super().__init__()
        self.geo, self.lead = geo, lead
        self.register_buffer("slots_h", torch.tensor(list(slots_h)), persistent=False)
        self.blocks = nn.ModuleList([CrossBlock(dim, heads, self_attn=False) for _ in range(layers)])
        self.out = nn.Linear(dim, n_diag)
        self.back = nn.Linear(n_diag, dim)  # quantitative values -> token, so F sees the numbers G committed to
        self.n_diag = n_diag
        self.asked = nn.Parameter(torch.zeros(dim))  # slot "the asked lead" when the lead is given only as text

    def forward(self, mem, mem_mask=None, slots: Optional[torch.Tensor] = None, use_lead: bool = True):
        """slots (B, T) hours per sample (lead questions: [0, asked lead]); default: the fixed slots_h.
        use_lead False: the asked slot gets one learned vector, so the lead reaches G only through the LLM memory."""
        B = mem.shape[0]
        g = self.geo()  # (P, d)
        if slots is None:
            q = (self.lead(self.slots_h)[:, None] + g[None]).flatten(0, 1)[None].expand(B, -1, -1)  # (B, T*P, d)
            T = len(self.slots_h)
        else:
            T = slots.shape[1]
            e = self.lead(slots.flatten()).view(B, T, -1)
            if not use_lead:
                e = torch.where(slots[..., None] > 0, self.lead(torch.zeros_like(slots[:1, :1]).flatten()) + self.asked, e)
            q = (e[:, :, None] + g[None, None]).flatten(1, 2)
        for blk in self.blocks:
            q = blk(q, mem, mem_mask)
        z = self.out(q)  # (B, T*P, n_diag)
        tokens = q + self.back(z)
        Hp, Wp = self.geo.grid
        z = z.view(B, T, Hp, Wp, self.n_diag).permute(0, 1, 4, 2, 3)
        return z, tokens


class ForecastDecoder(nn.Module):
    """F: (B, N, dim) memory -> Y^ (B, L, C, H, W) normalised fields (+ scale when gaussian).

    n_maps (default C): number of mean maps; more than C when the model turns them into fields itself (stage 04:
    one map per channel and spherical-harmonic band, or velocity and source maps of the advection step)."""

    def __init__(self, geo: GeoEmbedding, lead: LeadEmbedding, leads_h: Sequence[int], channels: int, patch: int,
                 dim: int, heads: int, layers: int, self_attn: bool = True, gaussian: bool = False,
                 cnn_hidden: int = 64, n_maps: Optional[int] = None):
        super().__init__()
        self.geo, self.lead, self.patch, self.C, self.gaussian = geo, lead, patch, channels, gaussian
        self.n_maps = n_maps or channels
        self.register_buffer("leads_h", torch.tensor(list(leads_h)), persistent=False)
        self.blocks = nn.ModuleList([CrossBlock(dim, heads, self_attn) for _ in range(layers)])
        n_out = self.n_maps + (channels if gaussian else 0)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
                                  nn.Linear(dim, patch * patch * n_out))
        self.cnn = nn.Sequential(Conv(n_out, cnn_hidden), nn.GELU(), Conv(cnn_hidden, cnn_hidden), nn.GELU(),
                                 Conv(cnn_hidden, n_out))
        nn.init.zeros_(self.cnn[-1].conv.weight)
        nn.init.zeros_(self.cnn[-1].conv.bias)

    def forward(self, mem, base: Optional[torch.Tensor] = None, mem_mask=None, leads: Optional[torch.Tensor] = None,
                use_lead: bool = True):
        """base (B, C, H, W): added to every lead (e.g. the downscaled issue-time state). Returns (mean, scale).
        leads (B,): one asked lead per sample (lead questions) -> (B, 1, C, H, W); use_lead False drops E_lead so
        the lead reaches F only through the LLM memory (the question text)."""
        B = mem.shape[0]
        Hp, Wp = self.geo.grid
        if leads is None:
            L = len(self.leads_h)
            q = (self.lead(self.leads_h)[:, None] + self.geo()[None]).flatten(0, 1)[None].expand(B, -1, -1)
        else:
            L = 1
            q = self.geo()[None].expand(B, -1, -1)
            if use_lead:
                q = q + self.lead(leads)[:, None]
        for blk in self.blocks:
            q = blk(q, mem, mem_mask)
        p, n_out = self.patch, self.n_maps + (self.C if self.gaussian else 0)
        x = self.head(q).view(B * L, Hp, Wp, p, p, n_out)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B * L, n_out, Hp * p, Wp * p)  # unpatchify
        x = (x + self.cnn(x)).view(B, L, n_out, Hp * p, Wp * p)
        mean = x[:, :, :self.n_maps]
        if base is not None:
            mean = mean + base[:, None]
        scale = F.softplus(x[:, :, self.n_maps:]) + 1e-3 if self.gaussian else None
        return mean, scale
