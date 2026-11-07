"""Spherical-harmonic band split and a ClimODE-style advection step (stage 04).

SphericalBands   x (..., H, W) on a regular lat-lon grid -> K band fields with sum exactly x. Band b keeps the
                 total wavenumbers l in (edges[b-1], edges[b]]; the last band is the rest (x minus the lower bands),
                 so splitting and recombining is lossless. Transform: rfft in longitude, then a Legendre projection
                 per zonal wavenumber m with Fejer's first quadrature rule, which is the rule of a grid of cell
                 centres without poles (WeatherBench2 64x32 and 128x64). It is exact for l + l' < n_lat, so the
                 low bands are orthogonal projections on these grids. A plain 2D FFT (ClimateLLM) treats latitude as
                 periodic, which it is not.
Advection        u (B, A, H, W), piecewise-constant velocity / source per lead segment -> u at every lead.
                 du/dt = -v . grad u - c u div v + s (ClimODE's continuity form; c = 1 with compression) on the
                 grid in cell units (as ClimODE: no sphere metric), integrated semi-Lagrangian (unconditionally
                 stable) with longitude periodic and latitude clamped.
"""

import math
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fejer1_weights(theta: np.ndarray, n: int) -> np.ndarray:
    """Weights in x = cos(theta) of Fejer's first rule, nodes theta_k = (2k + 1) pi / (2n) in any order (sum 2)."""
    j = np.arange(1, n // 2 + 1)
    return 2.0 / n * (1 - 2 * (np.cos(2 * np.outer(theta, j)) / (4 * j ** 2 - 1)).sum(1))


def legendre(lmax: int, mmax: int, x: np.ndarray) -> np.ndarray:
    """(mmax + 1, lmax + 1, n) associated Legendre functions, orthonormal on [-1, 1] (zero where l < m)."""
    x = np.asarray(x, np.float64)
    s = np.sqrt(np.clip(1 - x ** 2, 0, None))
    P = np.zeros((mmax + 1, lmax + 1, len(x)))
    pmm = np.full_like(x, np.sqrt(0.5))
    for m in range(mmax + 1):
        if m > 0:
            pmm = -np.sqrt((2 * m + 1) / (2 * m)) * s * pmm
        if m > lmax:
            break
        P[m, m] = pmm
        if m + 1 <= lmax:
            P[m, m + 1] = np.sqrt(2 * m + 3) * x * pmm
        for l in range(m + 2, lmax + 1):
            a = np.sqrt((4 * l * l - 1) / (l * l - m * m))
            b = np.sqrt(((l - 1) ** 2 - m * m) / (4 * (l - 1) ** 2 - 1))
            P[m, l] = a * (x * P[m, l - 1] - b * P[m, l - 2])
    return P


def lat_weights(lat_deg: Sequence[float]) -> np.ndarray:
    """Quadrature weights in sin(lat) for the given latitudes: Fejer's first rule when they are the cell centres of
    a regular pole-to-pole grid, else the exact cell areas."""
    lat = np.asarray(lat_deg, np.float64)
    n = len(lat)
    ref = -90 + (np.arange(n) + 0.5) * 180.0 / n
    if np.allclose(np.sort(lat), ref, atol=1e-4):
        return fejer1_weights(np.deg2rad(90 - lat), n)
    d = np.abs(np.gradient(lat))
    return np.abs(np.sin(np.deg2rad(np.clip(lat + d / 2, -90, 90))) - np.sin(np.deg2rad(np.clip(lat - d / 2, -90, 90))))


class SphericalBands(nn.Module):
    """Band split of fields on a regular lat-lon grid (see the module docstring)."""

    def __init__(self, lat_deg: Sequence[float], lon_deg: Sequence[float], edges: Sequence[int] = (6, 20)):
        super().__init__()
        lat = np.asarray(lat_deg, np.float64)
        self.H, self.W = len(lat), len(lon_deg)
        self.edges = [int(e) for e in edges]
        if sorted(self.edges) != self.edges or self.edges[-1] >= self.H:
            raise ValueError(f"band edges {edges} must increase and stay below n_lat {self.H}")
        lmax = self.H - 1
        self.M = min(lmax, self.W // 2)
        P = legendre(lmax, self.M, np.sin(np.deg2rad(lat)))  # (M+1, L+1, H)
        w = lat_weights(lat)
        l = np.arange(lmax + 1)
        K = []
        for e in self.edges:  # projector on l <= e, per m: P^T diag(1_{l<=e}) P diag(w)
            Pm = P * (l <= e)[None, :, None]
            K.append(np.einsum("mlh,mlk->mhk", Pm, P) * w[None, None, :])
        self.register_buffer("K", torch.tensor(np.stack(K), dtype=torch.float32), persistent=False)  # (E, M+1, H, H)

    @property
    def n_bands(self) -> int:
        return len(self.edges) + 1

    def lowpass(self, x: torch.Tensor) -> List[torch.Tensor]:
        """x (..., H, W) -> [projection on l <= e for e in edges]."""
        dt = x.dtype
        f = torch.fft.rfft(x.float(), dim=-1)[..., :self.M + 1]  # (..., H, M+1)
        out = []
        for K in self.K:
            g = torch.einsum("mhk,...km->...hm", K.to(torch.complex64), f)
            out.append(torch.fft.irfft(g, n=self.W, dim=-1).to(dt))
        return out

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """x (..., H, W) -> K band fields, each (..., H, W); their sum is x."""
        low = self.lowpass(x)
        bands = [low[0]] + [b - a for a, b in zip(low[:-1], low[1:])]
        return bands + [x - low[-1]]

    def combine(self, parts: torch.Tensor) -> torch.Tensor:
        """parts (..., K, H, W): one field per band -> sum over bands of each part projected on its own band."""
        if parts.shape[-3] != self.n_bands:
            raise ValueError(f"expected {self.n_bands} band parts, got {parts.shape[-3]}")
        low = self.lowpass(parts)  # each (..., K, H, W)
        out = low[0][..., 0, :, :]
        for b in range(1, len(self.edges)):
            out = out + low[b][..., b, :, :] - low[b - 1][..., b, :, :]
        k = self.n_bands - 1
        return out + parts[..., k, :, :] - low[-1][..., k, :, :]


class Advection(nn.Module):
    """Semi-Lagrangian continuity step on the grid (cell units), piecewise-constant velocity and source per segment.

    velocity: tanh(raw) * max_cells_per_h, so one segment of 24 h moves at most 24 * max_cells_per_h cells
    compression: u <- u * exp(-c * tanh(div v * dt / c)), a bounded form of du/dt = -u div v (c = 0.25)
    source: s per day, added as s * dt / 24
    """

    def __init__(self, dt_h: float = 6.0, max_cells_per_h: float = 1.0, compression: bool = True):
        super().__init__()
        self.dt_h, self.vmax, self.compression = float(dt_h), float(max_cells_per_h), compression

    @staticmethod
    def _div(v: torch.Tensor) -> torch.Tensor:
        """v (N, 2, H, W) in cells/h (x = lon, y = lat index) -> central-difference divergence (N, H, W)."""
        vx, vy = v[:, 0], v[:, 1]
        dx = (torch.roll(vx, -1, -1) - torch.roll(vx, 1, -1)) / 2
        p = F.pad(vy[:, None], (0, 0, 1, 1), mode="replicate")[:, 0]
        dy = (p[:, 2:] - p[:, :-2]) / 2
        return dx + dy

    def _warp(self, u: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """u (N, 1, H, W) sampled at x - d, d (N, 2, H, W) displacement in cells; lon periodic, lat clamped."""
        N, _, H, W = u.shape
        pad = int(math.ceil(float(d[:, 0].abs().max()))) + 2
        up = torch.cat([u[..., -pad:], u, u[..., :pad]], -1)
        yy, xx = torch.meshgrid(torch.arange(H, device=u.device, dtype=u.dtype),
                                torch.arange(W, device=u.device, dtype=u.dtype), indexing="ij")
        x = xx[None] - d[:, 0] + pad
        y = (yy[None] - d[:, 1]).clamp(0, H - 1)
        grid = torch.stack([2 * x / (W + 2 * pad - 1) - 1, 2 * y / (H - 1) - 1], -1)
        return F.grid_sample(up, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def forward(self, u0: torch.Tensor, vel: torch.Tensor, src: torch.Tensor, leads_h: Sequence[int]) -> torch.Tensor:
        """u0 (B, A, H, W); vel (B, L, A, 2, H, W) raw; src (B, L, A, H, W) per day -> (B, L, A, H, W)."""
        B, L, A, _, H, W = vel.shape
        u, t, out = u0.reshape(B * A, 1, H, W), 0.0, []
        for j, h in enumerate(leads_h):
            v = torch.tanh(vel[:, j].reshape(B * A, 2, H, W)) * self.vmax
            s = src[:, j].reshape(B * A, 1, H, W)
            n = max(1, int(math.ceil((h - t) / self.dt_h)))
            dt = (h - t) / n
            fac = None
            if self.compression:
                c = 0.25
                fac = torch.exp(-c * torch.tanh(self._div(v)[:, None] * dt / c))
            for _ in range(n):
                u = self._warp(u, v * dt)
                if fac is not None:
                    u = u * fac
                u = u + s * dt / 24.0
            out.append(u.view(B, A, H, W))
            t = float(h)
        return torch.stack(out, 1)
