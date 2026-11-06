"""Scores in ERA5 units, same definitions as pg-flow (scripts/run_step11_climode.py climode_scores):

rmse   sqrt(mean_hw(alpha (P - O)^2)), alpha = cos(lat) / mean cos(lat)
acc    anomalies against a train climatology, field mean removed, alpha-weighted correlation
crps   Gaussian closed form (per-grid marginal) or the ensemble estimator, alpha-weighted mean
"""

import math

import numpy as np
import torch

from .contract import area_weights


def rmse(P: np.ndarray, O: np.ndarray, lat) -> np.ndarray:
    """(..., H, W) -> (...)."""
    w = area_weights(lat)[:, None]
    return np.sqrt(((P - O) ** 2 * w).mean((-1, -2)))


def acc(P: np.ndarray, O: np.ndarray, clim: np.ndarray, lat) -> np.ndarray:
    w = area_weights(lat)[:, None]
    pa, oa = P - clim, O - clim
    pa = pa - pa.mean((-1, -2), keepdims=True)
    oa = oa - oa.mean((-1, -2), keepdims=True)
    return (w * pa * oa).sum((-1, -2)) / np.sqrt((w * pa ** 2).sum((-1, -2)) * (w * oa ** 2).sum((-1, -2)))


def crps_gaussian(mu: np.ndarray, sigma: np.ndarray, O: np.ndarray, lat) -> np.ndarray:
    z = (O - mu) / sigma
    erf = torch.erf(torch.from_numpy(np.asarray(z / math.sqrt(2), dtype=np.float64))).numpy()
    c = sigma * (z * erf + 2 * np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi))
    return (c * area_weights(lat)[:, None]).mean((-1, -2))


def crps_ensemble(E: np.ndarray, O: np.ndarray, lat, member_axis: int = 1) -> np.ndarray:
    """E (B, M, ...) members, O (B, ...): E|X - O| - 0.5 E|X - X'| (biased estimator, as properscoring)."""
    E = np.moveaxis(E, member_axis, 0)
    skill = np.abs(E - O[None]).mean(0)
    spread = np.abs(E[:, None] - E[None]).mean((0, 1))
    return ((skill - 0.5 * spread) * area_weights(lat)[:, None]).mean((-1, -2))
