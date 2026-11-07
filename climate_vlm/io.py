"""Forecast files in pg-flow's member layout (run_step13_seqfm.py --save-ens), readable by pg-flow's
score_step13.py / compare_step13_climode.py / coarsen_ens.py:

    <out>/<tag>_ens_<year>.npz
        E_minus_x0  (n_ic, M, L, 5, H, W) float16  members minus the issue-time analysis, ERA5 units
        O           (n_ic, L, 5, H, W)    float32  truth
        X0          (n_ic, 5, H, W)       float32  analysis at the issue time
        ics         (n_ic,)               6 h index of the issue time in its year
        leads       (L,)                  lead hours

The five variables are ClimODE's (z500, t850, t2m, u10, v10). A second file ``<tag>_all_<year>.npz`` keeps every
output channel (``channels`` key) with the mean, the Gaussian scale when present, and the truth.
"""

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .contract import CLIMODE_VARS


def write_year(out: str, tag: str, year: int, channels: Sequence[str], mean: np.ndarray, truth: np.ndarray,
               x0: np.ndarray, ics: Sequence[int], leads_h: Sequence[int], members: Optional[np.ndarray] = None,
               scale: Optional[np.ndarray] = None) -> List[str]:
    """mean / truth (n, L, C, H, W), x0 (n, C, H, W), members (n, M, L, C, H, W) or None (M = 1: the mean)."""
    os.makedirs(out, exist_ok=True)
    written = []
    have = [v for v in CLIMODE_VARS if v in channels]
    if len(have) == len(CLIMODE_VARS):
        k = [list(channels).index(v) for v in CLIMODE_VARS]
        E = mean[:, None] if members is None else members
        p = os.path.join(out, f"{tag}_ens_{year}.npz")
        np.savez(p, E_minus_x0=(E[:, :, :, k] - x0[:, None, None, k]).astype(np.float16),
                 O=truth[:, :, k].astype(np.float32), X0=x0[:, k].astype(np.float32),
                 ics=np.asarray(ics), leads=np.asarray(leads_h))
        written.append(p)
    p = os.path.join(out, f"{tag}_all_{year}.npz")
    extra: Dict[str, np.ndarray] = {} if scale is None else {"scale": scale.astype(np.float32)}
    np.savez(p, channels=np.array(channels), mean=mean.astype(np.float32), truth=truth.astype(np.float32),
             X0=x0.astype(np.float32), ics=np.asarray(ics), leads=np.asarray(leads_h), **extra)
    written.append(p)
    return written
