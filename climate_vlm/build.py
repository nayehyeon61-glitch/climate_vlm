"""Construction shared by the scripts: config, cache metadata, statistics, physics, objective, model."""

import os
from typing import Dict, Optional

import numpy as np
import torch
import yaml

from . import data, physics
from .losses import Objective
from .model import ClimateVLM, make_downscaler


def load_config(path: str) -> Dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def years(spec) -> list:
    """[1979, 2015] style ranges ("1979-2015") or explicit lists."""
    out = []
    for s in spec if isinstance(spec, list) else [spec]:
        if isinstance(s, str) and "-" in s:
            a, b = s.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(s))
    return out


def paths(cfg: Dict) -> Dict[str, str]:
    c = cfg["data"]["cache"]
    return {"cache": c, "stats": os.path.join(c, "stats.npz"), "clim": os.path.join(c, "clim.npy"),
            "diag": os.path.join(c, "diag_stats.npz")}


def physics_pair(cfg: Dict, meta: Dict, norm: data.Normalizer):
    """Physics on the state channels (issue-time labels) and on the output channels (forecast)."""
    ch = meta["channels"]
    st = [ch.index(c) for c in meta["state_channels"]]
    diags = cfg["model"].get("diagnostics", list(physics.DIAGS))
    ps = physics.Physics(meta["state_channels"], meta["lat"], meta["lon"], norm.mean[st], norm.std[st], diags)
    po = physics.Physics(ch, meta["lat"], meta["lon"], norm.mean, norm.std, diags)
    return ps, po


def dataset(cfg: Dict, split: str, ics: Optional[str] = None) -> data.ForecastDataset:
    """data.context_years (history only, e.g. 2019 before a 2020 test year) apply to validation and test."""
    d = cfg["data"]
    context = years(d.get("context_years", [])) if split != "train" else []
    return data.ForecastDataset(paths(cfg)["cache"], years(d[f"{split}_years"]), paths(cfg)["stats"], d["history_h"],
                                d["leads_h"], d["factor"], ics=ics or ("all" if split == "train" else "climode"),
                                hours=d.get("ic_hours", [0, 6, 12, 18]), stride=d.get("ic_stride", 1), context=context)


def diag_stats(cfg: Dict) -> Dict[str, np.ndarray]:
    z = np.load(paths(cfg)["diag"])
    if int(z["patch"]) != cfg["model"]["decoder"]["patch"]:
        raise ValueError("diag_stats.npz was computed for another decoder patch: rerun prepare_data.py --stats")
    return {k: z[k] for k in ("names", "mean", "std")}


def model(cfg: Dict, lora: bool = False, device="cpu") -> ClimateVLM:
    from .llm import load_qwen
    meta = data.load_meta(paths(cfg)["cache"])
    q = cfg["model"]["qwen"]
    qwen = load_qwen(q.get("name_or_path", "Qwen/Qwen3.5-0.8B"), q.get("dtype", "bfloat16"), q.get("tiny"))
    cfg = {**cfg, "model": {**cfg["model"], "lora": {**cfg["model"].get("lora", {}), "enabled": lora}}}
    n_diag = len(diag_stats(cfg)["names"])
    m = ClimateVLM(cfg, meta, qwen, data.static_fields(paths(cfg)["cache"]), n_diag).to(device)
    m.qwen_tokenizer = qwen.tokenizer
    return m


def downscaler(cfg: Dict, device="cpu"):
    meta = data.load_meta(paths(cfg)["cache"])
    return make_downscaler(cfg, meta).to(device), data.static_fields(paths(cfg)["cache"]).to(device), meta


def objective(cfg: Dict, meta: Dict, norm: data.Normalizer, stage: Optional[str] = None) -> Objective:
    """Loss weights: cfg.loss, overridden by cfg.train.stage<stage>.loss (plan p.7: forecast first, then
    diagnostics, then physics residuals with a small weight)."""
    ps, po = physics_pair(cfg, meta, norm)
    h = cfg["data"]["history_h"]
    w = {**cfg.get("loss", {}), **(cfg["train"].get(f"stage{stage}", {}).get("loss", {}) if stage else {})}
    return Objective(ps, po, diag_stats(cfg), cfg["model"]["decoder"]["patch"], w,
                     issue_slot=h.index(0) if 0 in h else len(h) - 1, lat=meta["lat"],
                     phys_floor_m=w.get("phys_floor_m", 1.0))


def load_own(module: torch.nn.Module, path: str, prefix: str = "") -> None:
    """Load a checkpoint written with ClimateVLM.own_state_dict (or a downscaler state under 'D.')."""
    sd = torch.load(path, map_location="cpu", weights_only=False)["state"]
    if prefix:
        sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = module.load_state_dict(sd, strict=False)
    if unexpected:
        raise KeyError(f"unexpected keys in {path}: {unexpected[:5]}")
