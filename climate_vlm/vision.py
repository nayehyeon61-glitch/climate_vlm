"""Stage 2 inputs: weather maps -> pretrained vision encoder E -> resampler P -> LLM-sized weather tokens.

Path A of the plan (section 04): reuse the natural-image encoder of the same VLM (Qwen3.5's ``visual`` tower,
including its own patch merger, which already projects into the LLM width). Each image holds three
variables of one token type at one history time, one per RGB channel:

    token types  lr (LR input interpolated to HR), mean (D_mu), tendency (d mean / dt), spread (optional)
    colour map   value = clip(z / range + 0.5) with z the z-score from FIXED train statistics, so absolute
                 intensity is kept across images (no per-image min/max); clipped fractions are reported
    resizing     integer ``image_scale`` (bilinear) so that H and W are multiples of patch * merge

The encoder outputs, tagged with learned time / type / group embeddings and the token's sphere position,
are read by a Perceiver resampler with ``n_tokens`` queries (the LLM token budget, plan: 256). Optional
numeric tokens (a patch embedding of the raw z-scores) join the resampler memory to carry what the colour
map loses; they do not change the LLM token count.
"""

from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import fourier


class WeatherImages(nn.Module):
    """Normalised fields -> RGB images in [0, 1] for the vision encoder."""

    def __init__(self, state_channels: Sequence[str], groups: Sequence[Sequence[str]], ranges: Dict[str, float]):
        super().__init__()
        self.groups = [list(g) for g in groups]
        for g in self.groups:
            if not 1 <= len(g) <= 3:
                raise ValueError(f"vision group {g}: 1 to 3 variables per image")
        idx = [[state_channels.index(v) for v in g] + [-1] * (3 - len(g)) for g in self.groups]
        self.register_buffer("idx", torch.tensor(idx), persistent=False)
        self.ranges = dict(ranges)

    def forward(self, maps: Dict[str, torch.Tensor], types: Sequence[str]):
        """maps[type] (B, S, C, H, W) -> images (B, S, K, G, 3, H, W) and the clipped fraction."""
        out, clipped = [], []
        for k in types:
            x = maps[k]
            pad = torch.zeros_like(x[:, :, :1])
            xp = torch.cat([x, pad], 2)  # index -1 -> zero channel
            img = xp[:, :, self.idx]  # (B, S, G, 3, H, W)
            v = img / (2.0 * self.ranges[k]) + (0.0 if k == "spread" else 0.5)
            clipped.append(((v < 0) | (v > 1)).float().mean())
            out.append(v.clamp(0, 1))
        return torch.stack(out, 2), torch.stack(clipped).mean()


class QwenVisionEncoder(nn.Module):
    """Frozen Qwen3.5 vision tower (patch embed, ViT blocks, patch merger) on (N, 3, H, W) images in [0, 1].

    Patches are flattened exactly like the Qwen2-VL / Qwen3-VL image processor (temporal frame duplicated,
    2 x 2 merge windows contiguous), so the pretrained weights see their usual input layout.
    """

    def __init__(self, visual: nn.Module, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5),
                 image_scale: int = 1):
        super().__init__()
        self.visual = visual
        cfg = visual.config
        self.p, self.tp, self.m = cfg.patch_size, cfg.temporal_patch_size, cfg.spatial_merge_size
        self.out_dim = cfg.out_hidden_size
        self.scale = image_scale
        self.register_buffer("mean", torch.tensor(image_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(image_std).view(1, 3, 1, 1), persistent=False)

    def grid(self, H: int, W: int):
        h, w = H * self.scale, W * self.scale
        unit = self.p * self.m
        if h % unit or w % unit:
            raise ValueError(f"image {h}x{w} must be a multiple of {unit}: change image_scale")
        return h // self.p, w // self.p

    def patchify(self, img: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) -> (N * gh * gw, 3 * tp * p * p) in the Qwen processor order."""
        N, C, H, W = img.shape
        p, m, tp = self.p, self.m, self.tp
        gh, gw = H // p, W // p
        x = img[:, None].expand(N, tp, C, H, W)  # one image = tp identical frames
        x = x.reshape(N, 1, tp, C, gh // m, m, p, gw // m, m, p)
        x = x.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)  # (N, t, gh/m, gw/m, m, m, C, tp, p, p)
        return x.reshape(N * gh * gw, C * tp * p * p)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) in [0, 1] -> (N, T, D_llm) merged tokens, row-major over the merged grid."""
        if self.scale != 1:
            img = F.interpolate(img, scale_factor=self.scale, mode="bilinear", align_corners=False)
        img = (img - self.mean) / self.std
        N, _, H, W = img.shape
        gh, gw = H // self.p, W // self.p
        thw = torch.tensor([[1, gh, gw]] * N, device=img.device)
        dtype = next(self.visual.parameters()).dtype
        out = self.visual(self.patchify(img).to(dtype), grid_thw=thw).pooler_output
        return out.view(N, (gh // self.m) * (gw // self.m), -1)


class Resampler(nn.Module):
    """Perceiver resampler P: memory (B, N, D_mem) -> (B, n_tokens, D_llm)."""

    def __init__(self, d_mem: int, d_llm: int, n_tokens: int, dim: int, heads: int, layers: int):
        super().__init__()
        self.inp = nn.Linear(d_mem, dim)
        self.queries = nn.Parameter(torch.randn(n_tokens, dim) * 0.02)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict({
                "nq": nn.LayerNorm(dim), "nm": nn.LayerNorm(dim),
                "att": nn.MultiheadAttention(dim, heads, batch_first=True),
                "nf": nn.LayerNorm(dim),
                "ff": nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))}))
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, d_llm))

    def forward(self, mem: torch.Tensor) -> torch.Tensor:
        m = self.inp(mem)
        q = self.queries[None].expand(len(m), -1, -1)
        for b in self.blocks:
            mm = b["nm"](m)
            q = q + b["att"](b["nq"](q), mm, mm, need_weights=False)[0]
            q = q + b["ff"](b["nf"](q))
        return self.out(q)


def sphere_positions(lat_deg, lon_deg, gh: int, gw: int, n_freq: int = 4) -> torch.Tensor:
    """(gh * gw, 3 + 6 n_freq) Fourier features of the unit-sphere centre of each cell of a gh x gw tiling."""
    la = np.deg2rad(np.asarray(lat_deg, dtype=np.float64)).reshape(gh, -1).mean(1)
    lo = np.asarray(lon_deg, dtype=np.float64).reshape(gw, -1)
    lo = np.deg2rad(lo[:, 0] + (lo[:, -1] - lo[:, 0]) / 2)
    L, G = np.meshgrid(la, lo, indexing="ij")
    xyz = torch.tensor(np.stack([np.cos(L) * np.cos(G), np.cos(L) * np.sin(G), np.sin(L)], -1).reshape(-1, 3),
                       dtype=torch.float32)
    return torch.cat([xyz, fourier(xyz, n_freq)], -1)


class WeatherTokenizer(nn.Module):
    """Images + encoder + tags + resampler: token maps of the history -> (B, n_tokens, D_llm)."""

    def __init__(self, encoder: QwenVisionEncoder, images: WeatherImages, types: Sequence[str], n_hist: int,
                 lat, lon, hr_shape, n_tokens: int, dim: int, heads: int, layers: int,
                 numeric_patch: int = 0, n_state: int = 0):
        super().__init__()
        self.encoder, self.images, self.types = encoder, images, list(types)
        H, W = hr_shape
        gh, gw = encoder.grid(H, W)
        gh, gw = gh // encoder.m, gw // encoder.m
        d = encoder.out_dim
        self.register_buffer("pos", sphere_positions(lat, lon, gh, gw), persistent=False)
        self.pos_mlp = nn.Sequential(nn.Linear(self.pos.shape[1], d), nn.GELU(), nn.Linear(d, d))
        self.time_emb = nn.Parameter(torch.zeros(n_hist, d))
        self.type_emb = nn.Parameter(torch.zeros(len(self.types), d))
        self.group_emb = nn.Parameter(torch.zeros(len(images.groups), d))
        for p in (self.time_emb, self.type_emb, self.group_emb):
            nn.init.normal_(p, std=0.02)
        self.numeric = None
        if numeric_patch:
            if H % numeric_patch or W % numeric_patch:
                raise ValueError("numeric_patch must divide the HR grid")
            self.numeric = nn.Conv2d(len(self.types) * n_state, d, numeric_patch, stride=numeric_patch)
            self.register_buffer("npos", sphere_positions(lat, lon, H // numeric_patch, W // numeric_patch),
                                 persistent=False)
            self.npos_mlp = nn.Sequential(nn.Linear(self.npos.shape[1], d), nn.GELU(), nn.Linear(d, d))
        self.resampler = Resampler(d, d, n_tokens, dim, heads, layers)

    def forward(self, maps: Dict[str, torch.Tensor], encoder_grad: bool = False):
        """maps[type] (B, S, C, H, W) -> tokens (B, n_tokens, D_llm), info dict."""
        img, clipped = self.images(maps, self.types)  # (B, S, K, G, 3, H, W)
        B, S, K, G = img.shape[:4]
        with torch.set_grad_enabled(encoder_grad and torch.is_grad_enabled()):
            feats = self.encoder(img.flatten(0, 3))  # (B*S*K*G, T, d)
        feats = feats.to(self.time_emb.dtype).view(B, S, K, G, feats.shape[1], -1)
        tag = (self.time_emb[:, None, None, None] + self.type_emb[None, :, None, None]
               + self.group_emb[None, None, :, None] + self.pos_mlp(self.pos)[None, None, None])
        mem = (feats + tag[None]).flatten(1, 4)
        if self.numeric is not None:
            x = torch.cat([maps[k] for k in self.types], 2).flatten(0, 1)  # (B*S, K*C, H, W)
            n = self.numeric(x).flatten(2).transpose(1, 2)  # (B*S, P, d)
            n = n + self.npos_mlp(self.npos)[None]
            n = n.view(B, S, *n.shape[1:]) + self.time_emb[None, :, None]
            mem = torch.cat([mem, n.flatten(1, 2)], 1)
        return self.resampler(mem), {"clip_frac": clipped, "n_memory": mem.shape[1]}


class OperatorTokenizer(nn.Module):
    """Stage 04: a neural operator in place of the vision tower (neuraloperator's SFNO, trainable).

    maps[type] (B, S, C, H, W) for every type and history time are stacked as input channels, the SFNO maps them to
    ``out`` latent fields on the same grid, a strided convolution cuts those into patch tokens, and the Perceiver
    resampler P reads them (with their sphere positions) into ``n_tokens`` LLM tokens, exactly as for the vision
    tower. Band types (band0, band1, ...) keep the spherical-harmonic split visible to the operator.
    """

    def __init__(self, types: Sequence[str], n_hist: int, n_state: int, lat, lon, hr_shape, d_llm: int,
                 n_tokens: int, dim: int, heads: int, layers: int, hidden: int = 64, n_layers: int = 4,
                 out: int = 256, patch: int = 8, modes=None):
        super().__init__()
        from neuralop.models import SFNO  # optional dependency: pip install neuraloperator torch-harmonics
        H, W = hr_shape
        if H % patch or W % patch:
            raise ValueError("sfno patch must divide the HR grid")
        self.types = list(types)
        cin = len(self.types) * n_hist * n_state
        self.operator = SFNO(n_modes=tuple(modes or (H // 2, W // 2)), in_channels=cin, out_channels=out,
                             hidden_channels=hidden, n_layers=n_layers)
        self.embed = nn.Conv2d(out, out, patch, stride=patch)
        self.register_buffer("pos", sphere_positions(lat, lon, H // patch, W // patch), persistent=False)
        self.pos_mlp = nn.Sequential(nn.Linear(self.pos.shape[1], out), nn.GELU(), nn.Linear(out, out))
        self.resampler = Resampler(out, d_llm, n_tokens, dim, heads, layers)

    def forward(self, maps: Dict[str, torch.Tensor], encoder_grad: bool = False):
        x = torch.cat([maps[k] for k in self.types], 2).flatten(1, 2).float()  # (B, S*K*C, H, W)
        z = self.embed(self.operator(x)).flatten(2).transpose(1, 2)  # (B, P, out)
        mem = z + self.pos_mlp(self.pos)[None]
        return self.resampler(mem), {"clip_frac": torch.zeros((), device=x.device), "n_memory": mem.shape[1]}
