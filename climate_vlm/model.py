"""Climate VLM: D (downscaler) -> E (vision encoder) -> P (resampler) -> LLM (+LoRA) -> G (physics) / F (fields).

    x_lr (B, S, C_state, h, w)  --D-->  lr / mean / tendency (/ spread) maps on the HR grid
                                --E,P-> weather tokens (B, n_tokens, D_llm)
    [prompt | weather tokens | forecast queries | explanation text]  --LLM-->  hidden states
    hidden(weather, queries) --proj--> memory --G--> z^ (diagnostics), Z_phys
                                          [memory, Z_phys] --F--> Y^ (B, L, C_out, H, W)

The explanation text sits after the query tokens, so (causal LLM) no hidden state that G or F reads has
seen it: teacher-forced future explanations cannot leak into the numeric forecast (plan p.5).
"""

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from . import contract
from .decoder import ForecastDecoder, GeoEmbedding, LeadEmbedding, PhysicsHead
from .downscaler import Downscaler
from .llm import DEFAULT_LORA_TARGETS, QwenParts, add_lora
from .vision import QwenVisionEncoder, WeatherImages, WeatherTokenizer

FROZEN_PREFIXES = ("tokenizer.encoder.", "lm_head.")


def make_downscaler(cfg: Dict, meta: Dict) -> Downscaler:
    ds = cfg["model"]["downscaler"]
    return Downscaler(len(meta["state_channels"]), cfg["data"]["factor"], (len(meta["lat"]), len(meta["lon"])),
                      lon_pad=ds.get("lon_pad", 16), unet=ds.get("unet"), residual=ds.get("residual", False),
                      res_unet=ds.get("res_unet"), sampler_steps=ds.get("sampler_steps", 18))


class ClimateVLM(nn.Module):
    def __init__(self, cfg: Dict, meta: Dict, qwen: QwenParts, static_hr: torch.Tensor, n_diag: int):
        super().__init__()
        m, v, d = cfg["model"], cfg["model"]["vision"], cfg["model"]["decoder"]
        self.state = list(meta["state_channels"])
        self.out = list(meta["channels"])
        self.history_h, self.leads_h = list(cfg["data"]["history_h"]), list(cfg["data"]["leads_h"])
        lat, lon = meta["lat"], meta["lon"]
        H, W = len(lat), len(lon)
        ds = m["downscaler"]
        self.D = make_downscaler(cfg, meta)
        self.spread_members = ds.get("spread_members", 0) if ds.get("residual", False) else 0
        self.register_buffer("static_hr", static_hr, persistent=False)
        types = ["lr", "mean", "tendency"] + (["spread"] if self.spread_members > 1 else [])
        enc = QwenVisionEncoder(qwen.visual, qwen.image_mean, qwen.image_std, v.get("image_scale", 1))
        self.tokenizer = WeatherTokenizer(
            enc, WeatherImages(self.state, v["groups"], v["ranges"]), types, len(self.history_h), lat, lon, (H, W),
            n_tokens=v["n_tokens"], dim=v["dim"], heads=v["heads"], layers=v["layers"],
            numeric_patch=v.get("numeric_patch", 0), n_state=len(self.state))
        self.llm, self.lm_head, self.D_llm = qwen.language_model, qwen.lm_head, qwen.hidden_size
        self.n_query = m.get("n_query_tokens", 16)
        self.fq = nn.Parameter(torch.randn(self.n_query, self.D_llm) * 0.02)
        dim = d["dim"]
        self.hproj = nn.Sequential(nn.LayerNorm(self.D_llm), nn.Linear(self.D_llm, dim))
        geo, lead = GeoEmbedding(lat, lon, d["patch"], dim), LeadEmbedding(dim)
        self.G = PhysicsHead(geo, lead, [0] + self.leads_h, n_diag, dim, d["heads"], d.get("g_layers", 2))
        self.F = ForecastDecoder(geo, lead, self.leads_h, len(self.out), d["patch"], dim, d["heads"],
                                 d.get("f_layers", 2), d.get("self_attn", True), d.get("gaussian", False),
                                 d.get("cnn_hidden", 64))
        self.base = d.get("residual_base", "persistence")
        self.out_from_state = [self.state.index(c) if c in self.state else -1 for c in self.out]
        if m.get("lora", {}).get("enabled", False):
            lo = m["lora"]
            add_lora(self.llm, lo.get("r", 16), lo.get("alpha", 32), lo.get("dropout", 0.05),
                     lo.get("targets") or DEFAULT_LORA_TARGETS)
        if m.get("gradient_checkpointing", False):
            self.llm.gradient_checkpointing_enable()
        self.d_frozen = True

    # ------------------------------------------------------------------------------------------ stages
    def set_stage(self, stage: str) -> List[nn.Parameter]:
        """'1' D_mu, '1res' D_res, '2' P/F/G (LLM frozen, gradient still flows through it), '3' + LoRA."""
        for p in self.parameters():
            p.requires_grad_(False)
        if stage == "1":
            mods = [self.D.mean]
        elif stage == "1res":
            mods = [self.D.res]
        elif stage in ("2", "3"):
            mods = [self.tokenizer, self.hproj, self.G, self.F]
        else:
            raise ValueError(stage)
        for mod in mods:
            for p in mod.parameters():
                p.requires_grad_(True)
        for p in self.tokenizer.encoder.parameters():
            p.requires_grad_(False)
        if stage in ("2", "3"):
            self.fq.requires_grad_(True)
        if stage == "3":
            for n, p in self.llm.named_parameters():
                if "lora_" in n:
                    p.requires_grad_(True)
        self.d_frozen = stage not in ("1", "1res")
        self.train(self.training)
        return [p for p in self.parameters() if p.requires_grad]

    def train(self, mode: bool = True):
        """Frozen parts stay in eval mode (CorrDiff's UNet has dropout; the plan freezes D and E after stage 1)."""
        super().train(mode)
        if self.d_frozen:
            self.D.eval()
        self.tokenizer.encoder.eval()
        return self

    def own_state_dict(self) -> Dict[str, torch.Tensor]:
        """Everything this repository trains: D, P (+ tags / numeric tokens), queries, G, F and LoRA."""
        return {k: v for k, v in self.state_dict().items()
                if not k.startswith(FROZEN_PREFIXES) and (not k.startswith("llm.") or "lora_" in k)}

    # ------------------------------------------------------------------------------------------ forward
    def _embed_ids(self, ids: List[List[int]], left: bool, device) -> (torch.Tensor, torch.Tensor):
        n = max((len(i) for i in ids), default=0)
        tok = torch.zeros(len(ids), n, dtype=torch.long, device=device)
        mask = torch.zeros(len(ids), n, dtype=torch.long, device=device)
        for b, i in enumerate(ids):
            if i:
                sl = slice(n - len(i), n) if left else slice(0, len(i))
                tok[b, sl], mask[b, sl] = torch.tensor(i, device=device), 1
        return self.llm.get_input_embeddings()(tok), mask

    def forward(self, x_lr: torch.Tensor, prompt_ids: Optional[List[List[int]]] = None,
                text_ids: Optional[List[List[int]]] = None) -> Dict[str, torch.Tensor]:
        B, dev = x_lr.shape[0], x_lr.device
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.d_frozen):
            maps = self.D.tokens(x_lr, self.static_hr, self.history_h, members=self.spread_members)
        wtok, info = self.tokenizer(maps)
        dtype = next(self.llm.parameters()).dtype
        pe, pm = self._embed_ids(prompt_ids or [[] for _ in range(B)], True, dev)
        te, tm = self._embed_ids(text_ids or [[] for _ in range(B)], False, dev)
        q = self.fq[None].expand(B, -1, -1)
        emb = torch.cat([pe.to(dtype), wtok.to(dtype), q.to(dtype), te.to(dtype)], 1)
        ones = torch.ones(B, wtok.shape[1] + self.n_query, dtype=torch.long, device=dev)
        mask = torch.cat([pm, ones, tm], 1)
        hidden = self.llm(inputs_embeds=emb, attention_mask=mask).last_hidden_state
        p0, n_mem = pe.shape[1], wtok.shape[1] + self.n_query
        mem = self.hproj(hidden[:, p0:p0 + n_mem].float())
        z_hat, z_tok = self.G(mem)
        base = None
        if self.base == "persistence":  # downscaled issue-time state (zero for output-only channels)
            now = maps["mean"][:, self.history_h.index(0) if 0 in self.history_h else -1]
            base = torch.cat([now, torch.zeros_like(now[:, :1])], 1)[:, self.out_from_state]
        mean, scale = self.F(torch.cat([mem, z_tok], 1), base)
        out = {"mean": mean, "scale": scale, "z_hat": z_hat, "maps": maps, **info}
        if te.shape[1]:
            s = p0 + n_mem
            out["text_logits"] = self.lm_head(hidden[:, s - 1:s - 1 + te.shape[1]]).float()
            out["text_mask"] = tm
        return out


def output_units(channels: Sequence[str]) -> List[str]:
    return [contract.unit_of(c) for c in channels]
