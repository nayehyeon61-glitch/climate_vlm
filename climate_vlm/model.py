"""Climate VLM: D (downscaler) -> E (vision encoder) -> P (resampler) -> LLM (+LoRA) -> G (physics) / F (fields).

Stage 04 options (model.spectral, model.vision.encoder): D's mean maps split into spherical-harmonic bands before E
(token types band0..), a neuraloperator SFNO in place of E, and F's maps recombined per band or advected (ClimODE).

    x_lr (B, S, C_state, h, w)  --D-->  lr / mean / tendency (/ spread) maps on the HR grid
                                --E,P-> weather tokens (B, n_tokens, D_llm)
    [prompt | weather tokens | (question) | forecast queries | explanation text]  --LLM-->  hidden states
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
from .data import lon0_grid
from .downscaler import Downscaler
from .llm import DEFAULT_LORA_TARGETS, QwenParts, add_lora
from .spectral import Advection, SphericalBands
from .vision import OperatorTokenizer, QwenVisionEncoder, WeatherImages, WeatherTokenizer

FROZEN_PREFIXES = ("tokenizer.encoder.", "lm_head.")


def make_downscaler(cfg: Dict, meta: Dict) -> Downscaler:
    ds = cfg["model"]["downscaler"]
    return Downscaler(len(meta["state_channels"]), cfg["data"]["factor"], (len(meta["lat"]), len(meta["lon"])),
                      lon_pad=ds.get("lon_pad", 16), unet=ds.get("unet"), residual=ds.get("residual", False),
                      res_unet=ds.get("res_unet"), sampler_steps=ds.get("sampler_steps", 18),
                      lon0=cfg["data"]["factor"] > 1 and lon0_grid(meta["lon"]))


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
        # stage 04: spherical-harmonic bands of one D map become token types band0, band1, ... (decompose before E)
        sp = m.get("spectral") or {}
        self.bands = SphericalBands(lat, lon, sp.get("edges", [6, 15])) if sp else None
        self.split = sp.get("split", "mean")
        have = ["lr", "mean", "tendency"] + (["spread"] if self.spread_members > 1 else [])
        if self.bands is not None:
            have += [f"band{k}" for k in range(self.bands.n_bands)]
        types = list(v.get("types") or [t for t in have if not t.startswith("band")])
        if set(types) - set(have):
            raise ValueError(f"vision.types {sorted(set(types) - set(have))} not available (have {have})")
        self.llm, self.lm_head, self.D_llm = qwen.language_model, qwen.lm_head, qwen.hidden_size
        if v.get("encoder", "qwen") == "sfno":  # stage 04: neural operator instead of the frozen vision tower
            o = v.get("sfno", {})
            self.tokenizer = OperatorTokenizer(
                types, len(self.history_h), len(self.state), lat, lon, (H, W), self.D_llm, v["n_tokens"], v["dim"],
                v["heads"], v["layers"], hidden=o.get("hidden", 64), n_layers=o.get("layers", 4),
                out=o.get("out", 256), patch=o.get("patch", 8), modes=o.get("modes"))
        else:
            enc = QwenVisionEncoder(qwen.visual, qwen.image_mean, qwen.image_std, v.get("image_scale", 1))
            self.tokenizer = WeatherTokenizer(
                enc, WeatherImages(self.state, v["groups"], v["ranges"]), types, len(self.history_h), lat, lon,
                (H, W), n_tokens=v["n_tokens"], dim=v["dim"], heads=v["heads"], layers=v["layers"],
                numeric_patch=v.get("numeric_patch", 0), n_state=len(self.state))
        self.n_query = m.get("n_query_tokens", 16)
        self.fq = nn.Parameter(torch.randn(self.n_query, self.D_llm) * 0.02)
        dim = d["dim"]
        self.hproj = nn.Sequential(nn.LayerNorm(self.D_llm), nn.Linear(self.D_llm, dim))
        geo, lead = GeoEmbedding(lat, lon, d["patch"], dim), LeadEmbedding(dim)
        self.G = PhysicsHead(geo, lead, [0] + self.leads_h, n_diag, dim, d["heads"], d.get("g_layers", 2))
        qn = m.get("question", {})  # lead questions (stage 03): the lead is asked in text, one lead per sequence
        self.ask, self.ask_lead_emb = bool(qn.get("enabled", False)), bool(qn.get("lead_embedding", True))
        # stage 04 output: direct (base + F), bands (base + sum of each band map projected on its band), advect
        # (ClimODE-style: F gives velocity and source per lead segment for the advected channels, bands for the rest)
        self.out_mode = sp.get("output", "direct")
        C, K = len(self.out), (self.bands.n_bands if self.bands is not None else 1)
        if self.out_mode != "direct" and self.bands is None:
            raise ValueError("spectral.output bands / advect needs model.spectral")
        self.adv_idx = []
        if self.out_mode == "advect":
            ad = sp.get("advect", {})
            self.adv_idx = [self.out.index(c) for c in ad.get("channels", [])]
            if not self.adv_idx or any(self.out_from_state_of(c) < 0 for c in ad.get("channels", [])):
                raise ValueError("spectral.advect.channels: output channels that are also state channels")
            if self.ask:
                raise ValueError("the advection output integrates over every lead: not with lead questions")
            self.advect = Advection(ad.get("dt_h", 6.0), ad.get("max_cells_per_h", 1.0), ad.get("compression", True))
        self.rest_idx = [i for i in range(C) if i not in self.adv_idx]
        n_maps = {"direct": C, "bands": C * K, "advect": 3 * len(self.adv_idx) + len(self.rest_idx) * K}[self.out_mode]
        self.F = ForecastDecoder(geo, lead, self.leads_h, C, d["patch"], dim, d["heads"],
                                 d.get("f_layers", 2), d.get("self_attn", True), d.get("gaussian", False),
                                 d.get("cnn_hidden", 64), n_maps=n_maps)
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
        for p in self.frozen_encoder.parameters():
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
        self.frozen_encoder.eval()
        return self

    @property
    def frozen_encoder(self) -> nn.Module:
        """The pretrained vision tower (never trained); an empty module for the neural-operator tokenizer."""
        enc = getattr(self.tokenizer, "encoder", None)
        return nn.Identity() if enc is None else enc

    def out_from_state_of(self, channel: str) -> int:
        return self.state.index(channel) if channel in self.state else -1

    def own_state_dict(self) -> Dict[str, torch.Tensor]:
        """Everything this repository trains: D, P (+ tags / numeric tokens), queries, G, F and LoRA."""
        return {k: v for k, v in self.state_dict().items()  # (neuraloperator adds a non-tensor "_metadata" entry)
                if torch.is_tensor(v) and not k.startswith(FROZEN_PREFIXES) and (not k.startswith("llm.") or "lora_" in k)}

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

    def _sequence(self, x_lr, prompt_ids, question_ids, B, dev):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.d_frozen):
            maps = self.D.tokens(x_lr, self.static_hr, self.history_h, members=self.spread_members)
        if self.bands is not None:
            maps.update({f"band{k}": b for k, b in enumerate(self.bands(maps[self.split]))})
        wtok, info = self.tokenizer(maps)
        dtype = next(self.llm.parameters()).dtype
        pe, pm = self._embed_ids(prompt_ids or [[] for _ in range(B)], True, dev)
        qe, qm = self._embed_ids(question_ids or [[] for _ in range(B)], True, dev)
        q = self.fq[None].expand(B, -1, -1)
        emb = torch.cat([pe.to(dtype), wtok.to(dtype), qe.to(dtype), q.to(dtype)], 1)
        mask = torch.cat([pm, torch.ones(B, wtok.shape[1], dtype=torch.long, device=dev), qm,
                          torch.ones(B, self.n_query, dtype=torch.long, device=dev)], 1)
        return maps, info, emb, mask, (pe.shape[1], wtok.shape[1], qe.shape[1])

    def forward(self, x_lr: torch.Tensor, prompt_ids: Optional[List[List[int]]] = None,
                text_ids: Optional[List[List[int]]] = None, question_ids: Optional[List[List[int]]] = None,
                ask_h: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Sequence [prompt | weather tokens | question | forecast queries | text]. With ask_h (B,) (lead questions)
        G and F answer only the asked lead: mean (B, 1, C, H, W), z_hat (B, 2, ...) for [issue time, asked lead]."""
        B, dev = x_lr.shape[0], x_lr.device
        maps, info, emb, mask, (p0, n_w, n_q) = self._sequence(x_lr, prompt_ids, question_ids, B, dev)
        te, tm = self._embed_ids(text_ids or [[] for _ in range(B)], False, dev)
        emb = torch.cat([emb, te.to(emb.dtype)], 1)
        mask = torch.cat([mask, tm], 1)
        hidden = self.llm(inputs_embeds=emb, attention_mask=mask).last_hidden_state
        s = p0 + n_w + n_q + self.n_query  # end of the query tokens
        # memory: weather tokens and queries (the question is read by the queries, which come after it)
        mem = self.hproj(torch.cat([hidden[:, p0:p0 + n_w], hidden[:, s - self.n_query:s]], 1).float())
        if ask_h is not None:
            slots = torch.stack([torch.zeros_like(ask_h), ask_h], 1)
            z_hat, z_tok = self.G(mem, slots=slots, use_lead=self.ask_lead_emb)
        else:
            z_hat, z_tok = self.G(mem)
        base = None
        if self.base == "persistence":  # downscaled issue-time state (zero for output-only channels)
            now = maps["mean"][:, self.history_h.index(0) if 0 in self.history_h else -1]
            base = torch.cat([now, torch.zeros_like(now[:, :1])], 1)[:, self.out_from_state]
        memory = torch.cat([mem, z_tok], 1)  # what F reads; also the prefix of stage 07's flow-matching expert
        raw, scale = self.F(memory, base if self.out_mode == "direct" else None, leads=ask_h,
                            use_lead=self.ask_lead_emb)
        mean = raw if self.out_mode == "direct" else self._fields(raw, base)
        out = {"mean": mean, "scale": scale, "z_hat": z_hat, "maps": maps, "memory": memory, **info}
        if te.shape[1]:
            out["text_logits"] = self.lm_head(hidden[:, s - 1:s - 1 + te.shape[1]]).float()
            out["text_mask"] = tm
        return out

    def _fields(self, raw: torch.Tensor, base: Optional[torch.Tensor]) -> torch.Tensor:
        """Stage 04 output maps (B, L, n_maps, H, W) -> fields (B, L, C, H, W): band maps of channel c are
        recombined by their spherical-harmonic projections; advected channels are integrated from the base."""
        B, L, _, H, W = raw.shape
        C, K = len(self.out), self.bands.n_bands
        base = base if base is not None else raw.new_zeros(B, C, H, W)
        out = raw.new_zeros(B, L, C, H, W)
        A = len(self.adv_idx)
        if A:
            vel = raw[:, :, :2 * A].reshape(B, L, A, 2, H, W)
            src = raw[:, :, 2 * A:3 * A]
            out[:, :, self.adv_idx] = self.advect(base[:, self.adv_idx], vel, src, self.leads_h)
        parts = raw[:, :, 3 * A:].reshape(B, L, len(self.rest_idx), K, H, W)
        out[:, :, self.rest_idx] = base[:, None, self.rest_idx] + self.bands.combine(parts)
        return out

    @torch.no_grad()
    def generate(self, x_lr: torch.Tensor, prompt_ids=None, question_ids=None, prefix_ids=None,
                 max_new_tokens: int = 160, eos_id: Optional[int] = None) -> List[List[int]]:
        """Greedy text after the query tokens (the answer to the question), with the LLM's KV cache."""
        B, dev = x_lr.shape[0], x_lr.device
        _, _, emb, mask, _ = self._sequence(x_lr, prompt_ids, question_ids, B, dev)
        if prefix_ids:
            pe, pm = self._embed_ids(prefix_ids, True, dev)
            emb, mask = torch.cat([emb, pe.to(emb.dtype)], 1), torch.cat([mask, pm], 1)
        o = self.llm(inputs_embeds=emb, attention_mask=mask, use_cache=True)
        past, h = o.past_key_values, o.last_hidden_state[:, -1]
        out, done = [[] for _ in range(B)], torch.zeros(B, dtype=torch.bool, device=dev)
        for _ in range(max_new_tokens):
            nxt = self.lm_head(h).argmax(-1)
            for b in range(B):
                if not done[b]:
                    out[b].append(int(nxt[b]))
            if eos_id is not None:
                done |= nxt == eos_id
            if bool(done.all()):
                break
            mask = torch.cat([mask, torch.ones(B, 1, dtype=mask.dtype, device=dev)], 1)
            o = self.llm(inputs_embeds=self.llm.get_input_embeddings()(nxt[:, None]), attention_mask=mask,
                         past_key_values=past, use_cache=True)
            past, h = o.past_key_values, o.last_hidden_state[:, -1]
        return [[t for t in seq if t != eos_id] for seq in out]


def output_units(channels: Sequence[str]) -> List[str]:
    return [contract.unit_of(c) for c in channels]
