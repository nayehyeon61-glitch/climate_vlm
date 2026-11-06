"""Stage 3 backbone: pretrained Qwen3.5 (vision tower E + hybrid Gated-DeltaNet / attention LLM) and LoRA.

``load_qwen`` returns the parts the climate model uses. ``name_or_path`` is a Hugging Face id or local folder
(default ``Qwen/Qwen3.5-0.8B``). With ``tiny`` (a dict of small text / vision config values) it builds the same
architecture with random weights: this is how the tests run without the checkpoint; it says nothing about the
pretrained model's skill.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn


@dataclass
class QwenParts:
    visual: nn.Module  # Qwen3_5VisionModel (patch embed, ViT, merger -> LLM width)
    language_model: nn.Module  # Qwen3_5TextModel
    lm_head: nn.Module
    tokenizer: Optional[object]
    image_mean: Sequence[float]
    image_std: Sequence[float]
    hidden_size: int


def load_qwen(name_or_path: str = "Qwen/Qwen3.5-0.8B", dtype: str = "bfloat16",
              tiny: Optional[Dict] = None) -> QwenParts:
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    if tiny is not None:
        cfg = Qwen3_5Config(text_config=tiny["text"], vision_config=tiny["vision"],
                            **{k: v for k, v in tiny.items() if k not in ("text", "vision")})
        model = Qwen3_5ForConditionalGeneration(cfg)
        tok, mean, std = None, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        from transformers import AutoProcessor
        model = Qwen3_5ForConditionalGeneration.from_pretrained(name_or_path, dtype=getattr(torch, dtype))
        proc = AutoProcessor.from_pretrained(name_or_path)
        tok = proc.tokenizer
        mean, std = proc.image_processor.image_mean, proc.image_processor.image_std
    return QwenParts(visual=model.model.visual, language_model=model.model.language_model, lm_head=model.lm_head,
                     tokenizer=tok, image_mean=tuple(mean), image_std=tuple(std),
                     hidden_size=model.config.text_config.hidden_size)


# Projections of Qwen3.5's full-attention, Gated-DeltaNet and MLP blocks.
DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                        "gate_proj", "up_proj", "down_proj"]


def add_lora(language_model: nn.Module, r: int = 16, alpha: int = 32, dropout: float = 0.05,
             targets: Sequence[str] = DEFAULT_LORA_TARGETS) -> nn.Module:
    """Inject LoRA adapters (peft) into the language model only; the vision tower keeps its weights."""
    from peft import LoraConfig, inject_adapter_in_model
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=list(targets))
    return inject_adapter_in_model(cfg, language_model)


def lora_parameters(module: nn.Module) -> List[nn.Parameter]:
    return [p for n, p in module.named_parameters() if "lora_" in n]


def lora_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {n: p.detach().cpu() for n, p in module.named_parameters() if "lora_" in n}


class PromptBuilder:
    """Metadata as text (variables, units, levels, grid, issue time, history, leads) and template explanations.

    compact: one short line (variables grouped by level, lead range) instead of every name and unit (about 70
    instead of 215 tokens; the prompt dominates the LLM cost on a CPU)."""

    def __init__(self, tokenizer, channels: Sequence[str], units: Sequence[str], grid: str,
                 history_h: Sequence[int], leads_h: Sequence[int], compact: bool = False):
        self.tok = tokenizer
        if compact:  # e.g. "ERA5 5.625 deg: z, t, q, u, v at 500/700/850 hPa; t2m, msl, tcwv; history -24..0 h; ..."
            from .contract import split_name
            levels: Dict[str, List[str]] = {}
            for c in channels:
                v, p = split_name(c)
                levels.setdefault(v, []).append("" if p is None else str(p))
            groups: Dict[str, List[str]] = {}
            for v, ps in levels.items():
                groups.setdefault("/".join(ps), []).append(v)
            var = "; ".join(", ".join(vs) + (f" at {k} hPa" if k else "") for k, vs in groups.items())
            self.head = (f"ERA5 {grid}: {var}; history {min(history_h)}..{max(history_h)} h; "
                         f"forecast +{min(leads_h)}..+{max(leads_h)} h. Issue time: ")
            return
        var = ", ".join(f"{c} [{u}]" for c, u in zip(channels, units))
        self.head = (f"You are a numerical weather forecaster. Inputs: ERA5 analyses on a {grid} grid, "
                     f"variables {var}, history {', '.join(f'{h:+d} h' for h in history_h)}. "
                     f"Predict the fields at {', '.join(f'+{h} h' for h in leads_h)}. Issue time: ")

    def prompt_ids(self, issue: Sequence[str]) -> List[List[int]]:
        if self.tok is None:
            return [[] for _ in issue]
        return [self.tok(self.head + s + ". Weather tokens:", add_special_tokens=False)["input_ids"] for s in issue]

    QUESTION = (" Question: from the current state, what are the physical state and the fields of the forecast at "
                "+{h:03d} h? Forecast tokens:")
    ANSWER = "Answer at +{h:03d} h: "

    def question_ids(self, leads_h: Sequence[int]) -> List[List[int]]:
        """Lead questions (stage 03): the lead in text, written with three digits so all questions have one length."""
        if self.tok is None:
            return [[] for _ in leads_h]
        return [self.tok(self.QUESTION.format(h=int(h)), add_special_tokens=False)["input_ids"] for h in leads_h]

    def answer_prefix_ids(self, leads_h: Sequence[int]) -> List[List[int]]:
        if self.tok is None:
            return [[] for _ in leads_h]
        return [self.tok(self.ANSWER.format(h=int(h)), add_special_tokens=False)["input_ids"] for h in leads_h]

    def text_ids(self, texts: Sequence[str]) -> List[List[int]]:
        if self.tok is None:
            return [[] for _ in texts]
        eos = self.tok.eos_token_id
        return [self.tok(t, add_special_tokens=False)["input_ids"] + ([eos] if eos is not None else []) for t in texts]


def explain(z_now, z_next, names: Sequence[str], lat, lon, lead_h: int, include_now: bool = True) -> str:
    """Template explanation from verified diagnostics (patch-pooled, physical units, (n_diag, Hp, Wp) arrays):
    location and size of the strongest positive / negative patch of each diagnostic now (optional) and at +lead_h."""
    import numpy as np
    Hp, Wp = z_now.shape[-2:]
    la = np.asarray(lat).reshape(Hp, -1).mean(1)
    lo = np.asarray(lon).reshape(Wp, -1).mean(1)
    words = {"tadv850": ("warm advection", "cold advection", "K/day"),
             "qadv850": ("moist advection", "dry advection", "g/kg/day"),
             "qconv850": ("moisture convergence", "moisture divergence", "g/kg/day"),
             "thick500_850": ("thickest 850-500 hPa layer", "thinnest 850-500 hPa layer", "m")}
    where = lambda r, c: f"{abs(la[r]):.0f}{'N' if la[r] >= 0 else 'S'} {lo[c] % 360:.0f}E"  # noqa: E731
    parts = []
    for when, z in ((("now", z_now),) if include_now else ()) + ((f"at +{lead_h} h", z_next),):
        for i, n in enumerate(names):
            pos, neg, u = words[n]
            hi = np.unravel_index(np.argmax(z[i]), z[i].shape)
            lo_ = np.unravel_index(np.argmin(z[i]), z[i].shape)
            parts.append((f"{when}: " if include_now else "") + f"{pos} {z[i][hi]:.1f} {u} near {where(*hi)}, "
                         f"{neg} {z[i][lo_]:.1f} {u} near {where(*lo_)}")
    return "; ".join(parts) + "."
