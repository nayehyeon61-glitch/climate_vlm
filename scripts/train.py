"""Three-stage training (plan section 03). Model selection by the validation FORECAST error.

    python scripts/train.py --config C --stage 1    --out runs/x                          # D_mu (CorrDiff regression)
    python scripts/train.py --config C --stage 1res --out runs/x --init runs/x/stage1.pt  # optional D_res (EDM)
    python scripts/train.py --config C --stage 2    --out runs/x --init runs/x/stage1.pt  # P, F, G; D, E, LLM frozen
    python scripts/train.py --config C --stage 3    --out runs/x --init runs/x/stage2.pt  # + LLM LoRA

Stage 2 keeps the LLM weights frozen but NOT the graph: the loss reaches P through the LLM forward
(plan p.3, "backward path"), so the LLM is never run under no_grad and its hidden states are never detached.
"""

import argparse
import itertools
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from climate_vlm import build, data  # noqa: E402
from climate_vlm.llm import PromptBuilder, explain  # noqa: E402
from climate_vlm.losses import downscale_loss  # noqa: E402
from climate_vlm.model import output_units  # noqa: E402
from climate_vlm.physics import area_weight  # noqa: E402


def log(out, msg):
    print(msg, flush=True)
    with open(os.path.join(out, "train.log"), "a") as fh:
        fh.write(msg + "\n")


def save(path, state, stage, cfg, **extra):
    torch.save({"state": state, "stage": stage, "config": cfg, **extra}, path + ".tmp")
    os.replace(path + ".tmp", path)


def text_batch(obj, batch, meta, lead_h):
    """Template explanations from the verified diagnostics of this batch (labels only)."""
    with torch.no_grad():
        z = obj.targets(batch["x_hr"], batch["y"]) * obj.ds.to(batch["y"]) + obj.dm.to(batch["y"])
    z = z.cpu().numpy()
    return [explain(z[b, 0], z[b, 1], obj.po.diags, meta["lat"], meta["lon"], lead_h) for b in range(len(z))]


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--config", required=True)
    a.add_argument("--stage", required=True, choices=["1", "1res", "2", "3"])
    a.add_argument("--out", required=True)
    a.add_argument("--init", help="checkpoint of the previous stage")
    a.add_argument("--steps", type=int)
    a.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = a.parse_args(argv)
    cfg = build.load_config(a.config)
    tc = cfg["train"][f"stage{a.stage}"]
    steps = a.steps or tc["steps"]
    dev = torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(cfg["train"].get("seed", 0))
    p = build.paths(cfg)
    norm = data.Normalizer.load(p["stats"])
    meta = data.load_meta(p["cache"])
    tr, va = build.dataset(cfg, "train"), build.dataset(cfg, "val")
    workers = cfg["train"].get("workers", 2)
    dl = DataLoader(tr, batch_size=tc["batch"], shuffle=True, num_workers=workers, drop_last=True,
                    persistent_workers=workers > 0)
    vdl = DataLoader(va, batch_size=tc["batch"], shuffle=False, num_workers=workers)
    n_val = cfg["train"].get("val_batches", 8)
    w_lat = area_weight(meta["lat"]).to(dev)
    amp = dev.type == "cuda" and cfg["train"].get("amp", True)
    ac = lambda: torch.autocast("cuda", dtype=torch.bfloat16) if amp else torch.autocast("cpu", enabled=False)  # noqa: E731
    log(a.out, f"stage {a.stage}: {len(tr)} train / {len(va)} val windows, {steps} steps, device {dev}")

    if a.stage in ("1", "1res"):
        D, s_hr, _ = build.downscaler(cfg, dev)
        if a.init:
            build.load_own(D, a.init, prefix="D.")
        if a.stage == "1res" and D.res is None:
            raise SystemExit("set model.downscaler.residual: true for stage 1res")
        for q in D.parameters():
            q.requires_grad_(False)
        net = D.mean if a.stage == "1" else D.res
        for q in net.parameters():
            q.requires_grad_(True)

        def train_mode():
            D.train()
            if a.stage == "1res":  # frozen regression mean, as physicsnemo CorrDiff: eval (no dropout)
                D.mean.eval()

        train_mode()

        def step_loss(b, train=True):
            x_lr = b["x_lr"].to(dev).flatten(0, 1)
            x_hr = b["x_hr"].to(dev).flatten(0, 1)
            if a.stage == "1":
                return downscale_loss(D.regression_loss(x_lr, x_hr, s_hr), w_lat)
            return downscale_loss(D.residual_loss(x_lr, x_hr, s_hr), w_lat)

        def val_score():
            D.eval()
            errs = []
            with torch.no_grad():
                for b in itertools.islice(vdl, n_val):
                    if a.stage == "1":
                        x_lr, x_hr = b["x_lr"].to(dev).flatten(0, 1), b["x_hr"].to(dev).flatten(0, 1)
                        errs.append(float(((D(x_lr, s_hr) - x_hr) ** 2 * w_lat.view(-1, 1)).mean()))
                    else:
                        torch.manual_seed(0)
                        errs.append(float(step_loss(b)))
            train_mode()
            return float(np.mean(errs))

        params = [q for q in D.parameters() if q.requires_grad]
        own = lambda: {f"D.{k}": v for k, v in D.state_dict().items()}  # noqa: E731
        extra = {}
    else:
        model = build.model(cfg, lora=a.stage == "3", device=dev)
        if a.init:
            build.load_own(model, a.init)
        elif a.stage == "2":
            log(a.out, "WARNING: stage 2 without --init: the downscaler is untrained")
        params = model.set_stage(a.stage)
        obj = build.objective(cfg, meta, norm, a.stage)
        use_prompt = cfg["model"].get("prompt", True) and model.qwen_tokenizer is not None
        pb = PromptBuilder(model.qwen_tokenizer, meta["state_channels"], output_units(meta["state_channels"]),
                           f"{180 / len(meta['lat']) * cfg['data']['factor']:g} deg", cfg["data"]["history_h"],
                           cfg["data"]["leads_h"], compact=cfg["model"].get("prompt") == "compact")
        lam_t = obj.lam["text"]
        log(a.out, f"loss weights {obj.lam}")

        def run(b):
            bb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            prompt = pb.prompt_ids(b["issue"]) if use_prompt else None
            text = None
            if lam_t and use_prompt:
                text = pb.text_ids(text_batch(obj, bb, meta, cfg["data"]["leads_h"][0]))
                n = max(len(t) for t in text)
                bb["text_ids"] = torch.tensor([t + [0] * (n - len(t)) for t in text], device=dev)
            with ac():
                out = model(bb["x_lr"], prompt, text)
            return out, bb

        def step_loss(b, train=True):
            out, bb = run(b)
            loss, parts = obj(out, bb)
            step_loss.parts = {**parts, "clip_frac": float(out["clip_frac"])}
            return loss

        def val_score():
            model.eval()
            errs = []
            with torch.no_grad():
                for b in itertools.islice(vdl, n_val):
                    out, bb = run(b)
                    errs.append(float(((out["mean"].float() - bb["y"]) ** 2 * w_lat.view(-1, 1)).mean()))
            model.train()
            return float(np.mean(errs))

        own = model.own_state_dict
        extra = {"lora": a.stage == "3"}
        n_train = sum(q.numel() for q in params)
        n_all = sum(q.numel() for q in model.parameters())
        log(a.out, f"trainable {n_train / 1e6:.2f} M of {n_all / 1e6:.1f} M parameters")

    opt = torch.optim.AdamW(params, lr=float(tc["lr"]), weight_decay=float(tc.get("weight_decay", 0.0)))
    warm = max(1, int(tc.get("warmup", 0.05) * steps))  # linear warm-up, then cosine decay (any number of steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    best, t0, it = float("inf"), time.time(), iter(())
    every = cfg["train"].get("val_every", 500)
    hist = []
    for step in range(1, steps + 1):
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl)
            b = next(it)
        loss = step_loss(b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg["train"].get("grad_clip", 1.0))
        opt.step()
        sched.step()
        if step % cfg["train"].get("log_every", 50) == 0 or step == 1:
            parts = getattr(step_loss, "parts", {})
            log(a.out, f"step {step} loss {loss.item():.4f} {json.dumps({k: round(v, 4) for k, v in parts.items()})} "
                       f"{time.time() - t0:.0f}s")
        if step % every == 0 or step == steps:
            v = val_score()
            hist.append({"step": step, "val": v})
            log(a.out, f"step {step} val {v:.4f}")
            if v < best:
                best = v
                save(os.path.join(a.out, f"stage{a.stage}.pt"), own(), a.stage, cfg, val=v, step=step, **extra)
    save(os.path.join(a.out, f"stage{a.stage}_last.pt"), own(), a.stage, cfg, step=steps, **extra)
    with open(os.path.join(a.out, f"stage{a.stage}_val.json"), "w") as fh:
        json.dump({"best": best, "history": hist}, fh)
    log(a.out, f"done: best val {best:.4f} -> {os.path.join(a.out, f'stage{a.stage}.pt')}")


if __name__ == "__main__":
    main()
