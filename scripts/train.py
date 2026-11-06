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


def text_batch(obj, batch, meta, lead_h, include_now=True):
    """Template explanations from the verified diagnostics of this batch (labels only). lead_h: one lead, or one per
    sample when batch["y"] holds only the asked lead of each sample (lead questions)."""
    with torch.no_grad():
        z = obj.targets(batch["x_hr"], batch["y"]) * obj.ds.to(batch["y"]) + obj.dm.to(batch["y"])
    z = z.cpu().numpy()
    leads = list(lead_h) if isinstance(lead_h, (list, tuple)) else [lead_h] * len(z)
    return [explain(z[b, 0], z[b, 1], obj.po.diags, meta["lat"], meta["lon"], leads[b], include_now)
            for b in range(len(z))]


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--config", required=True)
    a.add_argument("--stage", required=True, choices=["1", "1res", "2", "3"])
    a.add_argument("--out", required=True)
    a.add_argument("--init", help="checkpoint of the previous stage")
    a.add_argument("--steps", type=int)
    a.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a.add_argument("--resume", action="store_true",
                   help="continue from <out>/stage<N>_resume.pt (written at every validation) if it exists")
    a = a.parse_args(argv)
    cfg = build.load_config(a.config)
    tc = cfg["train"][f"stage{a.stage}"]
    steps = a.steps or tc["steps"]
    mb = tc.get("micro_batch") or tc["batch"]
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
        leads_h = cfg["data"]["leads_h"]
        qcfg = cfg["model"].get("question", {})
        g_lead = torch.Generator().manual_seed(cfg["train"].get("seed", 0))
        if model.ask:
            log(a.out, f"lead questions: one asked lead per sequence from {leads_h}, lead embedding "
                       f"{'on' if model.ask_lead_emb else 'off (lead only in the question text)'}")

        def run(b, lead_index=None):
            """lead_index: None = a random asked lead per sample (training), k = ask leads_h[k] of every sample."""
            bb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            prompt = pb.prompt_ids(b["issue"]) if use_prompt else None
            question, ask_h, text_leads = None, None, leads_h[0]
            if model.ask:
                B = bb["y"].shape[0]
                j = (torch.randint(0, len(leads_h), (B,), generator=g_lead) if lead_index is None
                     else torch.full((B,), lead_index, dtype=torch.long))
                bb["y"] = bb["y"][torch.arange(B), j.to(bb["y"].device)][:, None]
                text_leads = [leads_h[k] for k in j.tolist()]
                ask_h = torch.tensor(text_leads, device=dev)
                question = pb.question_ids(text_leads)
            text = None
            if lam_t and use_prompt:
                texts = text_batch(obj, bb, meta, text_leads, include_now=qcfg.get("answer_now", not model.ask))
                if model.ask:
                    texts = [pb.ANSWER.format(h=h) + t for h, t in zip(text_leads, texts)]
                text = pb.text_ids(texts)
                n = max(len(t) for t in text)
                bb["text_ids"] = torch.tensor([t + [0] * (n - len(t)) for t in text], device=dev)
            with ac():
                out = model(bb["x_lr"], prompt, text, question_ids=question, ask_h=ask_h)
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
                    for k in range(len(leads_h)) if model.ask else [None]:  # lead questions: every lead
                        out, bb = run(b, k)
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
    hist, start = [], 1
    rpath = os.path.join(a.out, f"stage{a.stage}_resume.pt")
    if a.resume and os.path.exists(rpath):
        r = torch.load(rpath, map_location="cpu", weights_only=False)
        sd = r["state"]
        target = D if a.stage in ("1", "1res") else model
        if a.stage in ("1", "1res"):
            sd = {k[2:]: v for k, v in sd.items() if k.startswith("D.")}
        target.load_state_dict(sd, strict=False)
        opt.load_state_dict(r["opt"])
        sched.load_state_dict(r["sched"])
        best, hist, start = r["best"], r["hist"], r["step"] + 1
        torch.set_rng_state(r["rng"])
        log(a.out, f"resumed from {rpath} at step {r['step']} (best val {best:.4f})")
    for step in range(start, steps + 1):
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl)
            b = next(it)
        opt.zero_grad(set_to_none=True)
        n_b, loss = len(b["t"]), 0.0
        for i in range(0, n_b, mb):  # micro-batches: same update as one batch, less memory (long answer texts)
            part = {k: v[i:i + mb] for k, v in b.items()}
            lp = step_loss(part) * (len(part["t"]) / n_b)
            lp.backward()
            loss = loss + lp.detach()
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
            torch.save({"state": own(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                        "best": best, "hist": hist, "rng": torch.get_rng_state()}, rpath + ".tmp")
            os.replace(rpath + ".tmp", rpath)
    save(os.path.join(a.out, f"stage{a.stage}_last.pt"), own(), a.stage, cfg, step=steps, **extra)
    with open(os.path.join(a.out, f"stage{a.stage}_val.json"), "w") as fh:
        json.dump({"best": best, "history": hist}, fh)
    log(a.out, f"done: best val {best:.4f} -> {os.path.join(a.out, f'stage{a.stage}.pt')}")


if __name__ == "__main__":
    main()
