"""Stage 02, step 4: Laya trained on the climate expert-selection task (preprocessing guide sec. 11).

    python scripts/stage02_laya.py dry-run --config configs/stage02_laya_router.yaml
    python scripts/stage02_laya.py train   --config configs/stage02_laya_router.yaml
    python scripts/stage02_laya.py predict --config configs/stage02_laya_router.yaml --model base|tuned

dry-run  laya's own check of router_train.jsonl (items, skipped questions and why, optimizer-update budget) with the
         tokenizer and token budgets of the base checkpoint, without loading its weights.
train    laya.train.finetune on router_train.jsonl (split B) from router.laya.base with the encoder frozen, so only
         the decision head learns (guide 11-2, CPU). laya's own calibration slice is off (calib_frac 0): the
         probabilities are calibrated on split C by stage02_router, apart from head training (guide 7);
         router_calib.jsonl is laya's held-out eval set (accuracy before / after, nothing is fitted on it).
predict  decisions of one checkpoint for splits B, C, D: probabilities of the three experts and, for the base
         checkpoint, the option states its head scores (input of the cached-representation head). Wall time per
         decision and the peak RSS of the process are recorded, so run each model in its own process.

-> <out>/laya_tuned/ (checkpoint, train_summary.json), <out>/laya_<model>_<split>.npz
"""

import argparse
import json
import os
import resource
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from climate_vlm import build  # noqa: E402
from scripts.stage02_common import EXPERTS  # noqa: E402

JSONL = {"B": "router_train", "C": "router_calib", "D": "router_test"}


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # Linux: KiB


def train_config(lc, out):
    from laya.train import TrainConfig
    t = dict(lc.get("train", {}))
    t.setdefault("freeze_encoder", True)
    t.setdefault("calib_frac", 0.0)
    t["eval_data"] = os.path.join(out, f"{JSONL['C']}.jsonl")
    return TrainConfig(**t)


def read_rows(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def predict(agent, rows, batch, capture):
    """Probabilities (N, 3) in EXPERTS order and, if ``capture``, the head's option states (N, 3, d)."""
    q = rows[0]["questions"]
    if any(r["questions"] != q for r in rows):
        raise ValueError("all rows must ask the same question")
    cap, hook = [], None
    if capture:
        hook = agent.model.scorer.register_forward_hook(lambda m, i, o: cap.append(i[0].detach().float()[:, :3]))
    probs, t0 = [], time.time()
    for i in range(0, len(rows), batch):
        res = agent.predict_batch([r["state"] for r in rows[i:i + batch]], q, batch_size=batch)
        probs += [[r["answers"]["expert"]["probabilities"][k] for k in EXPERTS] for r in res]
        if (i // batch) % 10 == 0:
            print(f"  {i + len(res)}/{len(rows)} {time.time() - t0:.0f} s", flush=True)
    sec = (time.time() - t0) / len(rows)
    reps = None
    if hook is not None:
        hook.remove()
        reps = torch.cat(cap).numpy()
        if len(reps) != len(rows):
            raise RuntimeError(f"captured {len(reps)} option states for {len(rows)} rows")
    return np.array(probs, np.float32), reps, sec


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("command", choices=["dry-run", "train", "predict"])
    a.add_argument("--config", required=True)
    a.add_argument("--model", choices=["base", "tuned"], default="base")
    a.add_argument("--splits", nargs="*", default=["B", "C", "D"])
    a.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 4)))
    a = a.parse_args(argv)
    torch.set_num_threads(a.threads)
    cfg = build.load_config(a.config)
    out, lc = cfg["out"], cfg["router"]["laya"]
    from laya import train as lt
    base_dir = lt.resolve_checkpoint_dir(lc["base"])
    tuned_dir = os.path.join(out, "laya_tuned")
    train_file = os.path.join(out, f"{JSONL['B']}.jsonl")

    if a.command == "dry-run":
        rep = lt.dry_run(train_file, base_dir, train_config(lc, out))
        print(json.dumps(rep, indent=1, default=str))
        print(lt.update_budget_message(rep.get("train_items", 0) or 1, train_config(lc, out)))
        return rep

    if a.command == "train":
        tc = train_config(lc, out)
        t0 = time.time()
        summary = lt.finetune(train_file, base_dir, tuned_dir, config=tc, device="cpu")
        rec = {"base": lc["base"], "base_dir": base_dir, "train_sec": time.time() - t0, "peak_rss_gb": peak_rss_gb(),
               "threads": a.threads, "config": {k: v for k, v in vars(tc).items()}, "summary": summary}
        with open(os.path.join(tuned_dir, "train_summary.json"), "w") as fh:
            json.dump(rec, fh, indent=1, default=str)
        print(f"laya fine-tune: {rec['train_sec'] / 60:.1f} min, peak RSS {rec['peak_rss_gb']:.2f} GB -> {tuned_dir}",
              flush=True)
        return rec

    import laya
    t0 = time.time()
    agent = laya.load(base_dir if a.model == "base" else tuned_dir, device="cpu")
    load_sec = time.time() - t0
    for s in a.splits:
        rows = read_rows(os.path.join(out, f"{JSONL[s]}.jsonl"))
        print(f"laya {a.model} on split {s}: {len(rows)} decisions", flush=True)
        probs, reps, sec = predict(agent, rows, int(lc.get("batch", 32)), capture=a.model == "base")
        res = {"probs": probs, "sec_per_decision": sec, "rss_gb": peak_rss_gb(), "load_sec": load_sec,
               "threads": a.threads, "experts": np.array(EXPERTS)}
        if reps is not None:
            res["reps"] = reps.astype(np.float16)
        np.savez(os.path.join(out, f"laya_{a.model}_{s}.npz"), **res)
        acc = float(np.mean(probs.argmax(1) == np.array([EXPERTS.index(r["expected"]["expert"]) for r in rows])))
        print(f"laya {a.model} split {s}: accuracy {acc:.3f}, {sec * 1e3:.0f} ms per decision, "
              f"peak RSS {res['rss_gb']:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
