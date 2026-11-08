"""Stage 03: generated answers to the lead questions, scored against the verified diagnostics of the truth.

    python scripts/stage03_answers.py --config configs/stage03_ask.yaml --ckpt runs/stage03/ask/stage3.pt \
        --out runs/stage03/ask/answers --n-ics 24

For n evenly spaced test ICs (ClimODE 2020 set) and every asked lead, the model generates its answer greedily after
"Answer at +hhh h:". Each answer is parsed into the strongest positive / negative patch of every diagnostic (value,
location) and compared with the same description of
    truth        the diagnostics of the verifying analysis (the training target of the answers)
and, as references,
    numeric      the model's own numeric forecast of that lead, described by the same template
    persistence  the issue-time state described as if it were the forecast
Scores per lead: share of items the answer states, mean absolute value error and mean great-circle distance (km) of
the stated location. -> <out>/answers.json (scores + every text), <out>/examples.md
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from climate_vlm import build, data  # noqa: E402
from climate_vlm.llm import PromptBuilder, explain  # noqa: E402
from climate_vlm.model import output_units  # noqa: E402

ITEM = re.compile(r"([A-Za-z0-9\- ]+?) (-?\d+(?:\.\d+)?) (K/day|g/kg/day|m) near (\d+)([NS]) (\d+)E")


def parse(text):
    """{phrase: (value, lat, lon)} of every "<phrase> <value> <unit> near <lat>N|S <lon>E" item."""
    out = {}
    for m in ITEM.finditer(text):
        phrase = m.group(1).strip().split(": ")[-1].replace("at +", "").strip()
        phrase = re.sub(r"^(?:\d+ h: )", "", phrase)
        lat = float(m.group(4)) * (1 if m.group(5) == "N" else -1)
        out.setdefault(phrase, (float(m.group(2)), lat, float(m.group(6))))
    return out


def gc_km(la1, lo1, la2, lo2):
    p1, p2, dl = np.deg2rad(la1), np.deg2rad(la2), np.deg2rad(lo2 - lo1)
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return float(6371.0 * np.arccos(np.clip(c, -1, 1)))


def compare(pred, truth):
    """Share of truth items stated, mean |value error|, mean distance (km) over the stated items."""
    hit = [k for k in truth if k in pred]
    if not hit:
        return {"stated": 0.0, "value_abs_err": None, "dist_km": None}
    return {"stated": len(hit) / len(truth),
            "value_abs_err": float(np.mean([abs(pred[k][0] - truth[k][0]) for k in hit])),
            "dist_km": float(np.mean([gc_km(pred[k][1], pred[k][2], truth[k][1], truth[k][2]) for k in hit]))}


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--config", required=True)
    a.add_argument("--ckpt", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--n-ics", type=int, default=24)
    a.add_argument("--batch", type=int, default=4)
    a.add_argument("--max-new-tokens", type=int, default=200)
    a = a.parse_args(argv)
    cfg = build.load_config(a.config)
    p = build.paths(cfg)
    meta, norm = data.load_meta(p["cache"]), data.Normalizer.load(p["stats"])
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = build.model(cfg, lora=bool(ck.get("lora", False)))
    build.load_own(model, a.ckpt)
    model.set_stage("2")
    model.eval()
    if not model.ask:
        raise SystemExit("the config has no lead questions (model.question.enabled)")
    obj = build.objective(cfg, meta, norm, "3")
    pb = PromptBuilder(model.qwen_tokenizer, meta["state_channels"], output_units(meta["state_channels"]),
                       f"{180 / len(meta['lat']) * cfg['data']['factor']:g} deg", cfg["data"]["history_h"],
                       cfg["data"]["leads_h"], compact=cfg["model"].get("prompt") == "compact")
    tok, leads = model.qwen_tokenizer, cfg["data"]["leads_h"]
    ds = build.dataset(cfg, "test", ics="climode")
    pick = np.unique(np.linspace(0, len(ds) - 1, min(a.n_ics, len(ds))).round().astype(int))
    ds.index = [ds.index[i] for i in pick]
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=0)
    names, lat, lon = obj.po.diags, meta["lat"], meta["lon"]
    den = lambda z: (z * obj.ds + obj.dm).numpy()  # noqa: E731
    rows = []
    for b in dl:
        B = len(b["t"])
        prompt = pb.prompt_ids(b["issue"])
        with torch.no_grad():
            z_true = obj.targets(b["x_hr"], b["y"])  # (B, 1 + L, n_diag, Hp, Wp) normalised
        for j, h in enumerate(leads):
            q = pb.question_ids([h] * B)
            with torch.no_grad():
                out = model(b["x_lr"], prompt, question_ids=q, ask_h=torch.full((B,), h))
                z_num = obj.zdiag(obj.po, out["mean"][:, 0].float())
            gen = model.generate(b["x_lr"], prompt, q, pb.answer_prefix_ids([h] * B), a.max_new_tokens,
                                 tok.eos_token_id)
            for i in range(B):
                zt, zn, z0 = den(z_true[i, 1 + j]), den(z_num[i]), den(z_true[i, 0])
                texts = {"answer": tok.decode(gen[i], skip_special_tokens=True),
                         "truth": explain(z0, zt, names, lat, lon, h, include_now=False),
                         "numeric": explain(z0, zn, names, lat, lon, h, include_now=False),
                         "persistence": explain(z0, z0, names, lat, lon, h, include_now=False)}
                ref = parse(texts["truth"])
                rows.append({"issue": b["issue"][i], "lead_h": h, "texts": texts,
                             "scores": {k: compare(parse(v), ref) for k, v in texts.items() if k != "truth"}})
            print(f"{b['issue'][0]} +{h} h: {texts['answer'][:160]}", flush=True)
    summary = {}
    for h in leads:
        r = [x for x in rows if x["lead_h"] == h]
        summary[str(h)] = {}
        for k in ("answer", "numeric", "persistence"):
            s = [x["scores"][k] for x in r]
            f = lambda m: (float(np.mean([v[m] for v in s if v[m] is not None]))  # noqa: E731
                           if any(v[m] is not None for v in s) else None)
            summary[str(h)][k] = {"stated": float(np.mean([v["stated"] for v in s])), "value_abs_err": f("value_abs_err"),
                                  "dist_km": f("dist_km")}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "answers.json"), "w") as fh:
        json.dump({"checkpoint": a.ckpt, "n_ics": len(pick), "leads": leads, "summary": summary, "rows": rows}, fh,
                  indent=1)
    L = ["| lead | text | items stated | mean abs value error | mean distance km |", "|---|---|---:|---:|---:|"]
    for h, d in summary.items():
        for k, v in d.items():
            fv = lambda x, f: "-" if x is None else f.format(x)  # noqa: E731
            L.append(f"| +{h} h | {k} | {v['stated']:.2f} | {fv(v['value_abs_err'], '{:.2f}')} | "
                     f"{fv(v['dist_km'], '{:.0f}')} |")
    L += ["", "## Examples", ""]
    for x in rows[:2 * len(leads)]:
        L += [f"**{x['issue']} +{x['lead_h']} h**", "", f"- answer: {x['texts']['answer']}",
              f"- truth: {x['texts']['truth']}", f"- numeric: {x['texts']['numeric']}", ""]
    with open(os.path.join(a.out, "examples.md"), "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L[:2 + 3 * len(leads)]))
    return summary


if __name__ == "__main__":
    main()
