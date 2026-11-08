"""Stage 03 (lead questions in text): shapes, no answer leak, the lead through the text only, generation, scripts."""

import copy
import os

import pytest
import torch
import yaml

from climate_vlm import build, data
from scripts import evaluate, stage03_answers, train


def ask_cfg(cfg, lead_embedding=True):
    c = copy.deepcopy(cfg)
    c["model"]["question"] = {"enabled": True, "lead_embedding": lead_embedding}
    return c


@pytest.fixture(scope="module")
def batch(cfg):
    ds = build.dataset(cfg, "train")
    return torch.utils.data.default_collate([ds[0], ds[1]])


def test_one_asked_lead_and_no_answer_leak(cfg, batch):
    m = build.model(ask_cfg(cfg), lora=True)
    m.eval()
    C = len(data.load_meta(cfg["data"]["cache"])["channels"])
    q, h = [[5, 6, 7], [5, 6, 8]], torch.tensor([24, 72])
    with torch.no_grad():
        a = m(batch["x_lr"], question_ids=q, ask_h=h, text_ids=[[1, 2, 3], [4, 5]])
        b = m(batch["x_lr"], question_ids=q, ask_h=h, text_ids=[[9, 9, 9, 9], [7]])
    assert a["mean"].shape == (2, 1, C, 32, 64) and a["z_hat"].shape == (2, 2, 4, 4, 8)
    assert a["text_logits"].shape[:2] == (2, 3)
    torch.testing.assert_close(a["mean"], b["mean"])  # the answer sits after the queries
    torch.testing.assert_close(a["z_hat"], b["z_hat"])


def test_lead_only_through_the_question_text(cfg, batch):
    m = build.model(ask_cfg(cfg, lead_embedding=False), lora=True)
    for p in m.parameters():  # make every path non-trivial (F's last conv starts at zero)
        p.data += 0.01 * torch.randn_like(p)
    m.eval()
    with torch.no_grad():
        a = m(batch["x_lr"], question_ids=[[5, 6, 7]] * 2, ask_h=torch.tensor([24, 24]))
        b = m(batch["x_lr"], question_ids=[[5, 6, 7]] * 2, ask_h=torch.tensor([120, 120]))
        c = m(batch["x_lr"], question_ids=[[5, 6, 9]] * 2, ask_h=torch.tensor([24, 24]))
    torch.testing.assert_close(a["mean"], b["mean"])  # the hour itself is not used ...
    assert not torch.allclose(a["mean"], c["mean"])  # ... the question text is


def test_generate_runs(cfg, batch):
    m = build.model(ask_cfg(cfg), lora=True)
    m.eval()
    out = m.generate(batch["x_lr"], None, [[5, 6, 7]] * 2, [[3, 4]] * 2, max_new_tokens=5, eos_id=None)
    assert len(out) == 2 and all(len(o) == 5 for o in out)


def test_stage03_scripts(cfg, tmp_path):
    c = ask_cfg(cfg)
    c["data"]["leads_h"] = [24, 72]
    c["train"]["stage3"] = {**c["train"].get("stage3", {}), "loss": {"diag": 0.1, "text": 0.1}}
    path = tmp_path / "ask.yaml"
    path.write_text(yaml.safe_dump({k: v for k, v in c.items() if k != "_path"}))
    run = str(tmp_path / "run")
    train.main(["--config", str(path), "--stage", "3", "--out", run, "--steps", "2"])
    evaluate.main(["--config", str(path), "--ckpt", f"{run}/stage3.pt", "--out", f"{run}/eval"])
    assert os.path.exists(f"{run}/eval/scores.json")


def test_answer_parser_and_scores():
    t = ("warm advection 6.1 K/day near 38N 125E, cold advection -7.0 K/day near 45S 140E; "
         "thickest 850-500 hPa layer 5600.2 m near 10N 20E")
    p = stage03_answers.parse(t)
    assert p["cold advection"] == (-7.0, -45.0, 140.0) and "thickest 850-500 hPa layer" in p
    s = stage03_answers.compare({"warm advection": (5.1, 38.0, 125.0)}, p)
    assert s["stated"] == pytest.approx(1 / 3) and s["value_abs_err"] == pytest.approx(1.0) and s["dist_km"] == 0.0
