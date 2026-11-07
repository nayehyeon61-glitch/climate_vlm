import numpy as np
import pytest
import torch

from climate_vlm import build, data, io
from climate_vlm.vision import QwenVisionEncoder


@pytest.fixture(scope="module")
def batch(cfg):
    ds = build.dataset(cfg, "train")
    return torch.utils.data.default_collate([ds[0], ds[1]])


def test_patchify_matches_qwen_image_processor(cfg):
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    m = build.model(cfg)
    enc: QwenVisionEncoder = m.tokenizer.encoder
    img = torch.rand(1, 3, 64, 128)
    proc = Qwen2VLImageProcessor(patch_size=enc.p, temporal_patch_size=enc.tp, merge_size=enc.m,
                                 image_mean=[0.5] * 3, image_std=[0.5] * 3, min_pixels=32 * 32, max_pixels=10 ** 7)
    ref = proc(images=[img[0].permute(1, 2, 0).numpy()], do_resize=False, do_rescale=False, return_tensors="pt")
    mine = enc.patchify((img - 0.5) / 0.5)
    torch.testing.assert_close(mine, ref["pixel_values"].float(), atol=1e-5, rtol=1e-5)


def test_forward_shapes_and_no_text_leak(cfg, batch):
    m = build.model(cfg, lora=True)
    m.eval()
    L, C = len(cfg["data"]["leads_h"]), len(data.load_meta(cfg["data"]["cache"])["channels"])
    with torch.no_grad():
        a = m(batch["x_lr"], text_ids=[[1, 2, 3], [4, 5]])
        b = m(batch["x_lr"], text_ids=[[9, 9, 9, 9, 9], [7]])
    assert a["mean"].shape == (2, L, C, 32, 64) and a["scale"].shape == a["mean"].shape
    assert a["z_hat"].shape == (2, 1 + L, 4, 4, 8)
    assert a["text_logits"].shape[:2] == (2, 3)
    # explanation tokens come after the query tokens: the numeric forecast cannot see them
    torch.testing.assert_close(a["mean"], b["mean"])
    torch.testing.assert_close(a["z_hat"], b["z_hat"])


def test_stage2_gradient_reaches_resampler_through_frozen_llm(cfg, batch):
    m = build.model(cfg)
    params = m.set_stage("2")
    m.train()
    assert not m.D.training and not m.tokenizer.encoder.training  # frozen CorrDiff UNet: no dropout
    out = m(batch["x_lr"])
    out["mean"].pow(2).mean().backward()
    assert all(p.grad is None for p in m.llm.parameters())
    assert all(p.grad is None for p in m.D.parameters())
    assert all(p.grad is None for p in m.tokenizer.encoder.parameters())
    q = m.tokenizer.resampler.queries
    assert q.grad is not None and q.grad.abs().sum() > 0  # only reachable through the LLM
    assert m.fq.grad is not None and m.fq.grad.abs().sum() > 0
    assert all(p.requires_grad for p in params)


def test_stage3_trains_lora_only_inside_llm(cfg, batch):
    m = build.model(cfg, lora=True)
    m.set_stage("3")
    names = [n for n, p in m.llm.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names)
    m(batch["x_lr"])["mean"].mean().backward()
    # lora_B starts at zero, so the first gradient lands on lora_B
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.llm.named_parameters() if "lora_B" in n)


def test_downscaler_residual_sampling_and_spread(cfg, batch):
    d, s_hr, meta = build.downscaler(cfg)
    x = batch["x_lr"][:, -1]
    ens = d.sample(x, s_hr, members=2)
    assert ens.shape == (2, 2, len(meta["state_channels"]), 32, 64)
    maps = d.tokens(batch["x_lr"], s_hr, cfg["data"]["history_h"], members=2)
    assert set(maps) == {"lr", "mean", "tendency", "spread"}
    assert float(maps["tendency"][:, 0].abs().max()) == 0.0


def test_objective_runs_with_all_terms(cfg, batch):
    meta = data.load_meta(cfg["data"]["cache"])
    norm = data.Normalizer.load(build.paths(cfg)["stats"])
    m = build.model(cfg)
    obj = build.objective(cfg, meta, norm, "3")
    out = m(batch["x_lr"])
    loss, parts = obj(out, batch)
    assert torch.isfinite(loss) and {"forecast", "diag", "cons", "phys"} <= set(parts)
    loss.backward()


def test_pgflow_member_file_roundtrip(tmp_path):
    ch = ["z500", "t850", "t2m", "u10", "v10", "msl"]
    rng = np.random.default_rng(0)
    mean = rng.normal(size=(3, 4, 6, 8, 16)).astype(np.float32) * 10 + 280
    truth, x0 = mean + 1, mean[:, 0] - 1
    io.write_year(str(tmp_path), "t", 2017, ch, mean, truth, x0, [2, 10, 18], [24, 48, 72, 120])
    z = np.load(tmp_path / "t_ens_2017.npz")
    assert z["E_minus_x0"].shape == (3, 1, 4, 5, 8, 16)
    ens = z["E_minus_x0"].astype(np.float32) + z["X0"][:, None, None]  # how pg-flow rebuilds members
    np.testing.assert_allclose(ens[:, 0], mean[:, :, :5], atol=0.05)
    assert list(z["leads"]) == [24, 48, 72, 120] and list(z["ics"]) == [2, 10, 18]
