import json
import os

import numpy as np

from scripts import evaluate, train


def test_three_stages_and_evaluation(cfg, tmp_path):
    c, out = cfg["_path"], str(tmp_path / "run")
    train.main(["--config", c, "--stage", "1", "--out", out, "--steps", "3"])
    train.main(["--config", c, "--stage", "1res", "--out", out, "--init", f"{out}/stage1.pt", "--steps", "2"])
    train.main(["--config", c, "--stage", "2", "--out", out, "--init", f"{out}/stage1res.pt", "--steps", "2"])
    train.main(["--config", c, "--stage", "3", "--out", out, "--init", f"{out}/stage2.pt", "--steps", "2"])
    evaluate.main(["--config", c, "--ckpt", f"{out}/stage3.pt", "--out", f"{out}/eval", "--members", "2"])
    s = json.load(open(f"{out}/eval/scores.json"))
    assert set(s["rmse"]) >= {"z500", "t850", "t2m", "u10", "v10"}
    assert all(np.isfinite(v) for c in s["rmse"].values() for v in c.values())
    z = np.load(f"{out}/eval/climate_vlm_ens_2017.npz")
    assert z["E_minus_x0"].shape[1:4] == (2, 4, 5)
    assert os.path.exists(f"{out}/stage3_last.pt")
