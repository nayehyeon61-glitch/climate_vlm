"""train.py --resume continues from the state written at the last validation."""

import copy

import yaml

from scripts import train


def test_resume(cfg, tmp_path):
    c = copy.deepcopy(cfg)
    c["train"]["val_every"] = 2
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({k: v for k, v in c.items() if k != "_path"}))
    out = str(tmp_path / "run")
    train.main(["--config", str(path), "--stage", "2", "--out", out, "--steps", "2"])
    train.main(["--config", str(path), "--stage", "2", "--out", out, "--steps", "4", "--resume"])
    log = open(f"{out}/train.log").read()
    assert "resumed from" in log and "at step 2" in log and "step 4 val" in log
