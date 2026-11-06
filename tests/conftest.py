import copy
import os
import sys

import pytest
import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    """Tiny config on freshly generated synthetic pg-flow files with a prepared cache."""
    from scripts import make_synthetic, prepare_data
    root = tmp_path_factory.mktemp("data")
    make_synthetic.main(["--out", str(root / "syn"), "--years", "2015", "2016", "2017", "--days", "20", "--nlat", "32"])
    with open(os.path.join(ROOT, "configs", "tiny_synthetic.yaml")) as fh:
        c = yaml.safe_load(fh)
    c["data"].update(kai_dir=str(root / "syn" / "kai6h"), era5_data=str(root / "syn" / "era5_data"),
                     cache=str(root / "cache"))
    path = root / "cfg.yaml"
    path.write_text(yaml.safe_dump(c))
    prepare_data.main(["--config", str(path), "--diag-samples", "40"])
    c["_path"] = str(path)
    return c


@pytest.fixture
def cfg_copy(cfg):
    return copy.deepcopy(cfg)
