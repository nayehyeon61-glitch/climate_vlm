"""Stage 06 (Laya typed questions after the forecast) end to end on the synthetic files; vega is stubbed."""

import json
import os

import numpy as np
import yaml

from scripts import stage02_forecasts, stage06_questions as s6, train
from tests.test_wb2_compare import _truth_files


class StubVega:
    """Counts reads, so the cache can be checked."""

    def __init__(self):
        self.reads = {"f": 0, "e": 0}
        self.engine = self

    def features(self, state, question):
        self.reads["f"] += 1
        return np.zeros(4)

    def decide(self, state, questions, alpha="0.1", risk=None, views=1):
        self.reads["e"] += 1
        return {"answers": {q: {"probs": {"x": 1.0}} for q in questions}}


class StubAgent:
    """laya.Agent(vegaml=True)'s surface that stage06_questions uses, answering at random."""

    def __init__(self):
        self.calibration, self.vega, self.rng = None, StubVega(), np.random.default_rng(0)

    def predict(self, state, questions):
        self.vega.decide(state, questions)
        out = {}
        for qid, q in questions.items():
            labs = s6.labels_for(q)
            p = self.rng.dirichlet(np.ones(len(labs)))
            out[qid] = {"probabilities": dict(zip(labs, p.tolist())), "prediction_set": [labs[int(p.argmax())]],
                        "abstain": False, "route": "auto" if p.max() > 0.5 else "review", "readout": "engine"}
        return {"answers": out}

    def calibrate(self, rows, questions):
        assert rows and all(set(r) == {"state", "gold"} for r in rows)
        self.calibration = "cal"
        return {"n_rows": len(rows)}

    def fit(self, rows, questions, calibration_rows=None):
        assert calibration_rows and rows[0]["state"] != calibration_rows[0]["state"]
        self.calibration = "fit"
        return {"heads": {q: {"used_by_auto": False} for q in questions}}

    def save_calibration(self, path):
        os.makedirs(path, exist_ok=True)


def test_fc_feature_index_matches_the_state():
    q = {"issue": np.array([24 * 365 * 50]), "leads": [24, 72], "subs": ["NW", "NE", "SW", "SE"],
         "T": np.array([[1.0, 2, 3, 4, 5]]), "Z": np.array([5500.0]), "adv": np.array([0.4]),
         "dT24": np.array([-1.0]), "dZ24": np.array([-20.0]),
         "fdT": np.array([[[-0.5, -1, -2, 0.3, 0.1], [-3.0, -4, -5, -2, -1]]]), "fdZ": np.array([[-10.0, -60.0]]),
         "tdT": np.zeros((1, 2, 5)), "tdZ": np.zeros((1, 2))}
    th = {24: {"tau_K": 1.0, "z500_terciles_m": [-15, 15]}, 72: {"tau_K": 2.5, "z500_terciles_m": [-40, 40]}}
    r = s6.rows_of(q, th, "Test")[0]
    x = np.array(r["x"])
    assert x[s6.fc_feature_idx("drop_72", [24, 72], 4)].tolist() == [-3.0]
    assert x[s6.fc_feature_idx("z500_24", [24, 72], 4)].tolist() == [-10.0]
    assert x[s6.fc_feature_idx("coldest_72", [24, 72], 4)].tolist() == [-4.0, -5.0, -2.0, -1.0]
    assert r["rule"] == {"drop_24": "false", "z500_24": "1", "coldest_24": "NE",
                         "drop_72": "true", "z500_72": "0", "coldest_72": "NE"}
    assert "+72 h: T850 -3.0 K (NW -4.0, NE -5.0, SW -2.0, SE -1.0), Z500 -60 m" in r["state"]
    assert r["gold"]["drop_72"] == "false" and r["gold"]["z500_72"] == "1"  # no change: neither drop nor a tercile


def test_read_cache_serves_subsets_from_one_pass(tmp_path):
    agent = StubAgent()
    sheet = {"a": {"type": "boolean"}, "b": {"type": "score"}}
    s6.install_cache(agent, s6.ReadCache(str(tmp_path / "c.sqlite")), sheet)
    agent.vega.engine.decide("s", sheet)
    r = agent.vega.engine.decide("s", {"b": sheet["b"]})
    assert list(r["answers"]) == ["b"]
    agent.vega.features("s", sheet["a"])
    agent.vega.features("s", sheet["a"])
    assert agent.vega.reads == {"f": 1, "e": 1}  # one engine pass and one feature read, the rest from the cache
    assert s6.ReadCache(str(tmp_path / "c.sqlite")).db.execute("select count(*) from c").fetchone()[0] == 2


def test_stage06_pipeline(cfg, tmp_path):
    _truth_files(cfg)
    c, run = cfg["_path"], str(tmp_path / "run")
    train.main(["--config", c, "--stage", "1", "--out", run, "--steps", "2"])
    train.main(["--config", c, "--stage", "2", "--out", run, "--init", f"{run}/stage1.pt", "--steps", "1"])
    out = str(tmp_path / "s06")
    per = {"router": {"split": "B", "years": [2016], "hours": [0, 6, 12, 18], "stride": 1, "margin_h": 0},
           "calib": {"split": "C", "years": [2017], "dates": ["2017-01-01", "2017-01-10"], "hours": [0, 12],
                     "stride": 1, "margin_h": 12},
           "test": {"split": "D", "years": [2017], "dates": ["2017-01-11", "2017-01-20"], "hours": [0, 12],
                    "stride": 1, "margin_h": 12}}
    s = {"base": {"config": c, "ckpt": f"{run}/stage2.pt"}, "out": out, "docs": str(tmp_path / "docs"),
         "leads": [24], "periods": per, "sample": {"router": 1, "calib": 1, "test": 1},
         "domain": {"name": "Test", "key": "T", "lat": [0, 90], "lon": [0, 180], "sub": [2, 2]}}
    sc = tmp_path / "s06.yaml"
    sc.write_text(yaml.safe_dump(s))
    stage02_forecasts.main(["--config", str(sc), "--batch", "4"])
    res = s6.main(["--config", str(sc), "rows"])
    assert set(res["questions"]) == {"drop_24", "z500_24", "coldest_24"}
    rows = res["rows"]["D"]
    assert rows and all("nan" not in r["state"].lower() and len(r["x"]) == 2 + 5 + 4 + 6 for r in rows)
    drop = np.mean([r["gold"]["drop_24"] == "true" for r in res["rows"]["B"]])
    assert drop <= 0.5  # tau leaves at most about 20 % of the B cases (at least 0.1 K on synthetic fields)
    s6.main(["--config", str(sc), "baselines"])
    agent = StubAgent()
    rep = s6.cmd_laya(s6.build.load_config(str(sc)), type("A", (), {"arms": ["zero", "cal", "fit"]})(), agent=agent)
    assert rep["rows"]["D"] == len(rows) and "fit" in rep
    out_r = s6.main(["--config", str(sc), "report"])
    assert set(out_r["scores"]) == set(s6.METHODS)
    for m in s6.METHODS:
        for v in out_r["scores"][m].values():
            assert 0 <= v["hit"] <= 1 and v["brier"] >= 0
    assert out_r["scores"]["laya_zero"]["drop_24"]["set_size"] == 1.0
    assert abs(out_r["scores"]["clim"]["drop_24"]["bss"]) < 1e-12
    assert os.path.exists(tmp_path / "docs" / "table.md")
    json.dumps(out_r)
