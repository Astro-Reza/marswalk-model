"""Tests for the GUI-free logic in marswalk_gui.core (no Qt needed)."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from marswalk_gui import core
from marswalk_gui.demo import MARKER, make_demo

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    data, model = make_demo(out)
    return out, data, model


def test_features_match_train_script():
    tree = ast.parse((ROOT / "scripts" / "train.py").read_text())
    feats = next(ast.literal_eval(n.value) for n in tree.body
                 if isinstance(n, ast.Assign) and n.targets[0].id == "FEATURES")
    assert feats == core.FEATURES
    assert set(core.FEATURE_RANGES) == set(core.FEATURES)


def test_demo_is_marked_synthetic(demo):
    out, data, _ = demo
    assert (out / MARKER).exists()
    assert core.load_dataset(str(data))["walker"].eq("synthetic-demo").all()


def test_dataset_summary_and_filters(demo):
    df = core.load_dataset(str(demo[1]))
    s = core.dataset_summary(df)
    assert s["rows"] == len(df) and s["maps"] == df["map_id"].nunique()
    assert s["mars_rows"] + s["earth_rows"] == s["ok_rows"]
    assert 0 <= s["fall_rate"] <= 1
    assert core.apply_filters(df, hide_falls=True)["fell"].sum() == 0
    assert (core.apply_filters(df, planet="mars")["gravity"] < 5).all()
    assert (core.apply_filters(df, planet="earth")["gravity"] >= 5).all()
    bad = pd.concat([df, pd.DataFrame([{"map_id": "x", "status": "error: boom", "fell": True}])], ignore_index=True)
    assert len(core.apply_filters(bad, only_ok=True)) == len(df)
    assert "cot_net_J_per_kg_m" in core.numeric_columns(df)


def test_load_dataset_empty_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        core.load_dataset(str(tmp_path))


def test_load_run_and_compare(demo):
    run = core.load_run(demo[2])
    assert set(run.models) == {"cot", "fall", "mos"} and run.lookup and run.split["test"]
    table = core.compare_table([run])
    assert set(table["target"]) == {"cot", "fall", "mos"}
    assert table.loc[table.target == "cot", "r2_test"].iloc[0] > 0.5
    with pytest.raises(FileNotFoundError):
        core.load_run(demo[1])  # a data folder has no metrics.json


def test_predictions(demo):
    df = core.load_dataset(str(demo[1]))
    run = core.load_run(demo[2])
    v = {f: lo_hi_d[2] for f, lo_hi_d in core.FEATURE_RANGES.items()}
    p = core.predict_features(run, v)
    assert 0 <= p["fall"] <= 1 and p["cot"] > 0
    res = core.test_predictions(run, df, "cot", "test")
    assert len(res) > 0 and set(res.map_id) <= set(run.split["test"])
    assert np.corrcoef(res.actual, res.predicted)[0, 1] > 0.9


def test_overfit_note():
    assert "no test rows" in core.overfit_note({"test": {}, "train": {}})
    note = core.overfit_note({"test": {"n": 100, "r2": 0.5}, "train": {"r2": 0.95}})
    assert "r2 drops" in note
    assert "only 10 test rows" in core.overfit_note({"test": {"n": 10, "r2": 0.9}, "train": {"r2": 0.9}})


def test_check_walker_file(tmp_path):
    check = {"walker": "kinesis", "policy": "p", "verdict": "v", "minetti_reference": {"-0.10": 1.0, "+0.00": 2.5, "+0.10": 4.0},
             "stages": {"1_flat_earth": {"runs": 3, "walked": 3, "fall_rate": 0.0, "mean_speed_mps": 1.2},
                        "4_sweep_earth": {"runs": 9, "walked": 9, "fall_rate": 0.0,
                                          "cot_gross_by_grade": {"-0.10": 1.1, "+0.00": 2.4, "+0.10": 4.2},
                                          "pearson_r_vs_minetti": 0.99}}}
    f = tmp_path / "check.json"
    f.write_text(json.dumps(check))
    c = core.load_check(f)
    assert list(core.check_stage_table(c)["stage"]) == ["1_flat_earth", "4_sweep_earth"]
    curves = core.check_cot_curves(c)
    assert curves["4_sweep_earth"][0] == [-0.1, 0.0, 0.1] and "minetti" in curves
    f.write_text("{}")
    with pytest.raises(ValueError):
        core.load_check(f)


def test_minetti_matches_check_walker():
    src = (ROOT / "scripts" / "check_walker.py").read_text()
    assert "280.5 * i**5 - 58.7 * i**4 - 76.8 * i**3 + 51.9 * i**2 + 19.6 * i + 2.5" in src
    assert core.minetti_walk(0.0) == pytest.approx(2.5)
