"""The data-quality gate: a healthy run passes, each kind of damage is caught.

Rows are made up (a Pandolf-like cost formula), shaped like generate.py output.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from marswalk_sim import __version__, config as C
from marswalk_sim import quality as Q

ROOT = Path(__file__).resolve().parents[1]


def healthy(n_maps=60, per_map=8, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_maps):
        fric, rough = rng.uniform(0.4, 1.0), rng.uniform(0, 0.03)
        earth = m % 5 == 0
        for j in range(per_map):
            slope = rng.uniform(-25, 25)
            g, suit = (C.G_EARTH, 0.0) if earth else (C.G_MARS, rng.uniform(0, 60))
            i = np.tan(np.radians(slope))
            cot = max(float(Q.minetti_walk(i)) * (1 + suit / 150) + rng.normal(0, 0.1), 0.4)
            fell = bool(rng.uniform() < (0.5 if abs(slope) > 20 else 0.05))
            rows.append(dict(
                map_id=f"m{m}", patch_id=f"m{m}-p{j}", status="ok", fell=fell, fall_reason="height" if fell else "",
                sim_version=__version__, walker="myoleg-deprl", slope_deg=slope,
                dtm_slope_along_deg=slope + rng.normal(0, 0.1), dtm_slope_cross_deg=rng.uniform(-10, 10),
                dtm_slope_p90_deg=abs(slope) + 2, dtm_roughness_m=rough, rock_density_per_m2=rng.uniform(0, 0.4),
                rock_height_m=rng.uniform(0.03, 0.25), friction=fric, gravity=g, suit_mass_kg=suit,
                pack_up_m=rng.uniform(-0.15, 0.25), pack_back_m=rng.uniform(0.05, 0.25),
                distance_m=3.0 if fell else 10.0, speed_mps=rng.uniform(0.9, 1.3), cot_net_J_per_kg_m=cot,
                cot_gross_J_per_kg_m=cot + 1.0, stride_time_mean_s=1.1, peak_grf_bw=1.3, mos_ml_min_m=0.05,
                path_slope_deg=slope + rng.normal(0, 0.5), com_height_start_m=1.0,
                com_height_min_m=0.3 if fell else 0.93, patch_length_m=10.5))
    return pd.DataFrame(rows)


def levels(report: dict) -> dict:
    return {c["check"]: c["level"] for c in report["checks"]}


def test_healthy_run_passes():
    r = Q.dataset_report(healthy(), metas=[{"shard_index": 0, "policy_walks_on_flat_earth": True, "finished": True}])
    assert r["verdict"] == "pass", [c for c in r["checks"] if c["level"] != "pass"]
    assert r["summary"]["rows"] == 480 and len(r["summary"]["by_slope"]) == 5
    assert levels(r)["earth_slope_curve"] == "pass"


def test_label_mask_needs_distance_and_no_fall():
    df = healthy(4)
    df.loc[0, ["fell", "distance_m"]] = [False, C.MIN_LABEL_DISTANCE_M - 0.5]   # stayed up but barely moved
    df.loc[1, "status"] = "fell_before_patch"
    walked, reached = Q.label_mask(df, "walked"), Q.label_mask(df, "reached_patch")
    assert not walked[0] and reached[0] and not reached[1]
    assert walked.sum() == (~df["fell"] & df["status"].eq("ok")).sum() - 1
    with pytest.raises(ValueError):
        Q.label_mask(df, "nonsense")


@pytest.mark.parametrize("damage,check,level", [
    (lambda d: pd.concat([d, d.head(8)], ignore_index=True), "unique_patches", "fail"),
    (lambda d: d.drop(columns="sim_version"), "sim_version", "fail"),
    (lambda d: d.assign(sim_version=np.where(d.index < 100, "0.1.0", __version__)), "sim_version", "fail"),
    (lambda d: d.assign(status=np.where(d.index % 4 == 0, "error: ValueError: boom", "ok")), "errors", "fail"),
    (lambda d: d.assign(fell=True), "mars_walked", "fail"),
    (lambda d: d.assign(speed_mps=0.05), "range_speed_mps", "fail"),
    (lambda d: d.assign(dtm_slope_along_deg=-d["slope_deg"]), "slope_feature", "fail"),
    (lambda d: d.assign(com_height_min_m=0.3), "missed_falls", "fail"),
    (lambda d: d[d["slope_deg"] < 10], "slope_coverage", "warn"),
    (lambda d: d.assign(friction=0.7), "feature_variation", "warn"),
    (lambda d: d.assign(fall_reason="turned"), "fall_reasons", "warn"),
    (lambda d: d.assign(cot_gross_J_per_kg_m=np.random.default_rng(1).uniform(1, 5, len(d))), "earth_slope_curve", "warn"),
    (lambda d: d.drop(columns="friction"), "schema", "fail"),
])
def test_damage_is_caught(damage, check, level):
    r = Q.dataset_report(damage(healthy()))
    assert levels(r).get(check) == level, r["checks"]
    assert r["verdict"] in ("fail", level)


def test_shard_metadata_checks():
    metas = [{"shard_index": 0, "policy_walks_on_flat_earth": True, "finished": True},
             {"shard_index": 1, "policy_walks_on_flat_earth": False, "finished": False}]
    lv = levels(Q.dataset_report(healthy(), metas))
    assert lv["calibration"] == "fail" and lv["shards_finished"] == "warn"


def test_validate_script_exit_codes(tmp_path):
    good, bad = tmp_path / "good", tmp_path / "bad"
    for folder, df in ((good, healthy()), (bad, healthy().drop(columns="sim_version"))):
        folder.mkdir()
        df.to_parquet(folder / "part-00000-0000.parquet", index=False)
        (folder / "meta-00000.json").write_text(json.dumps({"shard_index": 0, "policy_walks_on_flat_earth": True,
                                                            "finished": True}))
    cmd = [sys.executable, str(ROOT / "scripts" / "validate.py")]
    ok = subprocess.run(cmd + ["--data", str(good), "--out", str(good / "quality.json")], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert json.loads((good / "quality.json").read_text())["verdict"] == "pass" and "VERDICT: pass" in ok.stdout
    out = bad / "quality.json"
    assert subprocess.run(cmd + ["--data", str(bad), "--out", str(out)], capture_output=True).returncode == 1
    assert subprocess.run(cmd + ["--data", str(bad), "--out", str(out), "--no-fail"], capture_output=True).returncode == 0
    # training refuses the same data when asked to
    train = [sys.executable, str(ROOT / "scripts" / "train.py"), "--out", str(tmp_path / "model"), "--require-quality"]
    assert subprocess.run(train + ["--data", str(bad)], capture_output=True).returncode != 0
    done = subprocess.run(train + ["--data", str(good)], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    metrics = json.loads((tmp_path / "model" / "metrics.json").read_text())
    assert metrics["quality"]["verdict"] == "pass" and "test" in metrics["speed"]


def test_features_match_train_script():
    import ast
    tree = ast.parse((ROOT / "scripts" / "train.py").read_text())
    feats = next(ast.literal_eval(n.value) for n in tree.body
                 if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "FEATURES")
    assert feats == Q.FEATURES
