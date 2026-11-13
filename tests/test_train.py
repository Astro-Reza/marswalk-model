"""Exercises train.py on a synthetic dataset shaped like generate.py output.

The labels here come from a Pandolf-like formula plus noise, only so the
training code can be tested without a walking policy.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def fake_shard(rng, shard, n_maps=10, per_map=8):
    rows = []
    for m in range(n_maps):
        fric, rough = rng.uniform(0.4, 1.0), rng.uniform(0, 0.06)
        for j in range(per_map):
            slope = rng.uniform(-25, 25)
            g = 3.711 if rng.uniform() > 0.1 else 9.81
            suit = rng.uniform(0, 60) if g < 5 else 0.0
            i = np.tan(np.radians(slope))
            cot = (2.5 + 19.6 * i + 51.9 * i**2) * g / 9.81 * (1 + suit / 150) * (1 + 5 * rough) + rng.normal(0, 0.2)
            p_fall = 1 / (1 + np.exp(-(abs(slope) - 18) / 3 - 30 * rough + 2 * (fric - 0.7)))
            fell = bool(rng.uniform() < p_fall)
            rows.append(dict(map_id=f"s{shard}-m{m}", patch_id=f"s{shard}-m{m}-p{j}", status="ok", fell=fell,
                             dtm_slope_along_deg=slope, dtm_slope_cross_deg=rng.uniform(-10, 10),
                             dtm_slope_p90_deg=abs(slope) + 3, dtm_roughness_m=rough,
                             rock_density_per_m2=rng.uniform(0, 0.4), rock_height_m=rng.uniform(0.03, 0.25),
                             friction=fric, gravity=g, suit_mass_kg=suit, pack_up_m=0.0, pack_back_m=0.1,
                             cot_net_J_per_kg_m=max(cot, 0.3), mos_ml_min_m=0.08 - rough + rng.normal(0, 0.01)))
    rows.append(dict(map_id="s-x", patch_id="bad", status="fell_before_patch", fell=True))
    return pd.DataFrame(rows)


def test_train_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    data = tmp_path / "data"
    data.mkdir()
    for s in range(3):
        fake_shard(rng, s).to_parquet(data / f"part-{s:05d}-0000.parquet", index=False)
    out = tmp_path / "model"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py"), "--data", str(data), "--out", str(out)],
                   check=True)
    metrics = json.loads((out / "metrics.json").read_text())
    split = json.loads((out / "split.json").read_text())
    assert not set(split["train"]) & set(split["test"])
    assert not set(split["train"]) & set(split["val"])
    assert metrics["cot"]["test"]["r2"] > 0.8
    assert "auc" in metrics["fall"]["test"]
    assert (out / "planner_lookup_mars.json").exists()
