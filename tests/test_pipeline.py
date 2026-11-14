"""generate.py -> Parquet -> validate.py, with the non-walking `zero` controller.

The zero controller collapses at the start, so the labels are empty; what is
checked is that a shard writes every column the quality gate and the trainer
read, and that a shard which cannot walk is reported, not silently accepted.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from marswalk_sim import __version__
from marswalk_sim.quality import REQUIRED
from marswalk_sim.runner import LABEL_KEYS

ROOT = Path(__file__).resolve().parents[1]


def test_generate_then_validate(tmp_path):
    out = tmp_path / "run"
    gen = [sys.executable, str(ROOT / "scripts" / "generate.py"), "--out", str(out), "--num-maps", "1",
           "--patches-per-map", "2", "--walker", "myoleg-zero", "--seed", "7"]
    done = subprocess.run(gen, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr

    df = pd.read_parquet(out / "part-00000-0000.parquet")
    assert len(df) == 2 and df["patch_id"].is_unique and df["map_id"].nunique() == 1
    missing = [c for c in REQUIRED + list(LABEL_KEYS) + ["sim_version", "fall_reason", "patch_length_m",
                                                         "fine_roughness_m", "dtm_post_m", "run_seed"] if c not in df]
    assert not missing, missing
    assert df["sim_version"].eq(__version__).all() and df["run_seed"].eq(7).all()
    assert df["map_id"].str.startswith("r7-").all()          # the seed is part of the map name
    assert not df["status"].str.startswith("error").any(), df["status"].tolist()

    meta = json.loads((out / "meta-00000.json").read_text())
    assert meta["finished"] and meta["patches"] == 2 and meta["policy_walks_on_flat_earth"] is False

    val = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate.py"), "--data", str(out),
                          "--out", str(out / "quality.json")], capture_output=True, text=True)
    assert val.returncode == 1, val.stdout + val.stderr       # a walker that cannot walk must not pass
    report = json.loads((out / "quality.json").read_text())
    levels = {c["check"]: c["level"] for c in report["checks"]}
    assert levels["schema"] == "pass" and levels["sim_version"] == "pass" and levels["calibration"] == "fail"
