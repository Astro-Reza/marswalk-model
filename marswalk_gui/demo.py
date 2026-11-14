"""SYNTHETIC demo data so the GUI can be tried without a walking policy or cloud run.

The labels come from a Pandolf-like formula plus noise (same idea as tests/test_train.py). They are not
simulation output and not Mars measurements. Every demo folder carries a DEMO_SYNTHETIC.txt marker.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MARKER = "DEMO_SYNTHETIC.txt"


def synthetic_shard(rng: np.random.Generator, shard: int, n_maps: int = 12, per_map: int = 8) -> pd.DataFrame:
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
            rows.append(dict(
                map_id=f"s{shard}-m{m}", patch_id=f"s{shard}-m{m}-p{j}", status="ok",
                fell=bool(rng.uniform() < p_fall), walker="synthetic-demo",
                dtm_slope_along_deg=slope, dtm_slope_cross_deg=rng.uniform(-10, 10),
                dtm_slope_p90_deg=abs(slope) + 3, dtm_roughness_m=rough,
                rock_density_per_m2=rng.uniform(0, 0.4), rock_height_m=rng.uniform(0.03, 0.25),
                friction=fric, gravity=g, suit_mass_kg=suit, pack_up_m=0.0, pack_back_m=0.1,
                cot_net_J_per_kg_m=max(cot, 0.3), mos_ml_min_m=0.08 - rough + rng.normal(0, 0.01),
                speed_mps=rng.uniform(0.6, 1.4), distance_m=rng.uniform(6, 12)))
    return pd.DataFrame(rows)


def make_demo(out_dir: str | Path, seed: int = 0, train: bool = True) -> tuple[Path, Path | None]:
    """Write synthetic shards to <out_dir>/data and (optionally) train a surrogate into <out_dir>/model."""
    out = Path(out_dir)
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    (out / MARKER).write_text("Synthetic data from marswalk_gui.demo. Not simulation output, not Mars measurements.\n")
    rng = np.random.default_rng(seed)
    for s in range(4):
        synthetic_shard(rng, s).to_parquet(data / f"part-{s:05d}-0000.parquet", index=False)
    model = None
    if train:
        model = out / "model"
        subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py"), "--data", str(data), "--out", str(model),
                        "--seed", str(seed)], check=True, capture_output=True)
    return data, model
