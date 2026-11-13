#!/usr/bin/env python
"""Run this first on Google Cloud, before any big job.

1. Does the policy walk on flat Earth ground? (If not, nothing else matters.)
2. Earth slope sweep: does simulated cost of transport follow the shape of the
   Minetti et al. (2002) curve (cheapest on a gentle downhill)?
3. The same sweep at Mars gravity with a suit, for comparison.

    python scripts/check_walker.py --policy deprl --repeats 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marswalk_sim import config as C  # noqa: E402
from marswalk_sim.env import MarsPatchEnv  # noqa: E402
from marswalk_sim.policies import make_policy  # noqa: E402
from marswalk_sim.runner import rollout  # noqa: E402
from marswalk_sim.terrain import PatchParams  # noqa: E402

GRADES = [-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30]


def minetti_walk(i: float) -> float:
    """Minetti et al. 2002, cost of walking (J/kg/m) at gradient i (rise/run)."""
    return 280.5 * i**5 - 58.7 * i**4 - 76.8 * i**3 + 51.9 * i**2 + 19.6 * i + 2.5


def sweep(menv, policy, gravity: float, suit: float, repeats: int) -> pd.DataFrame:
    rows = []
    for gr in GRADES:
        for r in range(repeats):
            p = PatchParams(slope_deg=float(np.degrees(np.arctan(gr))), gravity=gravity, suit_mass_kg=suit,
                            friction=1.0, pack_up_m=0.05, pack_back_m=0.15, map_id="check",
                            patch_id=f"g{gr:+.2f}-r{r}", seed=1000 + r)
            row = rollout(menv, policy, p)
            rows.append({"grade": gr, "repeat": r, "status": row.get("status"), "fell": row.get("fell"),
                         "distance_m": row.get("distance_m"), "speed_mps": row.get("speed_mps"),
                         "cot_gross": row.get("cot_gross_J_per_kg_m"), "cot_net": row.get("cot_net_J_per_kg_m")})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, label: str) -> dict:
    ok = df[(df.status == "ok") & (~df.fell.astype(bool))]
    by = ok.groupby("grade")["cot_gross"].mean()
    out = {"label": label, "walked": int(len(ok)), "runs": int(len(df)),
           "fall_rate": float(df.fell.astype(bool).mean()),
           "cot_gross_by_grade": {f"{k:+.2f}": round(float(v), 3) for k, v in by.items()}}
    if len(by) >= 4:
        ref = np.array([minetti_walk(g) for g in by.index])
        out["pearson_r_vs_minetti"] = float(np.corrcoef(by.values, ref)[0, 1])
        out["cheapest_grade"] = float(by.idxmin())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="deprl", choices=["deprl", "zero", "random"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="check_walker.json")
    args = ap.parse_args()

    menv = MarsPatchEnv()
    policy = make_policy(args.policy, menv.env)

    from generate import calibrate  # same folder

    calib = calibrate(menv, policy)
    print("flat-ground calibration:", calib, flush=True)
    result = {"calibration": calib}
    if not calib["policy_walks_on_flat_earth"]:
        print("The policy does not walk on flat Earth ground. Fix this before generating data "
              "(try another MyoSuite/DEP-RL version, or fine-tune / train a policy).")
    else:
        earth = sweep(menv, policy, C.G_EARTH, 0.0, args.repeats)
        mars = sweep(menv, policy, C.G_MARS, 40.0, args.repeats)
        result["earth"] = summarise(earth, "Earth 9.81, no suit")
        result["mars"] = summarise(mars, "Mars 3.71, 40 kg suit")
        result["minetti_reference"] = {f"{g:+.2f}": round(minetti_walk(g), 3) for g in GRADES}
        print(json.dumps(result, indent=2))
    Path(args.out).write_text(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
