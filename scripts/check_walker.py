#!/usr/bin/env python
"""Run this first on Google Cloud, before any big job.

Stages, each run `--repeats` times on smooth ground:
  1. flat, Earth gravity, no suit      -> does the walker walk at all?
  2. flat, Mars gravity, no suit       -> does it survive 0.38 g?
  3. flat, Mars gravity, 40 kg suit    -> does it survive the pack?
  4. slope sweep, Earth, no suit       -> energy curve vs Minetti et al. 2002
  5. slope sweep, Mars, 40 kg suit     -> the curve the planner needs

    python scripts/check_walker.py --walker kinesis
    python scripts/check_walker.py --walker myoleg-deprl --repeats 3
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
from marswalk_sim.runner import rollout  # noqa: E402
from marswalk_sim.terrain import PatchParams  # noqa: E402
from marswalk_sim.walkers import WALKERS, make_walker  # noqa: E402

GRADES = [-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30]
SUIT_KG = 40.0


def minetti_walk(i: float) -> float:
    """Minetti et al. 2002, cost of walking (J/kg/m) at gradient i (rise/run)."""
    return 280.5 * i**5 - 58.7 * i**4 - 76.8 * i**3 + 51.9 * i**2 + 19.6 * i + 2.5


def run(menv, policy, opts, gravity, suit, grade, repeats, tag):
    rows = []
    for r in range(repeats):
        p = PatchParams(slope_deg=float(np.degrees(np.arctan(grade))), gravity=gravity, suit_mass_kg=suit,
                        friction=1.0, pack_up_m=0.05, pack_back_m=0.15, map_id=tag,
                        patch_id=f"{tag}-g{grade:+.2f}-r{r}", seed=1000 + r)
        row = rollout(menv, policy, p, **opts)
        rows.append({"stage": tag, "gravity": gravity, "suit_kg": suit, "grade": grade, "repeat": r,
                     "status": row.get("status"), "fell": bool(row.get("fell")),
                     "distance_m": row.get("distance_m"), "speed_mps": row.get("speed_mps"),
                     "cot_gross": row.get("cot_gross_J_per_kg_m"), "cot_net": row.get("cot_net_J_per_kg_m"),
                     "mos_ml_min_m": row.get("mos_ml_min_m")})
        print(f"  {tag} grade={grade:+.2f} rep={r}: status={rows[-1]['status']} fell={rows[-1]['fell']} "
              f"dist={rows[-1]['distance_m']} cot={rows[-1]['cot_gross']}", flush=True)
    return rows


def stage_summary(df: pd.DataFrame) -> dict:
    walked = df[(df.status == "ok") & (~df.fell)]
    return {"runs": int(len(df)), "walked": int(len(walked)), "fall_rate": float(df.fell.mean()),
            "mean_distance_m": float(df.distance_m.astype(float).mean()),
            "mean_speed_mps": float(walked.speed_mps.astype(float).mean()) if len(walked) else None}


def sweep_summary(df: pd.DataFrame) -> dict:
    out = stage_summary(df)
    walked = df[(df.status == "ok") & (~df.fell)]
    by = walked.groupby("grade")["cot_gross"].mean()
    out["cot_gross_by_grade"] = {f"{k:+.2f}": round(float(v), 3) for k, v in by.items()}
    out["fall_rate_by_grade"] = {f"{k:+.2f}": round(float(v), 2) for k, v in df.groupby("grade").fell.mean().items()}
    if len(by) >= 4:
        ref = np.array([minetti_walk(g) for g in by.index])
        out["pearson_r_vs_minetti"] = float(np.corrcoef(by.values, ref)[0, 1])
        out["cheapest_grade"] = float(by.idxmin())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--walker", default="myoleg-deprl", choices=WALKERS)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="check_walker.json")
    ap.add_argument("--continue-if-not-walking", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out).resolve()  # KINESIS changes the working directory
    menv, policy, opts = make_walker(args.walker)
    result = {"walker": args.walker, "policy": policy.name, "repeats": args.repeats, "stages": {}}
    rows = []

    flat = [("1_flat_earth", C.G_EARTH, 0.0), ("2_flat_mars", C.G_MARS, 0.0), ("3_flat_mars_suit", C.G_MARS, SUIT_KG)]
    for tag, g, suit in flat:
        print(f"stage {tag}", flush=True)
        r = run(menv, policy, opts, g, suit, 0.0, args.repeats, tag)
        rows += r
        result["stages"][tag] = stage_summary(pd.DataFrame(r))
        print(f"  -> {result['stages'][tag]}", flush=True)
        if tag == "1_flat_earth" and result["stages"][tag]["walked"] == 0 and not args.continue_if_not_walking:
            result["verdict"] = "does not walk on flat Earth ground; fix the walker before generating data"
            break
    else:
        for tag, g, suit in (("4_sweep_earth", C.G_EARTH, 0.0), ("5_sweep_mars_suit", C.G_MARS, SUIT_KG)):
            print(f"stage {tag}", flush=True)
            r = []
            for gr in GRADES:
                r += run(menv, policy, opts, g, suit, gr, args.repeats, tag)
            rows += r
            result["stages"][tag] = sweep_summary(pd.DataFrame(r))
            print(f"  -> {json.dumps(result['stages'][tag])}", flush=True)
        st = result["stages"]
        result["verdict"] = (
            f"flat Earth walked {st['1_flat_earth']['walked']}/{st['1_flat_earth']['runs']}, "
            f"flat Mars {st['2_flat_mars']['walked']}/{st['2_flat_mars']['runs']}, "
            f"Mars + suit {st['3_flat_mars_suit']['walked']}/{st['3_flat_mars_suit']['runs']}")
    result["minetti_reference"] = {f"{g:+.2f}": round(minetti_walk(g), 3) for g in GRADES}
    result["runs"] = rows
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print("VERDICT:", result.get("verdict"))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
