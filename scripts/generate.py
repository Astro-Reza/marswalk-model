#!/usr/bin/env python
"""Generate one shard of walking-simulation data.

Each shard simulates `--num-maps` maps x `--patches-per-map` patches and writes
Parquet part files to `--out`. On Google Cloud Batch every task runs this with
its own BATCH_TASK_INDEX, so shards never overlap and can be re-run safely.

Example (local smoke test, falls immediately because the policy is "zero"):
    python scripts/generate.py --out data/smoke --num-maps 1 --patches-per-map 2 --policy zero
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marswalk_sim import __version__, config as C  # noqa: E402
from marswalk_sim.env import MarsPatchEnv  # noqa: E402
from marswalk_sim.policies import make_policy  # noqa: E402
from marswalk_sim.runner import rollout  # noqa: E402
from marswalk_sim.terrain import PatchParams, sample_map_surface, sample_params  # noqa: E402


def calibrate(menv: MarsPatchEnv, policy, seconds: float = 4.0) -> dict:
    """Walk on flat Earth ground to check the policy walks and which way is forward."""
    p = PatchParams(gravity=C.G_EARTH, friction=1.0, map_id="calib", patch_id="calib", seed=1)
    from marswalk_sim.terrain import build_heightfield

    obs = menv.reset(p, build_heightfield(p, menv.nrow, menv.ncol))
    policy.reset()
    y0 = float(menv.com()[1])
    fell = False
    for _ in range(int(seconds / menv.dt)):
        obs, _, term, trunc, _ = menv.step(policy(obs))
        if term or trunc:
            fell = bool(term)
            break
    dy = float(menv.com()[1]) - y0
    walks = abs(dy) > 1.0 and not fell
    if walks and np.sign(dy) != menv.forward:
        menv.forward = int(np.sign(dy))
    return {"calib_dy_m": dy, "calib_fell": fell, "policy_walks_on_flat_earth": walks,
            "forward_sign": menv.forward, "calib_seconds": seconds}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output folder (a mounted bucket path on Cloud Batch)")
    ap.add_argument("--shard-index", type=int, default=int(os.environ.get("BATCH_TASK_INDEX", 0)))
    ap.add_argument("--num-maps", type=int, default=4)
    ap.add_argument("--patches-per-map", type=int, default=C.PATCHES_PER_MAP)
    ap.add_argument("--policy", default="deprl", choices=["deprl", "zero", "random"])
    ap.add_argument("--seed", type=int, default=2026, help="run seed; combined with the shard index")
    ap.add_argument("--flush-every", type=int, default=8, help="write a part file every N patches")
    ap.add_argument("--save-traj", action="store_true", help="also save per-patch trajectories (.npz)")
    ap.add_argument("--skip-calibration", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard = args.shard_index
    rng = np.random.default_rng([args.seed, shard])

    menv = MarsPatchEnv()
    policy = make_policy(args.policy, menv.env, seed=shard)

    meta = {
        "shard_index": shard, "seed": args.seed, "policy": policy.name, "env_id": C.ENV_ID,
        "marswalk_sim": __version__, "python": platform.python_version(), "started_utc": time.time(),
    }
    try:
        import mujoco, myosuite  # noqa: E401
        meta |= {"mujoco": mujoco.__version__, "myosuite": myosuite.__version__}
    except Exception:  # pragma: no cover
        pass
    if not args.skip_calibration:
        meta |= calibrate(menv, policy)
        print(f"[shard {shard}] calibration: {meta}", flush=True)
        if args.policy == "deprl" and not meta["policy_walks_on_flat_earth"]:
            print(f"[shard {shard}] WARNING: the policy does not walk on flat Earth ground. "
                  "Check MyoSuite/DEP-RL versions before spending compute.", flush=True)
    (out / f"meta-{shard:05d}.json").write_text(json.dumps(meta, indent=2, default=str))

    rows, part = [], 0
    total = args.num_maps * args.patches_per_map
    done = 0
    for k in range(args.num_maps):
        map_id = f"r{args.seed}-s{shard:05d}-m{k:03d}"
        surface = sample_map_surface(rng)
        for j in range(args.patches_per_map):
            p = sample_params(rng, map_id, f"{map_id}-p{j:02d}", shared=surface)
            try:
                row = rollout(menv, policy, p, keep_traj=args.save_traj)
            except Exception as e:  # keep the shard alive; record the failure
                row = p.to_dict() | {"status": f"error: {type(e).__name__}: {e}"[:300]}
            traj = row.pop("_traj", None)
            if traj is not None:
                (out / "traj").mkdir(exist_ok=True)
                np.savez_compressed(out / "traj" / f"{p.patch_id}.npz", **traj)
            row |= {"policy": policy.name, "forward_sign": menv.forward, "shard_index": shard}
            rows.append(row)
            done += 1
            print(f"[shard {shard}] {done}/{total} {p.patch_id} slope={p.slope_deg:+.1f} "
                  f"g={p.gravity:.2f} status={row.get('status')} fell={row.get('fell')} "
                  f"dist={row.get('distance_m', float('nan')):.2f} wall={row.get('wall_time_s', 0):.1f}s", flush=True)
            if len(rows) >= args.flush_every:
                pd.DataFrame(rows).to_parquet(out / f"part-{shard:05d}-{part:04d}.parquet", index=False)
                rows, part = [], part + 1
    if rows:
        pd.DataFrame(rows).to_parquet(out / f"part-{shard:05d}-{part:04d}.parquet", index=False)
    print(f"[shard {shard}] finished {done} patches", flush=True)


if __name__ == "__main__":
    main()
