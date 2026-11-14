#!/usr/bin/env python
"""Generate one shard of walking-simulation data.

Each shard simulates `--num-maps` maps x `--patches-per-map` patches and writes
Parquet part files to `--out`. On Google Cloud Batch every task runs this with
its own BATCH_TASK_INDEX, so shards never overlap and can be re-run safely.

Walkers: myoleg-deprl (default), kinesis, myoleg-zero / myoleg-random (smoke tests).

    python scripts/generate.py --out data/smoke --num-maps 1 --patches-per-map 2 --walker myoleg-zero
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
from marswalk_sim.runner import rollout  # noqa: E402
from marswalk_sim.terrain import PatchParams, build_heightfield, sample_map_surface, sample_params  # noqa: E402
from marswalk_sim.walkers import WALKERS, make_walker  # noqa: E402


def calibrate(menv, policy, seconds: float = 4.0, flat_start: float = C.FLAT_START_M) -> dict:
    """Walk on flat Earth ground: does the policy walk, and which way is forward?"""
    p = PatchParams(gravity=C.G_EARTH, friction=1.0, map_id="calib", patch_id="calib", seed=1)
    obs = menv.reset(p, build_heightfield(p, menv.nrow, menv.ncol, flat_start=flat_start))
    policy.reset()
    c0 = menv.com().copy()
    fell = False
    for _ in range(int(seconds / menv.dt)):
        obs, _, term, trunc, _ = menv.step(policy(obs))
        if term or trunc:
            fell = bool(term)
            break
    c1 = menv.com()
    ds = float(menv.frame.s_of(c1) - menv.frame.s_of(c0))
    drift = float(abs(menv.frame.lat_of(c1) - menv.frame.lat_of(c0)))
    walks = abs(ds) > 1.0 and not fell
    flipped = False
    if walks and ds < 0 and hasattr(type(menv), "forward"):
        menv.forward = -menv.forward  # MyoLeg: walking direction is a convention we can flip
        flipped, ds = True, -ds
    return {"calib_forward_m": ds, "calib_lateral_drift_m": drift, "calib_fell": fell,
            "policy_walks_on_flat_earth": walks, "calib_flipped_forward": flipped,
            "frame_axis": menv.frame.axis, "frame_sign": menv.frame.sign, "calib_seconds": seconds}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output folder (a mounted bucket path on Cloud Batch)")
    ap.add_argument("--shard-index", type=int, default=int(os.environ.get("BATCH_TASK_INDEX", 0)))
    ap.add_argument("--num-maps", type=int, default=4)
    ap.add_argument("--patches-per-map", type=int, default=C.PATCHES_PER_MAP)
    ap.add_argument("--walker", default="myoleg-deprl", choices=WALKERS)
    ap.add_argument("--seed", type=int, default=2026, help="run seed; combined with the shard index")
    ap.add_argument("--flush-every", type=int, default=8, help="write a part file every N patches")
    ap.add_argument("--save-traj", action="store_true", help="also save per-patch trajectories (.npz)")
    ap.add_argument("--skip-calibration", action="store_true")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    shard = args.shard_index
    rng = np.random.default_rng([args.seed, shard])

    menv, policy, opts = make_walker(args.walker, seed=shard)

    meta = {"shard_index": shard, "seed": args.seed, "walker": args.walker, "policy": policy.name,
            "marswalk_sim": __version__, "python": platform.python_version(), "started_utc": time.time()}
    try:
        import mujoco
        meta["mujoco"] = mujoco.__version__
    except Exception:  # pragma: no cover
        pass
    if not args.skip_calibration:
        meta.update(calibrate(menv, policy, flat_start=opts["flat_start"]))
        print(f"[shard {shard}] calibration: {meta}", flush=True)
        if args.walker in ("myoleg-deprl", "kinesis") and not meta["policy_walks_on_flat_earth"]:
            print(f"[shard {shard}] WARNING: the walker does not walk on flat Earth ground. "
                  "Run scripts/check_walker.py before spending compute.", flush=True)
    (out / f"meta-{shard:05d}.json").write_text(json.dumps(meta, indent=2, default=str))

    rows, part, done = [], 0, 0
    total = args.num_maps * args.patches_per_map
    for k in range(args.num_maps):
        map_id = f"r{args.seed}-s{shard:05d}-m{k:03d}"
        surface = sample_map_surface(rng)
        for j in range(args.patches_per_map):
            p = sample_params(rng, map_id, f"{map_id}-p{j:02d}", shared=surface)
            try:
                row = rollout(menv, policy, p, keep_traj=args.save_traj, **opts)
            except Exception as e:  # keep the shard alive; record the failure
                row = p.to_dict()
                row["status"] = f"error: {type(e).__name__}: {e}"[:300]
            traj = row.pop("_traj", None)
            if traj is not None:
                (out / "traj").mkdir(exist_ok=True)
                np.savez_compressed(out / "traj" / f"{p.patch_id}.npz", **traj)
            row.update(policy=policy.name, walker=args.walker, frame_axis=menv.frame.axis,
                       frame_sign=menv.frame.sign, shard_index=shard)
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
