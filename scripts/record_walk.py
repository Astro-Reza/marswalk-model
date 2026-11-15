"""Replay generated patches and record the walker's 3D pose every step, for scripts/view_walk.py.

Re-runs the walker on the same patch parameters as a generated run and saves, per patch, one .npz
with every visible geom (type, size, colour, mesh) and its world pose per frame, plus the terrain.
The policy and the start are not bit-identical to the original run, so a replay can walk or fall
where the original did not; the replay's own outcome is stored next to the original one.

    # in the KINESIS environment (WSL, Python 3.8)
    KINESIS_DIR=~/kinesis python scripts/record_walk.py --data data/kinesis-mars-ft-001 --out data/kinesis-mars-ft-001/replay
    python scripts/record_walk.py --data data/kinesis-mars-ft-001 --patch-id r2027-s00002-m000-p03
"""
from __future__ import annotations

import argparse
import glob
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from marswalk_sim.runner import rollout  # noqa: E402
from marswalk_sim.terrain import PatchParams  # noqa: E402
from marswalk_sim.walkers import make_walker  # noqa: E402


def pick_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Walked Mars patches first, then Mars patches that fell on the patch, then one Earth patch."""
    ok = df[df["status"].eq("ok")]
    fell = ok["fell"].astype(bool)
    mars = ok["gravity"].astype(float) < 5
    parts = [ok[mars & ~fell].sort_values("distance_m", ascending=False),
             ok[mars & fell].sort_values("distance_m", ascending=False),
             ok[~mars & ~fell].head(1)]
    return pd.concat(parts).head(n)


def static_scene(m) -> dict:
    """Geoms to draw: everything visible except planes and the terrain (stored separately)."""
    import mujoco
    keep = [i for i in range(m.ngeom)
            if m.geom_rgba[i, 3] > 0 and m.geom_group[i] < 3
            and m.geom_type[i] not in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD)]
    out = {"geom_ids": np.array(keep), "geom_type": m.geom_type[keep].copy(),
           "geom_size": m.geom_size[keep].copy(), "geom_rgba": m.geom_rgba[keep].copy(),
           "geom_mesh": np.array([m.geom_dataid[i] if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH else -1
                                  for i in keep])}
    for mid in sorted({int(x) for x in out["geom_mesh"] if x >= 0}):
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        out[f"mesh{mid}_v"] = m.mesh_vert[va:va + vn].copy()
        out[f"mesh{mid}_f"] = m.mesh_face[fa:fa + fn].copy()
    return out


def record(menv, policy, p: PatchParams, opts: dict) -> dict:
    m, d = menv.m, menv.d
    scene = static_scene(m)
    ids = scene["geom_ids"]
    frames = {"t": [], "xpos": [], "xmat": []}
    step = menv.step

    def recording_step(action):
        res = step(action)
        frames["t"].append(float(d.time))
        frames["xpos"].append(d.geom_xpos[ids].astype(np.float32))
        frames["xmat"].append(d.geom_xmat[ids].astype(np.float32))
        return res

    menv.step = recording_step
    try:
        row = rollout(menv, policy, p, **opts)
    finally:
        menv.step = step
    hid, tid = menv.hid, menv.tid
    nrow, ncol = int(m.hfield_nrow[hid]), int(m.hfield_ncol[hid])
    adr = int(m.hfield_adr[hid])
    scene.update(
        t=np.array(frames["t"]), xpos=np.stack(frames["xpos"]), xmat=np.stack(frames["xmat"]),
        hfield=m.hfield_data[adr:adr + nrow * ncol].reshape(nrow, ncol).copy(),
        hfield_size=m.hfield_size[hid].copy(), terrain_pos=d.geom_xpos[tid].copy(),
        terrain_mat=d.geom_xmat[tid].copy(),
        replay_status=str(row["status"]), replay_fell=bool(row["fell"]),
        replay_distance_m=float(row["distance_m"]), replay_speed_mps=float(row["speed_mps"]))
    return scene


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="generated run folder (part-*.parquet)")
    ap.add_argument("--out", help="output folder (default: <data>/replay)")
    ap.add_argument("--patch-id", nargs="*", help="patches to replay (default: --n picked ones)")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--walker", default="kinesis")
    args = ap.parse_args()

    df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{args.data}/part-*.parquet")))
    rows = df[df["patch_id"].isin(args.patch_id)] if args.patch_id else pick_rows(df, args.n)
    if rows.empty:
        raise SystemExit("no matching patches")
    out = Path(args.out or f"{args.data}/replay").resolve()  # KINESIS changes the working directory
    out.mkdir(parents=True, exist_ok=True)

    menv, policy, opts = make_walker(args.walker)
    names = [f.name for f in fields(PatchParams)]
    for _, r in rows.iterrows():
        p = PatchParams(**{k: r[k] for k in names})
        scene = record(menv, policy, p, opts)
        scene.update(patch_id=p.patch_id, gravity=float(p.gravity), suit_mass_kg=float(p.suit_mass_kg),
                     slope_deg=float(p.slope_deg), orig_fell=bool(r["fell"]),
                     orig_distance_m=float(r["distance_m"]), walker=args.walker)
        np.savez_compressed(out / f"{p.patch_id}.npz", **scene)
        print(f"{p.patch_id}: g={p.gravity:.2f} slope={p.slope_deg:+.1f} frames={len(scene['t'])} "
              f"replay fell={scene['replay_fell']} dist={scene['replay_distance_m']:.2f} "
              f"(original fell={bool(r['fell'])} dist={r['distance_m']:.2f})", flush=True)


if __name__ == "__main__":
    main()
