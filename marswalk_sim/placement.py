"""Put a Heightfield into a compiled MuJoCo model, along any horizontal axis.

Frame used throughout marswalk_sim:
  world point = origin + s * fwd + lat * side
where fwd is the walking direction (+/-x or +/-y) and side is the horizontal
axis 90 degrees from it. Python 3.8 compatible (used by the KINESIS backend).
"""
from __future__ import annotations

import mujoco
import numpy as np

from .terrain import Heightfield


class Frame:
    def __init__(self, origin_xy, axis: int, sign: int):
        if axis not in (0, 1) or sign not in (-1, 1):
            raise ValueError("axis must be 0 (x) or 1 (y) and sign +/-1")
        self.origin = np.array([float(origin_xy[0]), float(origin_xy[1]), 0.0])
        self.axis, self.sign = axis, sign
        self.fwd = np.zeros(3)
        self.fwd[axis] = sign
        self.side = np.zeros(3)
        self.side[1 - axis] = 1.0

    def s_of(self, p):
        p = np.asarray(p, dtype=float)
        return (p - self.origin) @ self.fwd

    def lat_of(self, p):
        p = np.asarray(p, dtype=float)
        return (p - self.origin) @ self.side

    def v_s(self, v):
        return np.asarray(v, dtype=float) @ self.fwd

    def v_lat(self, v):
        return np.asarray(v, dtype=float) @ self.side

    def world(self, s, lat):
        return self.origin + s * self.fwd + lat * self.side


def place_heightfield(m, d, tid: int, hid: int, hf: Heightfield, frame: Frame, friction: float,
                      base: float = 0.1) -> None:
    """Write hf into hfield `hid` used by geom `tid`, oriented by `frame`.

    If the geom was compiled with no offset from its body, MuJoCo marks it as
    sharing the body frame (geom_sameframe) and ignores later geom_pos changes,
    so that flag is cleared. The pose is also written into `d` so it holds
    before the next mj_forward.
    """
    nrow, ncol = int(m.hfield_nrow[hid]), int(m.hfield_ncol[hid])
    adr = int(m.hfield_adr[hid])
    z = hf.z  # rows = s, cols = lat
    if frame.axis == 1:            # walking along y: hfield rows (y) follow s
        grid = z if frame.sign > 0 else z[::-1, :]
        half_x = (hf.x[-1] - hf.x[0]) / 2
        half_y = (hf.s[-1] - hf.s[0]) / 2
    else:                          # walking along x: hfield cols (x) follow s
        grid = z.T if frame.sign > 0 else z.T[:, ::-1]
        half_x = (hf.s[-1] - hf.s[0]) / 2
        half_y = (hf.x[-1] - hf.x[0]) / 2
    if grid.shape != (nrow, ncol):
        raise ValueError(f"heightfield grid must be {(nrow, ncol)}, got {grid.shape}")

    zmin, zmax = float(grid.min()), float(grid.max())
    zr = max(zmax - zmin, 1e-3)
    m.hfield_data[adr:adr + grid.size] = ((grid - zmin) / zr).ravel().astype(np.float32)
    m.hfield_size[hid] = [half_x, half_y, zr, base]

    s_center = (hf.s[-1] + hf.s[0]) / 2
    lat_center = (hf.x[-1] + hf.x[0]) / 2
    center = frame.world(s_center, lat_center)
    m.geom_pos[tid] = [center[0], center[1], zmin]
    m.geom_quat[tid] = [1.0, 0.0, 0.0, 0.0]
    try:
        m.geom_sameframe[tid] = 0
    except AttributeError:  # very old MuJoCo
        pass
    d.geom_xpos[tid] = m.geom_pos[tid]
    d.geom_xmat[tid] = np.eye(3).ravel()
    # broad-phase bounds were compiled for the original hfield size
    m.geom_rbound[tid] = float(np.sqrt(half_x**2 + half_y**2 + (zr + base) ** 2))
    try:
        m.geom_aabb[tid] = [0.0, 0.0, (zr - base) / 2, half_x, half_y, (zr + base) / 2]
    except AttributeError:  # older MuJoCo without geom_aabb
        pass
    m.geom_friction[tid, 0] = friction
    m.geom_rgba[tid, 3] = 1.0
    m.geom_contype[tid] = 1
    m.geom_conaffinity[tid] = 1


def ground_z(hf: Heightfield, s: float, lat: float) -> float:
    i = int(np.clip(np.searchsorted(hf.s, s), 1, len(hf.s) - 1))
    j = int(np.clip(np.searchsorted(hf.x, lat), 1, len(hf.x) - 1))
    return float(hf.z[i, j])


def set_const_keep_state(m, d) -> None:
    """mj_setConst after changing masses, without losing the walker's pose.

    mj_setConst uses `d` as scratch space and leaves the joint angles at qpos0,
    so the reset pose (set by the environment) must be saved and put back.
    """
    saved = {k: getattr(d, k).copy() for k in ("qpos", "qvel", "act", "qacc_warmstart")}
    t = d.time
    mujoco.mj_setConst(m, d)
    for k, v in saved.items():
        getattr(d, k)[:] = v
    d.time = t
    mujoco.mj_forward(m, d)
