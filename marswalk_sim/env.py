"""MyoSuite myoLeg walker on a Mars terrain patch.

Wraps a MyoSuite leg-walking env and, on every reset, overwrites:
  * the terrain heightfield with our patch,
  * gravity,
  * terrain friction,
  * torso mass / COM / inertia to carry a suit + PLSS pack.
"""
from __future__ import annotations

import warnings

import mujoco
import numpy as np

from . import config as C
from .terrain import Heightfield, PatchParams

warnings.filterwarnings("ignore", category=UserWarning)

FOOT_BODIES = {"r": ("calcn_r", "toes_r"), "l": ("calcn_l", "toes_l")}

# MyoSuite's walk task rewards a positive com_vel[1], and com_vel is the
# negated MuJoCo cvel, so the walker heads toward world -y. calibrate() checks
# this with the real policy and overrides it if needed.
DEFAULT_FORWARD_SIGN = -1


class MarsPatchEnv:
    def __init__(self, env_id: str = C.ENV_ID, forward_sign: int = DEFAULT_FORWARD_SIGN):
        from myosuite.utils import gym  # imported here so the module loads without MyoSuite

        self.env = gym.make(env_id)
        self.uw = self.env.unwrapped
        self.m: mujoco.MjModel = self.uw.model
        self.d: mujoco.MjData = self.uw.data
        m = self.m

        self.tid = m.geom("terrain").id
        self.hid = int(m.geom_dataid[self.tid])
        self.nrow = int(m.hfield_nrow[self.hid])
        self.ncol = int(m.hfield_ncol[self.hid])
        self.hadr = int(m.hfield_adr[self.hid])
        self.forward = int(forward_sign)

        # The model also has a flat collision floor at z = 0. Left on, the walker
        # would stand on it wherever our terrain dips below zero (downhill).
        for name in ("floor",):
            try:
                gid = m.geom(name).id
            except KeyError:
                continue
            m.geom_contype[gid] = 0
            m.geom_conaffinity[gid] = 0
            m.geom_rgba[gid, 3] = 0.0

        self.torso = m.body("torso").id
        self._orig = {
            "mass": m.body_mass[self.torso].copy(),
            "ipos": m.body_ipos[self.torso].copy(),
            "inertia": m.body_inertia[self.torso].copy(),
        }
        self.human_mass = float(m.body_mass.sum())
        self.foot_body_ids = {
            side: {m.body(n).id for n in names} for side, names in FOOT_BODIES.items()
        }
        self.muscle_mask = m.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        self.dt = float(self.uw.dt)
        self.hf: Heightfield | None = None
        self.params: PatchParams | None = None

    # ------------------------------------------------------------------ setup
    def _apply_terrain(self, hf: Heightfield, friction: float) -> None:
        m = self.m
        z = hf.z
        if z.shape != (self.nrow, self.ncol):
            raise ValueError(f"heightfield must be {(self.nrow, self.ncol)}, got {z.shape}")
        # rows of the MuJoCo hfield run along +y; our rows run along +s
        if self.forward < 0:
            z = z[::-1, :]
        zmin, zmax = float(z.min()), float(z.max())
        zr = max(zmax - zmin, 1e-3)
        data = (z - zmin) / zr
        m.hfield_data[self.hadr:self.hadr + data.size] = data.ravel().astype(np.float32)

        sx = (hf.x[-1] - hf.x[0]) / 2
        sy = (hf.s[-1] - hf.s[0]) / 2
        base = 0.1
        m.hfield_size[self.hid] = [sx, sy, zr, base]
        s_center = (hf.s[-1] + hf.s[0]) / 2
        m.geom_pos[self.tid] = [0.0, self.forward * s_center, zmin]
        # broad-phase bounds were compiled for the original, smaller hfield
        m.geom_rbound[self.tid] = float(np.sqrt(sx**2 + sy**2 + (zr + base) ** 2))
        m.geom_aabb[self.tid] = [0.0, 0.0, (zr - base) / 2, sx, sy, (zr + base) / 2]
        m.geom_friction[self.tid, 0] = friction
        m.geom_rgba[self.tid, 3] = 1.0
        m.geom_contype[self.tid] = 1
        m.geom_conaffinity[self.tid] = 1

    def _apply_pack(self, p: PatchParams) -> None:
        m, d, t = self.m, self.d, self.torso
        m.body_mass[t] = self._orig["mass"]
        m.body_ipos[t] = self._orig["ipos"]
        m.body_inertia[t] = self._orig["inertia"]
        L = float(p.suit_mass_kg)
        if L <= 0:
            return
        mujoco.mj_forward(m, d)
        fwd = np.array([0.0, float(self.forward), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        pack_world = d.xipos[t] + up * p.pack_up_m - fwd * p.pack_back_m
        R = d.xmat[t].reshape(3, 3)
        pack_local = R.T @ (pack_world - d.xpos[t])
        m0 = float(self._orig["mass"])
        new_ipos = (m0 * self._orig["ipos"] + L * pack_local) / (m0 + L)
        # inertia: box-shaped pack (0.4 x 0.3 x 0.6 m) + parallel-axis terms,
        # added equally to the three principal moments (approximation)
        dists = np.linalg.norm(self._orig["ipos"] - new_ipos), np.linalg.norm(pack_local - new_ipos)
        box = L * (0.4**2 + 0.3**2 + 0.6**2) / 18.0
        extra = box + (2 / 3) * (m0 * dists[0] ** 2 + L * dists[1] ** 2)
        m.body_mass[t] = m0 + L
        m.body_ipos[t] = new_ipos
        m.body_inertia[t] = self._orig["inertia"] + extra

    # ------------------------------------------------------------------- api
    def reset(self, p: PatchParams, hf: Heightfield) -> np.ndarray:
        obs, _ = self.env.reset(seed=int(p.seed) % (2**31 - 1))
        self.m.opt.gravity[:] = [0.0, 0.0, -float(p.gravity)]
        self._apply_terrain(hf, p.friction)
        self._apply_pack(p)
        mujoco.mj_setConst(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)
        self.hf, self.params = hf, p
        return obs

    def step(self, action):
        return self.env.step(action)

    # -------------------------------------------------------------- helpers
    def total_mass(self) -> float:
        return float(self.m.body_mass.sum())

    def com(self) -> np.ndarray:
        mass = self.m.body_mass[:, None]
        return (mass * self.d.xipos).sum(0) / mass.sum()

    def to_s(self, world_y: float) -> float:
        return self.forward * world_y

    def ground_z(self, s: float, x: float) -> float:
        hf = self.hf
        i = np.clip(np.searchsorted(hf.s, s), 1, len(hf.s) - 1)
        j = np.clip(np.searchsorted(hf.x, x), 1, len(hf.x) - 1)
        return float(hf.z[i, j])

    def foot_contacts(self) -> dict:
        """Normal contact force (N) between each foot and the terrain."""
        m, d = self.m, self.d
        out = {"r": 0.0, "l": 0.0}
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            if self.tid not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == self.tid else c.geom1
            b = int(m.geom_bodyid[other])
            for side, ids in self.foot_body_ids.items():
                if b in ids:
                    mujoco.mj_contactForce(m, d, i, f6)
                    out[side] += abs(float(f6[0]))
        return out

    def foot_positions(self) -> dict:
        d, m = self.d, self.m
        return {side: d.xpos[m.body(names[0]).id].copy() for side, names in FOOT_BODIES.items()} | {
            side + "_toe": d.xpos[m.body(names[1]).id].copy() for side, names in FOOT_BODIES.items()
        }

    def muscle_power(self) -> np.ndarray:
        """Mechanical power of each muscle (W); positive = shortening work."""
        d = self.d
        return (d.actuator_force * d.actuator_velocity)[self.muscle_mask]
