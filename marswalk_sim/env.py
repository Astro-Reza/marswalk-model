"""MyoSuite myoLeg walker on a Mars terrain patch.

Wraps a MyoSuite leg-walking env and, on every reset, overwrites:
  * the terrain heightfield with our patch,
  * gravity,
  * terrain friction,
  * torso mass / COM / inertia to carry a suit + PLSS pack.

Any walker backend (see kinesis_backend/) exposes the same interface:
reset(p, hf), step(a), dt, frame, nrow, ncol, human_mass, total_mass(), com(),
foot_contacts(), foot_positions(), muscle_power(), act(), ground_z(s, lat).
"""
from __future__ import annotations

import warnings

import mujoco
import numpy as np

from . import config as C
from .placement import Frame, ground_z, place_heightfield
from .terrain import Heightfield, PatchParams

warnings.filterwarnings("ignore", category=UserWarning)

FOOT_BODIES = {"r": ("calcn_r", "toes_r"), "l": ("calcn_l", "toes_l")}

# MyoSuite's walk task rewards a positive com_vel[1], and com_vel is the
# negated MuJoCo cvel, so the walker heads toward world -y. calibrate() checks
# this with the real policy and overrides it if needed.
DEFAULT_FORWARD_SIGN = -1


def apply_pack(m, d, body_id: int, orig: dict, p: PatchParams, frame: Frame) -> None:
    """Add the suit/PLSS mass to one body (restoring its original values first)."""
    m.body_mass[body_id] = orig["mass"]
    m.body_ipos[body_id] = orig["ipos"]
    m.body_inertia[body_id] = orig["inertia"]
    L = float(p.suit_mass_kg)
    if L <= 0:
        return
    mujoco.mj_forward(m, d)
    up = np.array([0.0, 0.0, 1.0])
    pack_world = d.xipos[body_id] + up * p.pack_up_m - frame.fwd * p.pack_back_m
    R = d.xmat[body_id].reshape(3, 3)
    pack_local = R.T @ (pack_world - d.xpos[body_id])
    m0 = float(orig["mass"])
    new_ipos = (m0 * orig["ipos"] + L * pack_local) / (m0 + L)
    # box-shaped pack (0.4 x 0.3 x 0.6 m) + parallel-axis terms, added equally
    # to the three principal moments (approximation)
    d0 = np.linalg.norm(orig["ipos"] - new_ipos)
    d1 = np.linalg.norm(pack_local - new_ipos)
    extra = L * (0.4**2 + 0.3**2 + 0.6**2) / 18.0 + (2 / 3) * (m0 * d0**2 + L * d1**2)
    m.body_mass[body_id] = m0 + L
    m.body_ipos[body_id] = new_ipos
    m.body_inertia[body_id] = orig["inertia"] + extra


class MarsPatchEnv:
    name = "myoleg"

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
        self.frame = Frame((0.0, 0.0), axis=1, sign=forward_sign)

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

        self.pack_body = m.body("torso").id
        self._orig = {
            "mass": m.body_mass[self.pack_body].copy(),
            "ipos": m.body_ipos[self.pack_body].copy(),
            "inertia": m.body_inertia[self.pack_body].copy(),
        }
        self.human_mass = float(m.body_mass.sum())
        self.foot_body_ids = {side: {m.body(n).id for n in names} for side, names in FOOT_BODIES.items()}
        self.muscle_mask = m.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        self.dt = float(self.uw.dt)
        self.hf: Heightfield | None = None
        self.params: PatchParams | None = None

    # the walking direction along y; generate.calibrate() may flip it
    @property
    def forward(self) -> int:
        return self.frame.sign

    @forward.setter
    def forward(self, sign: int) -> None:
        self.frame = Frame((0.0, 0.0), axis=1, sign=int(sign))

    def reset(self, p: PatchParams, hf: Heightfield) -> np.ndarray:
        obs, _ = self.env.reset(seed=int(p.seed) % (2**31 - 1))
        self.m.opt.gravity[:] = [0.0, 0.0, -float(p.gravity)]
        place_heightfield(self.m, self.d, self.tid, self.hid, hf, self.frame, p.friction)
        apply_pack(self.m, self.d, self.pack_body, self._orig, p, self.frame)
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

    def ground_z(self, s: float, lat: float) -> float:
        return ground_z(self.hf, s, lat)

    def foot_contacts(self) -> dict:
        return foot_contacts(self.m, self.d, self.tid, self.foot_body_ids)

    def foot_positions(self) -> dict:
        return foot_positions(self.m, self.d)

    def muscle_power(self) -> np.ndarray:
        """Mechanical power of each muscle (W); positive = shortening work."""
        d = self.d
        return (d.actuator_force * d.actuator_velocity)[self.muscle_mask]

    def act(self) -> np.ndarray:
        return self.d.act


def foot_contacts(m, d, terrain_geom: int, foot_body_ids: dict) -> dict:
    """Normal contact force (N) between each foot and the terrain."""
    out = {"r": 0.0, "l": 0.0}
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if terrain_geom not in (c.geom1, c.geom2):
            continue
        other = c.geom2 if c.geom1 == terrain_geom else c.geom1
        b = int(m.geom_bodyid[other])
        for side, ids in foot_body_ids.items():
            if b in ids:
                mujoco.mj_contactForce(m, d, i, f6)
                out[side] += abs(float(f6[0]))
    return out


def foot_positions(m, d) -> dict:
    out = {}
    for side, (heel, toe) in FOOT_BODIES.items():
        out[side] = d.xpos[m.body(heel).id].copy()
        out[side + "_toe"] = d.xpos[m.body(toe).id].copy()
    return out
