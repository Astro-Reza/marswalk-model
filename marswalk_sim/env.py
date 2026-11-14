"""MyoSuite myoLeg walker on a Mars terrain patch.

Wraps a MyoSuite leg-walking env and, on every reset, overwrites:
  * the terrain heightfield with our patch,
  * gravity,
  * terrain friction,
  * torso mass / COM / inertia to carry a suit + PLSS pack.

MyoSuite measures heights from z = 0: the fall check (centre of mass below
0.8 m) and three observation entries (pelvis height, centre-of-mass height,
foot heights). On a slope that calls a walker fallen as soon as the ground has
dropped 0.2 m, never calls it fallen once the ground has risen, and feeds the
policy heights it never saw in training. This wrapper measures all of them from
the terrain under the body instead. On flat ground nothing changes.

Any walker backend (see kinesis_backend/) exposes the same interface:
reset(p, hf), step(a), dt, frame, nrow, ncol, human_mass, total_mass(), com(),
foot_contacts(), foot_positions(), muscle_power(), act(), ground_z(s, lat).
Optional: fall_reason() -> str, used to label why a rollout terminated.

MARSWALK_ARENA_MB caps MuJoCo's contact arena for the MyoLeg model. The model
asks for several GB by default, which fails to allocate on a small or busy
Windows machine; 256 is plenty for walking.
"""
from __future__ import annotations

import contextlib
import os
import warnings

import mujoco
import numpy as np

from . import config as C
from .placement import Frame, ground_z, place_heightfield, set_const_keep_state
from .terrain import Heightfield, PatchParams

warnings.filterwarnings("ignore", category=UserWarning)

FOOT_BODIES = {"r": ("calcn_r", "toes_r"), "l": ("calcn_l", "toes_l")}

# MyoSuite's walk task rewards a positive com_vel[1], and com_vel is the
# negated MuJoCo cvel, so the walker heads toward world -y. calibrate() checks
# this with the real policy and overrides it if needed.
DEFAULT_FORWARD_SIGN = -1


@contextlib.contextmanager
def _arena_cap(megabytes: int):
    """Compile MyoSuite models with a fixed arena size (see MARSWALK_ARENA_MB)."""
    if megabytes <= 0:
        yield
        return
    from myosuite.core.model_builder import ModelBuilder

    orig = ModelBuilder.build_spec

    def build_spec(self):
        spec = orig(self)
        spec.memory = megabytes << 20
        return spec

    ModelBuilder.build_spec = build_spec
    try:
        yield
    finally:
        ModelBuilder.build_spec = orig


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

        with _arena_cap(int(os.environ.get("MARSWALK_ARENA_MB", "0") or 0)):
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
        self.init_joint_noise = C.INIT_JOINT_NOISE_RAD
        self._measure_from_terrain()

    def _ground_under(self, xyz) -> float:
        if self.hf is None:
            return 0.0
        return ground_z(self.hf, float(self.frame.s_of(xyz)), float(self.frame.lat_of(xyz)))

    def _measure_from_terrain(self) -> None:
        """Make MyoSuite's fall check and height observations relative to the terrain."""
        uw, d = self.uw, self.d
        raw_height, raw_feet, raw_obs = uw._get_height, uw._get_feet_heights, uw._get_obs_dict
        talus = [self.m.body("talus_l").id, self.m.body("talus_r").id]  # same order as MyoSuite

        def height() -> float:
            return raw_height() - self._ground_under(uw._get_com())

        def feet_heights() -> np.ndarray:
            return raw_feet() - np.array([self._ground_under(d.xpos[i]) for i in talus])

        def obs_dict(accessor) -> dict:
            obs = raw_obs(accessor)  # picks up height() and feet_heights() above
            obs["qpos_without_xy"][0] -= self._ground_under(d.qpos[:3])  # pelvis height
            return obs

        self._raw_height = raw_height
        uw._get_height, uw._get_feet_heights, uw._get_obs_dict = height, feet_heights, obs_dict

    def fall_reason(self) -> str:
        """Which MyoSuite termination rule holds right now ('' if none)."""
        uw = self.uw
        if uw._get_height() < uw.min_height:
            return "height"
        if uw._get_rot_condition():
            return "turned"     # heading more than ~55 deg off the walking axis
        if hasattr(uw, "_get_knee_condition") and uw._get_knee_condition():
            return "crouch"     # centre of mass less than 0.61 m above the feet
        return ""

    # the walking direction along y; generate.calibrate() may flip it
    @property
    def forward(self) -> int:
        return self.frame.sign

    @forward.setter
    def forward(self, sign: int) -> None:
        self.frame = Frame((0.0, 0.0), axis=1, sign=int(sign))

    def reset(self, p: PatchParams, hf: Heightfield) -> np.ndarray:
        uw = self.uw
        self.hf = None  # MyoSuite resets onto its own flat terrain at z = 0
        self.env.reset(seed=int(p.seed) % (2**31 - 1))
        if self.init_joint_noise > 0:
            rng = np.random.default_rng([int(p.seed) % (2**31 - 1), 7])
            self.d.qpos[7:] += rng.normal(0.0, self.init_joint_noise, size=self.m.nq - 7)
            uw._project_joint_couplings()
        self.m.opt.gravity[:] = [0.0, 0.0, -float(p.gravity)]
        place_heightfield(self.m, self.d, self.tid, self.hid, hf, self.frame, p.friction)
        apply_pack(self.m, self.d, self.pack_body, self._orig, p, self.frame)
        set_const_keep_state(self.m, self.d)
        self.hf, self.params = hf, p
        # the observation MyoSuite returned was taken before the patch was applied
        obs = uw._obs_dict_to_vec(uw._get_obs_dict(uw._accessor))
        return uw._ensure_obs_gymnasium_compliant(obs)

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
