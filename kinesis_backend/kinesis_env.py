"""KINESIS (EPFL, ICRA 2026) MyoLeg walker on Mars terrain patches.

Uses KINESIS's pretrained *directional control* policy: the imitation policy is
given a goal ~1.3 m ahead of the pelvis every step, so it walks without any
reference motion clip. That means no AMASS/KIT data and no SMPL model are
needed; the motion library is replaced by a stub.

Changes made to the KINESIS environment (subclass, the KINESIS repo is untouched):
  * a 100 x 100 heightfield geom is injected into the model XML; the flat
    floor stops colliding and the terrain copies its contact settings
  * root height (observation and fall check) is measured from the terrain
    under the pelvis, not from z = 0, so slopes do not look like falls
  * gravity and the suit/PLSS mass are set per patch

Runs inside the KINESIS Python 3.8 environment (see kinesis_backend/README.md
and Dockerfile.kinesis). Set KINESIS_DIR to the KINESIS checkout.
"""
from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from marswalk_sim.env import FOOT_BODIES, apply_pack, foot_contacts, foot_positions
from marswalk_sim.placement import Frame, ground_z, place_heightfield, set_const_keep_state
from marswalk_sim.terrain import Heightfield, PatchParams

KINESIS_DIR = os.environ.get("KINESIS_DIR", "/opt/kinesis")
KINESIS_MODEL = os.environ.get("KINESIS_MODEL", "legs")            # legs | legs_abs | legs_back
KINESIS_CHECKPOINT_DIR = os.environ.get("KINESIS_CHECKPOINT_DIR", "")  # default: <KINESIS_DIR>/data/trained_models/<model>/target-reach
KINESIS_FLAT_START_M = 3.0   # longer run-in: the walker may turn to face the goal first
GOAL_LEAD_M = 1.265          # KINESIS default distance of the moving goal ahead of the pelvis
WALK_DIRECTION = 0.0         # radians; 0 = goal toward world -x (KINESIS convention)
HFIELD_N = 100
TERRAIN = "mars_terrain"

_CONFIGS = {
    "legs": ("config_legs", "eval_run_legs"),
    "legs_abs": ("config_legs_abs", "eval_run_legs_abs"),
    "legs_back": ("config_legs_back", "eval_run_legs_back"),
}


def patch_xml(src: str, n: int = HFIELD_N) -> str:
    """Copy the model XML with a heightfield geom added; return the new path.

    The copy is written next to the original so relative include/mesh paths
    still resolve. Repeated <asset>/<worldbody> sections are merged by MuJoCo.
    """
    tree = ET.parse(src)
    root = tree.getroot()
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "hfield", name=TERRAIN, nrow=str(n), ncol=str(n), size="8 6 1 0.1")
    wb = root.find("worldbody")
    if wb is None:
        wb = ET.SubElement(root, "worldbody")
    ET.SubElement(wb, "geom", name=TERRAIN, type="hfield", hfield=TERRAIN, pos="0 0 -10",
                  rgba="0.76 0.45 0.30 1", contype="1", conaffinity="1", condim="3")
    dst = os.path.join(os.path.dirname(os.path.abspath(src)), "_marswalk_" + os.path.basename(src))
    tree.write(dst)
    return dst


def prepare_terrain_model(m) -> dict:
    """After compile: copy floor contact settings to the terrain and disable all planes."""
    tid = m.geom(TERRAIN).id
    planes = [i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE]
    if planes:
        f = planes[0]
        m.geom_contype[tid] = m.geom_contype[f] or 1
        m.geom_conaffinity[tid] = m.geom_conaffinity[f] or 1
        m.geom_condim[tid] = m.geom_condim[f]
        m.geom_priority[tid] = m.geom_priority[f]
        m.geom_solref[tid] = m.geom_solref[f]
        m.geom_solimp[tid] = m.geom_solimp[f]
        m.geom_friction[tid] = m.geom_friction[f]
    for i in planes:
        m.geom_contype[i] = 0
        m.geom_conaffinity[i] = 0
        m.geom_rgba[i, 3] = 0.0
    return {"terrain_geom": tid, "hfield": int(m.geom_dataid[tid]), "disabled_planes": planes}


def _kinesis_imports():
    if KINESIS_DIR not in sys.path:
        sys.path.insert(0, KINESIS_DIR)
    os.chdir(KINESIS_DIR)  # KINESIS uses paths relative to its repo root
    from src.agents.agent_directional import AgentDirectional
    from src.env.myolegs_directional_control import MyoLegsDirectional
    from src.learning.learning_utils import to_test
    return AgentDirectional, MyoLegsDirectional, to_test


class _StubMotionLib:
    """Stands in for KinesisCore: directional control never reads reference motions."""

    def num_all_motions(self):
        return 1

    def load_motions(self, *args, **kwargs):
        return None

    def sample_motions(self, *args, **kwargs):
        return np.array([0])

    def get_motion_length(self, *args, **kwargs):
        return np.array([1e9])


def _make_classes():
    AgentDirectional, MyoLegsDirectional, to_test = _kinesis_imports()

    class MarsDirectional(MyoLegsDirectional):
        def __init__(self, cfg):
            self.mars_hf = None
            self.mars_frame = None
            super().__init__(cfg)

        def create_sim(self, xml_path):
            super().create_sim(patch_xml(xml_path))
            self.mars_info = prepare_terrain_model(self.mj_model)

        def setup_motionlib(self):
            self.motion_lib = _StubMotionLib()
            self._sampled_motion_ids = np.array([0])
            self._motion_start_times = np.zeros(1)
            self._motion_start_times_offset = np.zeros(1)

        def sample_motions(self):
            return None

        def reset_task(self, options=None):
            super().reset_task(options)
            self.stop = False
            self.direction = WALK_DIRECTION
            self.speed = np.array([-GOAL_LEAD_M, 0.0, 0.0])

        def _ground_under_root(self) -> float:
            if self.mars_hf is None:
                return 0.0
            root = self.mj_data.qpos[:3]
            return ground_z(self.mars_hf, self.mars_frame.s_of(root), self.mars_frame.lat_of(root))

        def compute_proprioception(self):
            obs = super().compute_proprioception()
            gz = self._ground_under_root()
            if gz != 0.0 and "root_height" in self.cfg.run.proprioceptive_inputs:
                self.proprioception["root_height"] = self.proprioception["root_height"] - gz
                obs[0] -= gz  # root_height is the first proprioceptive input
            return obs

    class MarsAgent(AgentDirectional):
        def setup_env(self):
            self.env = MarsDirectional(self.cfg)

    return MarsAgent, to_test


def load_agent():
    import torch
    from hydra import compose, initialize_config_dir

    if KINESIS_MODEL not in _CONFIGS:
        raise ValueError(f"KINESIS_MODEL must be one of {list(_CONFIGS)}")
    config_name, run_cfg = _CONFIGS[KINESIS_MODEL]
    ckpt_dir = KINESIS_CHECKPOINT_DIR or os.path.join(KINESIS_DIR, "data", "trained_models", KINESIS_MODEL, "target-reach")
    if not os.path.exists(os.path.join(ckpt_dir, "model.pth")):
        found = []
        for root, _, files in os.walk(os.path.join(KINESIS_DIR, "data", "trained_models")):
            found += [os.path.join(root, f) for f in files if f.endswith(".pth")]
        raise FileNotFoundError(
            f"No model.pth in {ckpt_dir}. Run src/utils/download_models.py in {KINESIS_DIR}, or set "
            f"KINESIS_CHECKPOINT_DIR. Checkpoints found: {found[:20]}")

    MarsAgent, to_test = _make_classes()
    with initialize_config_dir(config_dir=os.path.join(KINESIS_DIR, "cfg"), version_base=None):
        cfg = compose(config_name=config_name, overrides=[
            "exp_name=target-reach", f"run={run_cfg}", "learning=directional", "epoch=-1",
            "run.headless=True", "run.im_eval=False", "run.test=True", "run.num_motions=0",
            "no_log=True", "headless=True", "test=True",
        ])
    cfg.output_dir = ckpt_dir
    torch.set_default_dtype(torch.float32)
    agent = MarsAgent(cfg, torch.float32, torch.device("cpu"), training=True, checkpoint_epoch=-1)
    to_test(*agent.sample_modules)
    return agent


class KinesisMarsEnv:
    """Same interface as marswalk_sim.env.MarsPatchEnv."""

    name = "kinesis"

    def __init__(self):
        self.agent = load_agent()
        self.kenv = self.agent.env
        self.m = self.kenv.mj_model
        self.d = self.kenv.mj_data
        info = self.kenv.mars_info
        self.tid, self.hid = info["terrain_geom"], info["hfield"]
        self.nrow = int(self.m.hfield_nrow[self.hid])
        self.ncol = int(self.m.hfield_ncol[self.hid])
        self.dt = float(self.kenv.dt)
        self.human_mass = float(self.m.body_mass.sum())
        names = {self.m.body(i).name for i in range(self.m.nbody)}
        missing = [n for pair in FOOT_BODIES.values() for n in pair if n not in names]
        if missing:
            raise RuntimeError(f"KINESIS model lacks foot bodies {missing}; update FOOT_BODIES")
        self.foot_body_ids = {side: {self.m.body(n).id for n in pair} for side, pair in FOOT_BODIES.items()}
        self.pack_body = self.m.body("torso").id if "torso" in names else self.m.body("pelvis").id
        self._orig = {k: getattr(self.m, "body_" + k)[self.pack_body].copy() for k in ("mass", "ipos", "inertia")}
        self.muscle_mask = self.m.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        self.frame = Frame((0.0, 0.0), axis=0, sign=-1)
        self.hf: Heightfield | None = None

    def reset(self, p: PatchParams, hf: Heightfield):
        np.random.seed(int(p.seed) % (2**31 - 1))
        self.kenv.mars_hf = None  # flat during reset so the start pose is computed on z = 0
        self.kenv.reset()
        root = self.d.qpos[:3].copy()
        # direction 0: the goal sits toward world -x from the pelvis
        self.frame = Frame((root[0], root[1]), axis=0, sign=-1)
        self.m.opt.gravity[:] = [0.0, 0.0, -float(p.gravity)]
        place_heightfield(self.m, self.d, self.tid, self.hid, hf, self.frame, p.friction)
        apply_pack(self.m, self.d, self.pack_body, self._orig, p, self.frame)
        set_const_keep_state(self.m, self.d)
        self.kenv.mars_hf, self.kenv.mars_frame = hf, self.frame
        self.hf = hf
        return self.kenv.compute_observations()

    def step(self, action):
        return self.kenv.step(action)

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
        return (self.d.actuator_force * self.d.actuator_velocity)[self.muscle_mask]

    def act(self) -> np.ndarray:
        return self.d.act


class KinesisPolicy:
    name = "kinesis-directional"

    def __init__(self, kenv: KinesisMarsEnv):
        self.agent = kenv.agent

    def reset(self):
        pass

    def __call__(self, obs):
        import torch

        state = self.agent.preprocess_obs(obs)
        with torch.no_grad():
            a = self.agent.policy_net.select_action(torch.from_numpy(state).to(self.agent.dtype), True)[0].numpy()
        return self.agent.preprocess_actions(a)
