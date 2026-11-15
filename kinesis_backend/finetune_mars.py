#!/usr/bin/env python
"""Fine-tune the KINESIS target-reach policy for Mars gravity and a suit pack.

Same task, reward and PPO loop that KINESIS trained target-reach with (pointgoal:
walk to a random goal within 2 m, reward for getting closer, staying upright and
saving energy), started from the pretrained weights. Per episode the env draws
  gravity   : Mars (3.711) half of the time, else uniform 3.711 ... 9.81
  suit pack : suit_mass_kg, pack_up_m, pack_back_m from marswalk_sim.config.RANGES
Flat ground only (shortcut: no terrain patches during training; add them once flat Mars walks).

Runs in the KINESIS Python 3.8 env (see README.md). Checkpoints go to
<KINESIS_DIR>/data/trained_models/legs/target-reach-mars/model.pth (every 50 epochs);
use them with KINESIS_CHECKPOINT_DIR=<that folder> scripts/check_walker.py --walker kinesis.

    KINESIS_DIR=~/kinesis python kinesis_backend/finetune_mars.py --epochs 500 --threads 8
    # extra KINESIS/hydra overrides after --, e.g.  -- learning.min_batch_size=25600
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from kinesis_backend.kinesis_env import KINESIS_DIR, _StubMotionLib, _kinesis_imports  # noqa: E402
from marswalk_sim import config as C  # noqa: E402
from marswalk_sim.env import apply_pack  # noqa: E402
from marswalk_sim.placement import Frame, set_const_keep_state  # noqa: E402
from marswalk_sim.terrain import PatchParams  # noqa: E402

P_MARS = 0.5
OUT_NAME = "target-reach-mars"


def make_agent_class():
    _kinesis_imports()  # sys.path + chdir into the KINESIS repo
    from src.agents.agent_pointgoal import AgentPointGoal
    from src.env.myolegs_pointgoal import MyoLegsPointGoal

    class MarsPointGoal(MyoLegsPointGoal):
        def setup_motionlib(self):
            self.motion_lib = _StubMotionLib()
            self._sampled_motion_ids = np.array([0])
            self._motion_start_times = np.zeros(1)
            self._motion_start_times_offset = np.zeros(1)

        def sample_motions(self):
            return None

        def reset(self, seed=None, options=None):
            out = super().reset(seed=seed, options=options)
            m, d = self.mj_model, self.mj_data
            if not hasattr(self, "_pack_orig"):
                names = {m.body(i).name for i in range(m.nbody)}
                self._pack_body = m.body("torso" if "torso" in names else "pelvis").id
                self._pack_orig = {k: getattr(m, "body_" + k)[self._pack_body].copy() for k in ("mass", "ipos", "inertia")}
            g = C.G_MARS if np.random.rand() < P_MARS else np.random.uniform(C.G_MARS, C.G_EARTH)
            m.opt.gravity[:] = [0.0, 0.0, -g]
            R = C.RANGES
            p = PatchParams(suit_mass_kg=np.random.uniform(*R["suit_mass_kg"]),
                            pack_up_m=np.random.uniform(*R["pack_up_m"]),
                            pack_back_m=np.random.uniform(*R["pack_back_m"]))
            apply_pack(m, d, self._pack_body, self._pack_orig, p, Frame((0.0, 0.0), axis=0, sign=-1))
            set_const_keep_state(m, d)
            return out

    class MarsAgent(AgentPointGoal):
        def setup_env(self):
            self.env = MarsPointGoal(self.cfg)

    return MarsAgent


def main() -> None:
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=500, help="PPO epochs to run on top of the loaded checkpoint")
    ap.add_argument("--threads", type=int, default=8, help="sampling worker processes (WSL RAM limits this)")
    ap.add_argument("--from-dir", default=os.path.join(KINESIS_DIR, "data", "trained_models", "legs", "target-reach"))
    args = ap.parse_args(argv)
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")  # KINESIS logs the per-epoch line at INFO

    out_dir = os.path.join(KINESIS_DIR, "data", "trained_models", "legs", OUT_NAME)
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(os.path.join(out_dir, "model.pth")):  # first run starts from the pretrained policy
        shutil.copy(os.path.join(args.from_dir, "model.pth"), os.path.join(out_dir, "model.pth"))

    import torch
    from hydra import compose, initialize_config_dir

    MarsAgent = make_agent_class()
    with initialize_config_dir(config_dir=os.path.join(KINESIS_DIR, "cfg"), version_base=None):
        cfg = compose(config_name="config_legs", overrides=[
            f"exp_name={OUT_NAME}", "run=train_run_legs", "learning=pointgoal", "epoch=-1",
            "run.num_motions=0", f"run.num_threads={args.threads}", "no_log=True", "headless=True",
        ] + extra)
    cfg.output_dir = out_dir
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    agent = MarsAgent(cfg, torch.float32, device, training=True, checkpoint_epoch=-1)

    # self-check: one reset in the main process really changes gravity and torso mass
    agent.env.reset()
    g = -float(agent.env.mj_model.opt.gravity[2])
    assert C.G_MARS - 1e-6 <= g <= C.G_EARTH + 1e-6, g
    assert agent.env.mj_model.body_mass[agent.env._pack_body] >= agent.env._pack_orig["mass"]

    agent.cfg.learning.max_epoch = agent.epoch + args.epochs
    print(f"fine-tuning from epoch {agent.epoch} to {agent.cfg.learning.max_epoch} on {device}, "
          f"{args.threads} threads, batch {cfg.learning.min_batch_size}; checkpoints in {out_dir}", flush=True)
    agent.optimize_policy()
    agent.save_curr()


if __name__ == "__main__":
    main()
