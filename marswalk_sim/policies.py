"""Controllers that drive the walker.

deprl   : DEP-RL pretrained myoLegWalk policy (Schumacher et al.). It was
          trained at Earth gravity with no suit, so expect falls on Mars until
          it is fine-tuned. Weights download from Google Drive once; bake them
          into the Docker image so cloud tasks do not each download them.
zero    : all-zero muscle commands (falls; for smoke tests)
random  : random muscle commands (falls; for smoke tests)
"""
from __future__ import annotations

import os

import numpy as np

DEPRL_URLS = {
    "model": "https://drive.google.com/uc?id=1UkiUozk-PM8JbQCZNoy2Jr2t1StLcAzx",
    "config": "https://drive.google.com/uc?id=1knsol05ZL14aqyuaT-TlOvahr31gKQwE",
}
DEFAULT_DEPRL_DIR = os.environ.get(
    "DEPRL_BASELINE_DIR", os.path.join(os.getcwd(), "baselines_DEPRL", "myoLegWalk_20230514", "myoLeg")
)


def ensure_deprl_weights(folder: str = DEFAULT_DEPRL_DIR) -> str:
    folder = os.path.join(folder, "")  # deprl expects a trailing separator
    ckpt = os.path.join(folder, "checkpoints", "step_150000000.pt")
    cfg = os.path.join(folder, "config.yaml")
    if not (os.path.exists(ckpt) and os.path.exists(cfg)):
        import gdown

        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        gdown.download(DEPRL_URLS["model"], ckpt, quiet=False)
        gdown.download(DEPRL_URLS["config"], cfg, quiet=False)
    return folder


class Policy:
    name = "base"

    def reset(self) -> None:  # noqa: D401
        pass

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ZeroPolicy(Policy):
    name = "zero"

    def __init__(self, action_dim: int):
        self.a = np.zeros(action_dim, dtype=np.float32)

    def __call__(self, obs):
        return self.a


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, action_space, seed: int = 0):
        self.space = action_space
        self.space.seed(seed)

    def __call__(self, obs):
        return self.space.sample()


class DeprlPolicy(Policy):
    name = "deprl-myoLegWalk-20230514"

    def __init__(self, env, folder: str = DEFAULT_DEPRL_DIR):
        from deprl.utils.load_utils import load as deprl_load

        folder = ensure_deprl_weights(folder)
        self.agent = deprl_load(folder, env)

    def __call__(self, obs):
        a = self.agent.test_step(obs, muscle_states=None, steps=1e6)
        a = np.asarray(a)
        return a[0] if a.ndim > 1 else a


def make_policy(name: str, env, seed: int = 0) -> Policy:
    if name == "deprl":
        return DeprlPolicy(env)
    if name == "zero":
        return ZeroPolicy(env.action_space.shape[0])
    if name == "random":
        return RandomPolicy(env.action_space, seed)
    raise ValueError(f"unknown policy {name!r} (deprl | zero | random)")
