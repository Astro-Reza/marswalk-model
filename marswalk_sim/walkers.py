"""Pick a walker backend + controller by name.

    myoleg-deprl   MyoSuite myoLeg + DEP-RL pretrained policy (Python 3.11 image)
    myoleg-zero    MyoSuite myoLeg + zero muscle input (smoke tests; falls)
    myoleg-random  MyoSuite myoLeg + random muscle input (smoke tests; falls)
    kinesis        KINESIS MyoLeg + pretrained directional-control policy
                   (Python 3.8 image, see kinesis_backend/README.md)
"""
from __future__ import annotations

from . import config as C

WALKERS = ("myoleg-deprl", "myoleg-zero", "myoleg-random", "kinesis")


def make_walker(name: str, seed: int = 0):
    """Return (env, policy, rollout_kwargs)."""
    if name.startswith("myoleg-"):
        from .env import MarsPatchEnv
        from .policies import make_policy

        env = MarsPatchEnv()
        policy = make_policy(name.split("-", 1)[1], env.env, seed=seed)
        return env, policy, {"flat_start": C.FLAT_START_M}
    if name == "kinesis":
        from kinesis_backend.kinesis_env import KinesisMarsEnv, KinesisPolicy, KINESIS_FLAT_START_M

        env = KinesisMarsEnv()
        return env, KinesisPolicy(env), {"flat_start": KINESIS_FLAT_START_M}
    raise ValueError(f"unknown walker {name!r}; choose from {WALKERS}")
