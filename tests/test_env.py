"""Checks that a patch is applied to the MuJoCo model correctly.

Run: python -m pytest -q tests   (needs MyoSuite installed)
"""
import mujoco
import numpy as np
import pytest

from marswalk_sim import config as C
from marswalk_sim.env import MarsPatchEnv
from marswalk_sim.terrain import PatchParams, build_heightfield


@pytest.fixture(scope="module")
def menv():
    return MarsPatchEnv()


def ray_ground(menv, s, x):
    """Height of the terrain geom under (s, x), found with a downward ray."""
    pnt = np.array([x, menv.forward * s, 50.0])
    vec = np.array([0.0, 0.0, -1.0])
    dist = mujoco.mj_rayHfield(menv.m, menv.d, menv.tid, pnt, vec)
    assert dist >= 0, "ray did not hit the terrain"
    return 50.0 - dist


@pytest.mark.parametrize("slope", [-20.0, 0.0, 15.0])
def test_heights_match(menv, slope):
    p = PatchParams(slope_deg=slope, cross_slope_deg=4.0, rough_amp_m=0.03, rock_density_per_m2=0.1,
                    gravity=C.G_MARS, seed=7)
    hf = build_heightfield(p, menv.nrow, menv.ncol)
    menv.reset(p, hf)
    for s in (3.0, 7.0, 11.0):
        for x in (-3.0, 2.5):
            want = menv.ground_z(s, x)
            got = ray_ground(menv, s, x)
            assert abs(got - want) < 0.08 + 0.2 * abs(np.tan(np.radians(slope))), (s, x, got, want)


def test_start_is_flat_and_at_zero(menv):
    p = PatchParams(slope_deg=25.0, seed=3)
    menv.reset(p, build_heightfield(p, menv.nrow, menv.ncol))
    assert abs(ray_ground(menv, -1.0, 2.0)) < 1e-3


def test_floor_plane_disabled(menv):
    gid = menv.m.geom("floor").id
    assert menv.m.geom_contype[gid] == 0 and menv.m.geom_conaffinity[gid] == 0


def test_gravity_and_pack(menv):
    p0 = PatchParams(suit_mass_kg=0.0, gravity=C.G_EARTH, seed=1)
    menv.reset(p0, build_heightfield(p0, menv.nrow, menv.ncol))
    m0 = menv.total_mass()
    assert menv.m.opt.gravity[2] == pytest.approx(-C.G_EARTH)
    p1 = PatchParams(suit_mass_kg=40.0, gravity=C.G_MARS, pack_up_m=0.2, pack_back_m=0.2, seed=1)
    menv.reset(p1, build_heightfield(p1, menv.nrow, menv.ncol))
    assert menv.total_mass() == pytest.approx(m0 + 40.0)
    assert menv.m.opt.gravity[2] == pytest.approx(-C.G_MARS)
    # resetting back without a pack restores the original torso
    menv.reset(p0, build_heightfield(p0, menv.nrow, menv.ncol))
    assert menv.total_mass() == pytest.approx(m0)


def test_feet_touch_terrain(menv):
    p = PatchParams(gravity=C.G_EARTH, seed=2)
    obs = menv.reset(p, build_heightfield(p, menv.nrow, menv.ncol))
    zero = np.zeros(menv.env.action_space.shape[0], dtype=np.float32)
    seen = 0.0
    for _ in range(60):
        obs, *_ = menv.step(zero)
        f = menv.foot_contacts()
        seen = max(seen, f["r"] + f["l"])
    assert seen > 0.3 * menv.total_mass() * C.G_EARTH
