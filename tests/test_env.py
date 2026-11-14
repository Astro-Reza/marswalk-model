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
    w = menv.frame.world(s, x)
    pnt = np.array([w[0], w[1], 50.0])
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
            assert abs(got - want) < 0.03 + 0.17 * abs(np.tan(np.radians(slope))), (s, x, got, want)


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


def test_reset_keeps_walker_pose(menv):
    """mj_setConst must not move the walker back to qpos0 (regression)."""
    import mujoco
    from marswalk_sim.placement import set_const_keep_state

    menv.env.reset(seed=0)
    q = menv.d.qpos.copy()
    mujoco.mj_setConst(menv.m, menv.d)
    assert not np.allclose(q, menv.d.qpos)  # the raw call does disturb the pose
    menv.d.qpos[:] = q
    set_const_keep_state(menv.m, menv.d)
    assert np.allclose(q, menv.d.qpos)


def stand_at(menv, s, pelvis_height):
    """Move the walker, in its reset pose, to s metres along the patch."""
    w = menv.frame.world(s, 0.0)
    menv.d.qpos[0], menv.d.qpos[1] = w[0], w[1]
    menv.d.qpos[2] = menv.ground_z(s, 0.0) + pelvis_height
    mujoco.mj_forward(menv.m, menv.d)


@pytest.mark.parametrize("slope", [-20.0, 20.0])
def test_fall_check_follows_terrain(menv, slope):
    """MyoSuite measures the fall height from z = 0; on a slope it must be from the ground."""
    p = PatchParams(slope_deg=slope, gravity=C.G_MARS, seed=1)
    menv.reset(p, build_heightfield(p, menv.nrow, menv.ncol))
    assert abs(menv.ground_z(6.0, 0.0)) > 1.5          # the ground here is far from z = 0
    stand_at(menv, 6.0, 1.0)                           # upright, 2 m below / above the start
    assert menv.uw._get_height() == pytest.approx(1.0, abs=0.08)
    assert menv.fall_reason() != "height"
    stand_at(menv, 6.0, 0.45)                          # pelvis 45 cm above the ground: collapsed
    assert menv.uw._get_done() == 1 and menv.fall_reason() == "height"


def test_height_observations_follow_terrain(menv):
    """Pelvis, centre-of-mass and foot heights in the observation are heights above the ground."""
    uw = menv.uw
    flat = PatchParams(seed=1)
    menv.reset(flat, build_heightfield(flat, menv.nrow, menv.ncol))
    want = {k: np.array(v, dtype=float) for k, v in uw._get_obs_dict(uw._accessor).items()}
    p = PatchParams(slope_deg=0.0, cross_slope_deg=0.0, seed=1)
    hf = build_heightfield(p, menv.nrow, menv.ncol)
    hf.z += 3.0                                        # the same flat ground, 3 m higher
    menv.reset(p, hf)
    stand_at(menv, 0.0, 1.0)
    got = uw._get_obs_dict(uw._accessor)
    assert got["qpos_without_xy"][0] == pytest.approx(want["qpos_without_xy"][0], abs=1e-6)
    assert got["height"][0] == pytest.approx(want["height"][0], abs=1e-6)
    assert np.allclose(got["feet_heights"], want["feet_heights"], atol=1e-6)
    assert menv.d.qpos[2] == pytest.approx(4.0)        # only the observation changed, not the state


def test_reset_observation_matches_myosuite_on_flat_ground(menv):
    """On flat ground at z = 0 the wrapper must not change what the policy sees."""
    noise, menv.init_joint_noise = menv.init_joint_noise, 0.0
    try:
        p = PatchParams(gravity=C.G_EARTH, friction=1.0, seed=5)
        ours = menv.reset(p, build_heightfield(p, menv.nrow, menv.ncol))
        theirs, _ = menv.env.reset(seed=5)
    finally:
        menv.init_joint_noise = noise
    assert ours.shape == theirs.shape and np.allclose(ours, theirs, atol=1e-5)


def test_terrain_friction_reaches_the_feet(menv):
    """The foot geoms have friction 1.0; the patch friction must still be the one used."""
    p = PatchParams(gravity=C.G_EARTH, friction=0.4, seed=2)
    menv.reset(p, build_heightfield(p, menv.nrow, menv.ncol))
    zero = np.zeros(menv.env.action_space.shape[0], dtype=np.float32)
    used = set()
    for _ in range(40):
        menv.step(zero)
        used |= {round(float(menv.d.contact[i].friction[0]), 3) for i in range(menv.d.ncon)
                 if menv.tid in (menv.d.contact[i].geom1, menv.d.contact[i].geom2)}
    assert used == {0.4}


def test_start_pose_depends_on_seed_only(menv):
    """Repeats must differ from each other, and the same seed must give the same start."""
    def start(seed):
        p = PatchParams(seed=seed)
        menv.reset(p, build_heightfield(p, menv.nrow, menv.ncol))
        return menv.d.qpos.copy()

    a, b, a2 = start(11), start(12), start(11)
    assert np.array_equal(a, a2) and not np.allclose(a, b)
    assert np.abs(a - b)[7:].max() < 0.1 and np.array_equal(a[:7], b[:7])  # small, joints only
