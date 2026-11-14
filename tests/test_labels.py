"""Terrain features and rollout labels, checked with a scripted walker (no MyoSuite needed).

The scripted walker glides over the patch at a known speed, with known muscle
power and step timing, so every label has a value that can be worked out by hand.
"""
import types

import numpy as np
import pytest

from marswalk_sim import __version__, config as C
from marswalk_sim.placement import Frame, ground_z
from marswalk_sim.runner import LABEL_KEYS, empty_labels, rollout
from marswalk_sim.terrain import PatchParams, build_heightfield, patch_stats, to_dtm_posts

SPEED, POWER, MASS, STRIDE = 1.2, 300.0, 75.0, 1.0


class Glider:
    """Walker stand-in: centre of mass 1 m above the ground, moving at SPEED."""
    name, dt, nrow, ncol, human_mass = "glider", 0.01, 100, 100, MASS

    def __init__(self, fall_at_s=None, speed=SPEED):
        self.frame = Frame((0.0, 0.0), axis=1, sign=-1)
        self.d = types.SimpleNamespace(time=0.0)
        self.fall_at_s, self.speed = fall_at_s, speed

    def reset(self, p, hf):
        self.hf, self.s, self.d.time, self.fallen = hf, 0.0, 0.0, False
        return np.zeros(3)

    def step(self, action):
        self.s += self.speed * self.dt
        self.d.time += self.dt
        self.fallen = self.fall_at_s is not None and self.s >= self.fall_at_s
        return np.zeros(3), 0.0, self.fallen, False, {}

    def fall_reason(self):
        return "height" if self.fallen else ""

    def total_mass(self):
        return MASS

    def ground_z(self, s, lat):
        return ground_z(self.hf, s, lat)

    def com(self):
        height = 0.3 if self.fallen else 1.0
        return self.frame.world(self.s, 0.0) + [0.0, 0.0, self.ground_z(self.s, 0.0) + height]

    def foot_contacts(self):  # right foot on the ground for the first half of each stride
        right = (self.d.time % STRIDE) < STRIDE / 2
        bw = MASS * 3.711
        return {"r": bw if right else 0.0, "l": 0.0 if right else bw}

    def foot_positions(self):
        c = self.com()
        return {"r": c + [0.1, 0, -1], "l": c + [-0.1, 0, -1], "r_toe": c + [0.1, -0.2, -1], "l_toe": c + [-0.1, -0.2, -1]}

    def muscle_power(self):
        return np.array([POWER, -POWER / 2])

    def act(self):
        return np.full(4, 0.5)


def policy():
    p = types.SimpleNamespace(name="scripted", reset=lambda: None)
    return lambda obs: np.zeros(1), p


def run(walker, **params):
    act, pol = policy()
    act.reset = pol.reset
    p = PatchParams(seed=3, **params)
    return rollout(walker, act, p)


# --------------------------------------------------------------------------- terrain features
def test_dtm_features_are_measured_at_one_metre_posts():
    """Short bumps are in the fine_* stats but mostly invisible at DTM spacing; the slope is in both."""
    p = PatchParams(slope_deg=12.0, cross_slope_deg=-4.0, rough_amp_m=0.05, rough_wavelength_m=0.3, seed=2)
    hf = build_heightfield(p, 100, 100)
    st = patch_stats(hf)
    assert st["dtm_slope_along_deg"] == pytest.approx(12.0, abs=0.3)
    assert st["dtm_slope_cross_deg"] == pytest.approx(-4.0, abs=0.3)
    assert st["fine_roughness_m"] == pytest.approx(0.05, rel=0.25)
    assert st["dtm_roughness_m"] < 0.5 * st["fine_roughness_m"]
    assert st["dtm_slope_p90_deg"] < st["fine_slope_p90_deg"]
    assert st["dtm_post_m"] == C.DTM_POST_M
    sc, xc, zc = to_dtm_posts(hf.s, hf.x, hf.z)
    assert sc[1] - sc[0] == pytest.approx(1.0, abs=0.1) and xc[1] - xc[0] == pytest.approx(1.0, abs=0.1)
    assert zc.shape == (len(sc), len(xc))


def test_long_wavelength_roughness_is_seen_by_the_dtm():
    p = PatchParams(rough_amp_m=0.05, rough_wavelength_m=2.0, seed=2)
    st = patch_stats(build_heightfield(p, 100, 100))
    assert st["dtm_roughness_m"] > 0.6 * st["fine_roughness_m"]


# --------------------------------------------------------------------------- labels
def test_labels_of_a_known_walk():
    row = run(Glider(), slope_deg=10.0, suit_mass_kg=30.0)
    assert row["status"] == "ok" and row["fell"] is False and row["fall_reason"] == ""
    assert row["sim_version"] == __version__
    assert row["speed_mps"] == pytest.approx(SPEED, rel=0.01)
    # 10 s at 1.2 m/s ends 0.5 m short of the far edge; the first 2 m are run-in
    assert row["distance_m"] == pytest.approx(SPEED * C.MAX_SIM_TIME_S - C.FLAT_START_M - C.BLEND_M, abs=0.05)
    assert not row["reached_end"] and run(Glider(speed=1.5))["reached_end"]
    # net cost = (positive work / 0.25 + negative work / 1.2) per kg per metre
    want = (POWER / C.EFF_POSITIVE + POWER / 2 / C.EFF_NEGATIVE) / (MASS * SPEED)
    assert row["cot_net_J_per_kg_m"] == pytest.approx(want, rel=0.01)
    assert row["cot_gross_J_per_kg_m"] == pytest.approx(want + C.BASAL_W_PER_KG / SPEED, rel=0.01)
    assert row["stride_time_mean_s"] == pytest.approx(STRIDE, abs=0.02)
    assert row["path_slope_deg"] == pytest.approx(10.0, abs=0.2)
    assert row["path_roughness_m"] < 0.01 and abs(row["lateral_drift_m"]) < 1e-9
    assert row["com_height_start_m"] == pytest.approx(1.0) and row["com_height_min_m"] == pytest.approx(1.0)
    assert row["dtm_slope_along_deg"] == pytest.approx(10.0, abs=0.3)
    assert set(LABEL_KEYS) <= set(row)


def test_fall_on_the_patch_and_before_it():
    row = run(Glider(fall_at_s=6.0), slope_deg=-15.0)
    assert row["status"] == "ok" and row["fell"] is True and row["fall_reason"] == "height"
    assert row["distance_m"] == pytest.approx(6.0 - (C.FLAT_START_M + C.BLEND_M), abs=0.05)
    assert row["com_height_min_m"] == pytest.approx(0.3, abs=0.01) and not row["reached_end"]
    early = run(Glider(fall_at_s=0.5), slope_deg=-15.0)
    assert early["status"] == "fell_before_patch" and np.isnan(early["cot_net_J_per_kg_m"])
    assert set(early) == set(row)                      # same columns whatever happened


def test_a_walker_that_never_arrives_is_not_a_walk():
    row = run(Glider(speed=0.05))
    assert row["status"] == "too_short" and row["fell"] is False and row["truncated"] is False


def test_empty_labels_have_every_column():
    e = empty_labels("error: boom")
    assert set(LABEL_KEYS) <= set(e) and e["status"] == "error: boom" and e["sim_version"] == __version__
