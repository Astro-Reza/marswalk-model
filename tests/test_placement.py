"""Heightfield placement along every axis, and the KINESIS XML patch.

The XML test uses MyoSuite's leg model as a stand-in for the KINESIS model
(the KINESIS XML is downloaded from Hugging Face at image build time).
"""
import mujoco
import numpy as np
import pytest

from kinesis_backend.kinesis_env import TERRAIN, patch_xml, prepare_terrain_model
from marswalk_sim.placement import Frame, place_heightfield
from marswalk_sim.terrain import PatchParams, build_heightfield

TINY = """
<mujoco>
  <asset><hfield name="h" nrow="100" ncol="100" size="1 1 1 0.1"/></asset>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1"/>
    <geom name="t" type="hfield" hfield="h"/>
  </worldbody>
</mujoco>
"""


@pytest.mark.parametrize("axis,sign", [(0, -1), (0, 1), (1, -1), (1, 1)])
def test_place_along_axes(axis, sign):
    m = mujoco.MjModel.from_xml_string(TINY)
    d = mujoco.MjData(m)
    tid = m.geom("t").id
    hid = int(m.geom_dataid[tid])
    p = PatchParams(slope_deg=12.0, cross_slope_deg=5.0, rough_amp_m=0.02, seed=4)
    hf = build_heightfield(p, 100, 100)
    fr = Frame((0.6, -1.2), axis=axis, sign=sign)
    place_heightfield(m, d, tid, hid, hf, fr, friction=0.8)
    mujoco.mj_forward(m, d)  # must not move the terrain back
    for s in (0.0, 4.0, 9.0):
        for lat in (-2.0, 1.5):
            w = fr.world(s, lat)
            dist = mujoco.mj_rayHfield(m, d, tid, np.array([w[0], w[1], 50.0]), np.array([0, 0, -1.0]))
            got = 50.0 - dist
            i = np.argmin(abs(hf.s - s))
            j = np.argmin(abs(hf.x - lat))
            assert abs(got - hf.z[i, j]) < 0.12, (axis, sign, s, lat, got, hf.z[i, j])


def test_patch_xml_with_include(tmp_path):
    (tmp_path / "floor.xml").write_text(
        '<mujocoinclude><worldbody><geom name="floor" type="plane" size="20 20 0.1" condim="3"/></worldbody></mujocoinclude>')
    (tmp_path / "model.xml").write_text("""
<mujoco>
  <include file="floor.xml"/>
  <worldbody>
    <body name="pelvis" pos="0 0 1"><freejoint/><geom type="sphere" size="0.1"/>
      <body name="calcn_r"><geom type="box" size="0.1 0.05 0.02" pos="0 0 -0.9"/></body>
    </body>
  </worldbody>
</mujoco>""")
    dst = patch_xml(str(tmp_path / "model.xml"))
    m = mujoco.MjModel.from_xml_path(dst)
    info = prepare_terrain_model(m)
    tid = info["terrain_geom"]
    assert m.geom(TERRAIN).id == tid and m.hfield_nrow[info["hfield"]] == 100
    assert info["disabled_planes"] == [m.geom("floor").id]
    assert m.geom_contype[m.geom("floor").id] == 0 and m.geom_contype[tid] > 0
    # flat patch under the body: it must land on the terrain, not fall through
    d = mujoco.MjData(m)
    hf = build_heightfield(PatchParams(seed=1), 100, 100)
    place_heightfield(m, d, tid, info["hfield"], hf, Frame((0, 0), axis=0, sign=-1), friction=0.8)
    for _ in range(1000):
        mujoco.mj_step(m, d)
    assert d.qpos[2] > 0.8
