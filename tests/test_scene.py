"""Scene compiles, has every named part the rest of the stack relies on, and is physically sane."""

import json
from pathlib import Path

import mujoco
import numpy as np

from souschef_env import constants as C
from souschef_env.gripper import aperture

ROOT = Path(__file__).resolve().parent.parent


def test_scene_loads_and_has_parts(model):
    for arm in C.ARMS:
        p = C.ARM_PREFIX[arm]
        for j in (*C.ARM_JOINTS, "gripper"):
            assert model.joint(p + j).id >= 0
            assert model.actuator(p + j).id >= 0
        assert model.site(p + "gripperframe").id >= 0
        assert model.geom(f"{p}gripper_pad").id >= 0
        assert model.geom(f"{p}moving_jaw_so101_v1_pad").id >= 0
        assert model.camera(f"wrist_{arm}").id >= 0
    assert model.camera("overhead").id >= 0
    for o in C.OBJECTS:
        assert model.joint(f"{o}_free").id >= 0
    assert model.joint("drawer_slide").id >= 0
    assert model.site("drawer_handle_site").id >= 0
    assert model.site("mug_inside").id >= 0
    assert model.nu == C.N_ACTIONS


def test_gripper_is_torque_actuator(model):
    for arm in C.ARMS:
        aid = model.actuator(C.ARM_PREFIX[arm] + "gripper").id
        assert model.actuator_gaintype[aid] == mujoco.mjtGain.mjGAIN_FIXED
        assert model.actuator_biastype[aid] == mujoco.mjtBias.mjBIAS_NONE
        assert model.actuator_gainprm[aid, 0] == 1.0


def test_jaw_pads_parallel_and_measured(model):
    """Pads lie on the measured mesh faces: 15.8 mm apart at the reference angle, <0.5 deg tilt."""
    gap = aperture(model, "arm_a_", 0.0)
    assert abs(gap - 0.0158) < 0.0005
    assert aperture(model, "arm_a_", 1.6) > 0.04  # opens wider than any prop
    report = json.loads((ROOT / "results" / "jaw_pads.json").read_text())
    for pad in report.values():
        assert pad["tilt_deg"] < 0.5
        assert pad["residual_m"] < 0.001


def test_props_settle_and_water_stays_in_bottle(model, data):
    from souschef_env import oracles
    for _ in range(500):
        mujoco.mj_step(model, data)
    for o in C.OBJECTS:
        z = data.xpos[model.body(o).id][2]
        assert -0.001 < z < 0.03, (o, z)
    bottle = data.xpos[model.body("bottle").id]
    inside = sum(np.hypot(*(data.xpos[model.body(f"water_{i}").id][:2] - bottle[:2])) < 0.02 for i in range(C.N_WATER))
    assert inside == C.N_WATER
    assert oracles.water_in_mug(model, data) == 0


def test_drawer_slides_open_under_a_pull(model, data):
    from souschef_env import oracles
    jid = model.joint("drawer_slide").id
    data.qfrc_applied[model.jnt_dofadr[jid]] = 3.0
    for _ in range(500):
        mujoco.mj_step(model, data)
    assert oracles.drawer_open(model, data)
    assert oracles.drawer_qpos(model, data) > 0.10


def test_reach_envelope_results_exist():
    r = json.loads((ROOT / "results" / "reach_envelope.json").read_text())
    assert r["handover_mode"] == "via_table"
    x, y, z = r["handoff_pose_m"]
    assert abs(x) < 0.1 and abs(y) < 0.1
    for h, e in r["overlap"].items():
        assert e["ik_feasible_cells"] > 50
