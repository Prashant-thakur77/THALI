import mujoco
import numpy as np

from souschef_env import constants as C
from souschef_env import oracles
from souschef_env.scene_description import describe, scene_state


def _teleport(model, data, body, xy, z=0.005):
    q = model.jnt_qposadr[model.joint(f"{body}_free").id]
    data.qpos[q : q + 3] = [xy[0], xy[1], z]
    mujoco.mj_forward(model, data)


def test_object_in_zone_after_teleport(model, data):
    assert not oracles.object_in_zone(model, data, "plate", "plate")
    _teleport(model, data, "plate", C.ZONES["plate"][0])
    assert oracles.object_in_zone(model, data, "plate", "plate")
    assert oracles.subgoals(model, data)["plate_placed"]


def test_any_cutlery_counts(model, data):
    _teleport(model, data, "spoon_2", C.ZONES["spoon"][0], z=0.01)
    assert oracles.subgoals(model, data)["spoon_placed"]
    assert not oracles.subgoals(model, data)["fork_placed"]


def test_water_in_mug_counts_spheres_moved_into_mug(model, data):
    mug = data.xpos[model.body("mug").id]
    for i in range(8):
        q = model.jnt_qposadr[model.joint(f"water_{i}_free").id]
        data.qpos[q : q + 3] = [mug[0], mug[1], 0.015 + i * 0.006]  # all inside the 6..70 mm interior
    mujoco.mj_forward(model, data)
    assert oracles.water_in_mug(model, data) == 8
    assert oracles.poured(model, data)


def test_nothing_held_at_rest(model, data):
    for o in C.OBJECTS:
        assert oracles.held_by(model, data, o) is None
    assert oracles.mug_held(model, data) is None
    assert not oracles.task_success(model, data)


def test_scene_description_mentions_everything(model, data):
    txt = describe(model, data)
    for o in C.OBJECTS:
        assert o in txt
    assert "CLOSED" in txt and "Arm A" in txt and "Arm B" in txt
    st = scene_state(model, data)
    assert st["objects"]["fork_1"]["in_drawer"] and not st["objects"]["plate"]["in_drawer"]
    assert set(st["subgoals"]) == set(oracles.FULL_TASK)
