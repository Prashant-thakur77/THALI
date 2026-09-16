"""Ground-truth success checks read straight from MuJoCo state.

These are the sim oracles the plan's camera-based VLM state check is compared
against (eval/camera_vs_oracle.py).  Everything is a pure function of
(model, data) so the same code serves the env's reward, the expert's retry
logic and the eval scripts.
"""

from __future__ import annotations

import mujoco
import numpy as np

from souschef_env import constants as C

# Sub-goals of the full "set the table and pour" task, in the order the plan states them.
FULL_TASK = ("drawer_open", "plate_placed", "fork_placed", "spoon_placed", "mug_placed", "poured")


def _body_geoms(model: mujoco.MjModel, body: str) -> set[int]:
    bid = model.body(body).id
    return {g for g in range(model.ngeom) if model.geom_bodyid[g] == bid}


def _pad_ids(model: mujoco.MjModel, arm: str) -> tuple[int, int]:
    p = C.ARM_PREFIX[arm]
    return model.geom(f"{p}gripper_pad").id, model.geom(f"{p}moving_jaw_so101_v1_pad").id


def drawer_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    return float(data.qpos[model.jnt_qposadr[model.joint("drawer_slide").id]])


def drawer_open(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    return drawer_qpos(model, data) > C.DRAWER_OPEN_QPOS


def held_by(model: mujoco.MjModel, data: mujoco.MjData, obj: str) -> str | None:
    """Arm whose *both* jaw pads touch ``obj`` (or whose grasp-assist weld on it is active), else None.

    A one-pad touch is a nudge, not a grasp.  The weld only ever activates after a two-pad contact grasp
    (env._try_weld), so counting it keeps the oracle true while the carry unloads one pad.
    """
    geoms = _body_geoms(model, obj)
    for arm in C.ARMS:
        eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"weld_{arm}_{obj}")
        if eq >= 0 and data.eq_active[eq]:
            return arm
    for arm in C.ARMS:
        fixed, moving = _pad_ids(model, arm)
        touching = {fixed: False, moving: False}
        for i in range(data.ncon):
            c = data.contact[i]
            for pad in (fixed, moving):
                if (c.geom1 == pad and c.geom2 in geoms) or (c.geom2 == pad and c.geom1 in geoms):
                    touching[pad] = True
        if all(touching.values()):
            return arm
    return None


def mug_held(model: mujoco.MjModel, data: mujoco.MjData) -> str | None:
    return held_by(model, data, "mug")


def object_pos(model: mujoco.MjModel, data: mujoco.MjData, obj: str) -> np.ndarray:
    return data.xpos[model.body(obj).id].copy()


def object_in_zone(model: mujoco.MjModel, data: mujoco.MjData, obj: str, zone: str) -> bool:
    """Object rests on the table inside the zone's radius and nobody is holding it."""
    (zx, zy), radius = C.ZONES[zone]
    p = object_pos(model, data, obj)
    on_table = p[2] < 0.03
    return bool(np.hypot(p[0] - zx, p[1] - zy) < radius and on_table and held_by(model, data, obj) is None)


def any_in_zone(model: mujoco.MjModel, data: mujoco.MjData, objs: tuple[str, ...], zone: str) -> bool:
    return any(object_in_zone(model, data, o, zone) for o in objs)


def water_in_mug(model: mujoco.MjModel, data: mujoco.MjData) -> int:
    """Count of water spheres inside the mug's interior cylinder (site ``mug_inside``)."""
    sid = model.site("mug_inside").id
    centre = data.site_xpos[sid]
    R = data.site_xmat[sid].reshape(3, 3)
    radius, half_h = model.site_size[sid][0], model.site_size[sid][1]
    n = 0
    for i in range(C.N_WATER):
        p = R.T @ (data.xpos[model.body(f"water_{i}").id] - centre)
        if np.hypot(p[0], p[1]) < radius and abs(p[2]) < half_h:
            n += 1
    return n


def poured(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    return water_in_mug(model, data) >= C.POURED_MIN_SPHERES


def subgoals(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, bool]:
    return {
        "drawer_open": drawer_open(model, data),
        "plate_placed": object_in_zone(model, data, "plate", "plate"),
        "fork_placed": any_in_zone(model, data, ("fork_1", "fork_2"), "fork"),
        "spoon_placed": any_in_zone(model, data, ("spoon_1", "spoon_2"), "spoon"),
        "mug_placed": object_in_zone(model, data, "mug", "mug"),
        "poured": poured(model, data),
    }


def task_success(model: mujoco.MjModel, data: mujoco.MjData, goals: tuple[str, ...] = FULL_TASK) -> bool:
    sg = subgoals(model, data)
    return all(sg[g] for g in goals)


def skill_success(model: mujoco.MjModel, data: mujoco.MjData, skill: str, **kw) -> bool:
    """Per-skill oracle used by the expert/runtime after each skill.

    skill in {open_drawer, pick_place, handoff, hold_mug, pour}; kw carries obj/zone/arm.
    """
    if skill == "open_drawer":
        return drawer_open(model, data)
    if skill == "pick_place":
        return object_in_zone(model, data, kw["obj"], kw["zone"])
    if skill == "handoff":
        return held_by(model, data, kw["obj"]) == kw["to_arm"]
    if skill == "hold_mug":
        return mug_held(model, data) == kw["arm"]
    if skill == "pour":
        return poured(model, data)
    raise ValueError(skill)
