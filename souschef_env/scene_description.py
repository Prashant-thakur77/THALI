"""Structured scene description for the planner: object poses and held-state as text + dict.

The VLM planner (planner/plan.py) gets the overhead frame *and* this text, so it
never has to guess a coordinate from pixels.  Positions are in cm in the table
frame (x toward arm B, y toward the cabinet), rounded so the text is stable.
"""

from __future__ import annotations

import json

import mujoco
import numpy as np

from souschef_env import constants as C
from souschef_env import oracles


def _cm(v: float) -> float:
    return float(np.round(v * 100, 1))


def scene_state(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    objs = {}
    for name in C.OBJECTS:
        p = oracles.object_pos(model, data, name)
        objs[name] = {
            "x_cm": _cm(p[0]),
            "y_cm": _cm(p[1]),
            "z_cm": _cm(p[2]),
            "held_by": oracles.held_by(model, data, name),
            "in_drawer": name in C.CUTLERY and p[1] > C.CABINET_POS[1] - 0.12 and p[2] < 0.04,
        }
    arms = {}
    for arm in C.ARMS:
        p = C.ARM_PREFIX[arm]
        site = data.site_xpos[model.site(p + "gripperframe").id]
        jaw = float(data.qpos[model.jnt_qposadr[model.joint(p + "gripper").id]])
        arms[arm] = {"x_cm": _cm(site[0]), "y_cm": _cm(site[1]), "z_cm": _cm(site[2]),
                     "jaw": "open" if jaw > 0.6 else "closed", "base_x_cm": _cm(C.ARM_BASE_POS[arm][0])}
    zones = {k: {"x_cm": _cm(v[0][0]), "y_cm": _cm(v[0][1]), "radius_cm": _cm(v[1])} for k, v in C.ZONES.items()}
    return {
        "drawer": {"open": oracles.drawer_open(model, data), "travel_cm": _cm(oracles.drawer_qpos(model, data))},
        "objects": objs,
        "arms": arms,
        "zones": zones,
        "water_in_mug": oracles.water_in_mug(model, data),
        "subgoals": oracles.subgoals(model, data),
    }


def describe(model: mujoco.MjModel, data: mujoco.MjData) -> str:
    s = scene_state(model, data)
    lines = [
        "Table frame: x runs from arm A (x=-24cm) to arm B (x=+24cm); +y is toward the cabinet/drawer.",
        f"Drawer: {'OPEN' if s['drawer']['open'] else 'CLOSED'} ({s['drawer']['travel_cm']}cm out).",
    ]
    for name, o in s["objects"].items():
        where = "in the drawer" if o["in_drawer"] else "on the table"
        held = f", held by arm {o['held_by'].upper()}" if o["held_by"] else ""
        lines.append(f"{name}: ({o['x_cm']}, {o['y_cm']})cm {where}{held}.")
    for arm, a in s["arms"].items():
        lines.append(f"Arm {arm.upper()}: gripper at ({a['x_cm']}, {a['y_cm']}, {a['z_cm']})cm, jaw {a['jaw']}.")
    lines.append("Zones: " + "; ".join(f"{k} at ({v['x_cm']}, {v['y_cm']})cm" for k, v in s["zones"].items()) + ".")
    lines.append(f"Water spheres in mug: {s['water_in_mug']}/{C.N_WATER}.")
    return "\n".join(lines)


def describe_json(model: mujoco.MjModel, data: mujoco.MjData) -> str:
    return json.dumps(scene_state(model, data), indent=None)
