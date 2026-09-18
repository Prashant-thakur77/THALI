"""Fixed constants for the Thali dinner-table environment.

Shape follows gym-aloha's constants.py (DT/FPS, joint and action name lists,
ASSETS_DIR) but the arms are two SO-101s, so every value is ours.

World frame: the table top is the plane z = 0, so every object height reads
directly as height above the table.  Arm A sits at -x facing +x, arm B at +x
facing -x.  +y is "away from the diner" (the cabinet side).
"""

from __future__ import annotations

from pathlib import Path

DT = 0.02  # control period, s  (50 Hz, matches LeRobot/ALOHA convention)
FPS = 50
PHYSICS_TIMESTEP = 0.002  # MuJoCo integrator step; DT / PHYSICS_TIMESTEP substeps per control step
N_SUBSTEPS = int(round(DT / PHYSICS_TIMESTEP))

ASSETS_DIR = Path(__file__).parent.resolve() / "assets"
SO101_XML = ASSETS_DIR / "so101" / "so101_new_calib.xml"
SCENE_XML = ASSETS_DIR / "dinner_table.xml"
ARMS_XML = ASSETS_DIR / "arms_only.xml"  # IK model: the two arms alone, same mounts

ARMS = ("a", "b")
ARM_PREFIX = {"a": "arm_a_", "b": "arm_b_"}
# Base positions on the table top.  0.24 m each side of centre: the SO-101's
# usable top-down reach is ~0.25-0.30 m, so 0.35 m (the plan's first guess)
# leaves no shared workspace at all -- measured in reach.py, see results/reach_envelope.json.
ARM_BASE_POS = {"a": (-0.24, 0.0, 0.0), "b": (0.24, 0.0, 0.0)}
ARM_BASE_YAW = {"a": 0.0, "b": 3.141592653589793}

# The five pose joints of one SO-101, in kinematic order, then the jaw.
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER_JOINT = "gripper"
JOINTS_PER_ARM = len(ARM_JOINTS) + 1  # 6

# Observation "agent_pos" and action layout: [arm_a 5 joints, arm_a gripper, arm_b 5 joints, arm_b gripper].
# Gripper entries are normalised: 0 = closed / squeezing, 1 = fully open.
JOINTS = tuple(f"{ARM_PREFIX[a]}{j}" for a in ARMS for j in (*ARM_JOINTS, GRIPPER_JOINT))
ACTIONS = JOINTS
N_ACTIONS = len(ACTIONS)  # 12

# Ready pose: both arms raised over their own half of the table, jaws open.
# (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll) per arm.
HOME_QPOS_ARM = (0.0, -1.5, 0.4, 1.5, 0.0)  # site ~0.15 m ahead of the base, ~0.19 m up: parked arms stay clear of the shared zone (Phase 2)
GRIPPER_OPEN_Q = 1.6   # rad, near the joint's upper stop (range -0.17..1.745)
GRIPPER_CLOSED_Q = -0.1

CAMERAS = ("overhead", "wrist_a", "wrist_b")
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320

# ---- table + fixtures ------------------------------------------------------
TABLE_HALF = (0.45, 0.36, 0.02)  # half-extents of the table-top box
TABLE_Z = 0.0                     # top surface
FLOOR_Z = -0.75

# Cabinet at the far (+y) edge; the drawer slides toward -y (toward the arms).
CABINET_POS = (-0.16, 0.31, 0.0)  # toward arm A: the open drawer must not crowd the bottle grasp
DRAWER_TRAVEL = 0.12
DRAWER_OPEN_QPOS = 0.08          # oracle threshold, from the plan (drawer qpos > 0.08)

# ---- manipulable objects ---------------------------------------------------
# Every object the planner can name.  Cutlery lives in the drawer at reset.
OBJECTS = ("plate", "mug", "bottle", "fork_1", "fork_2", "spoon_1", "spoon_2")
CUTLERY = ("fork_1", "fork_2", "spoon_1", "spoon_2")
N_WATER = 20                    # free spheres inside the bottle
POURED_MIN_SPHERES = 6          # "poured" = at least this many spheres inside the mug
POUR_TARGET_SPHERES = {"little": 3, "normal": 6, "full": 12}   # "a little" / default / "fill it up" -> spheres to deliver

# Place-setting target zones on the table top (x, y) and acceptance radius (m).
# Start boxes (build_scene + randomize.PLACEMENT_HALF) never overlap a zone: plate starts at y <= -0.21, mug at y <= -0.18.
ZONES = {
    "plate": ((0.00, -0.10), 0.05),
    "fork": ((-0.13, -0.10), 0.04),   # arm A's side
    "spoon": ((0.13, -0.10), 0.04),   # arm B's side (out of A's reach: spoons are handed over)
    "mug": ((0.10, 0.04), 0.04),
}
# Where the mug is held while the other arm pours (x, y, z of the mug base).
POUR_POSE = (0.0, -0.02, 0.05)   # base 5 cm up: 2.6 cm above a placed plate's rim 8 cm away (3 cm left the mug resting on it on some layouts)  # 13 cm from the bottle (+y), clear of the plate zone (-y)


def load_model(xml_path):
    """MjModel from ``xml_path``, preferring the precompiled ``.mjb`` beside it when it is at least as new as the XML.

    Compiling the SO-101 meshes (645k faces) needs ~600 MB of transient memory; the binary model loads in ~120 MB.
    ``python -m souschef_env.build_scene`` writes both files; hosts without the .mjb fall back to the XML."""
    import os
    import mujoco
    xml_path = str(xml_path)
    mjb = os.path.splitext(xml_path)[0] + ".mjb"
    if os.path.exists(mjb) and os.path.getmtime(mjb) >= os.path.getmtime(xml_path):
        try:
            return mujoco.MjModel.from_binary_path(mjb)
        except Exception:
            pass
    return mujoco.MjModel.from_xml_path(xml_path)
