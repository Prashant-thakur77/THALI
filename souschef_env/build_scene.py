"""Compose the Thali dinner-table scene and write assets/dinner_table.xml.

Two SO-101 arms (vendored TheRobotStudio MJCF, Apache-2.0) are attached via
MjSpec with prefixes ``arm_a_`` / ``arm_b_`` -- the same pattern as VectorForge,
inzuppato and ashish-doing, chosen over ``<include>`` because attach prefixes
every name and lets us inject wrist cameras and jaw pads into the arm bodies.

Object sizes come from VectorForge's measured gripper envelope: the jaws open to
~46 mm and a top-down grasp needs its grip feature 12-88 mm above the surface.
So the plate is grasped by a thin raised rim, the mug by a 36 mm barrel, the
bottle by its body, cutlery by a 12 mm bar on a riser.  Mug and bottle are
hollow (rings of boxes) so ~20 free "water" spheres can be poured between them.

Run:  python -m souschef_env.build_scene
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

from souschef_env import constants as C
from souschef_env.gripper import add_jaw_pads, isolate_grasp_contacts, torque_control_gripper

RING_SEGMENTS = 12

# Table materials: 8 train + 2 held-out test (indices 8, 9).  Procedural textures
# so the scene stays self-contained.
TABLE_MATERIALS = [
    ("wood_light", "flat", (0.78, 0.62, 0.42), (0.78, 0.62, 0.42)),
    ("wood_dark", "flat", (0.42, 0.28, 0.16), (0.42, 0.28, 0.16)),
    ("checker_grey", "checker", (0.55, 0.55, 0.55), (0.35, 0.35, 0.35)),
    ("checker_blue", "checker", (0.35, 0.45, 0.65), (0.20, 0.25, 0.40)),
    ("cloth_white", "flat", (0.92, 0.92, 0.90), (0.92, 0.92, 0.90)),
    ("cloth_red", "checker", (0.75, 0.20, 0.18), (0.85, 0.80, 0.75)),
    ("marble", "gradient", (0.85, 0.85, 0.88), (0.60, 0.60, 0.66)),
    ("slate", "gradient", (0.25, 0.27, 0.30), (0.12, 0.13, 0.15)),
    ("test_green", "checker", (0.30, 0.55, 0.30), (0.20, 0.35, 0.20)),   # held-out
    ("test_orange", "gradient", (0.90, 0.55, 0.20), (0.60, 0.30, 0.10)),  # held-out
]
BACKDROP_MATERIALS = [  # the plan's "skyboxes": a large wall behind the table
    ("room_grey", "gradient", (0.55, 0.58, 0.62), (0.30, 0.32, 0.36)),
    ("room_warm", "gradient", (0.80, 0.70, 0.60), (0.45, 0.38, 0.30)),
    ("room_dark", "flat", (0.15, 0.15, 0.18), (0.15, 0.15, 0.18)),
]


def _add_texture_material(spec: mujoco.MjSpec, name: str, kind: str, rgb1, rgb2) -> None:
    tex = spec.add_texture()
    tex.name = f"tex_{name}"
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.builtin = {"flat": mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                   "checker": mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                   "gradient": mujoco.mjtBuiltin.mjBUILTIN_GRADIENT}[kind]
    tex.rgb1 = list(rgb1)
    tex.rgb2 = list(rgb2)
    tex.width = tex.height = 256
    mat = spec.add_material()
    mat.name = f"mat_{name}"
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    mat.texrepeat = [4, 4]
    mat.reflectance = 0.05


def _add_ring(body: mujoco.MjsBody, name: str, inner_r: float, wall: float, z0: float, height: float,
              rgba, mass_total: float) -> None:
    """Hollow cylinder approximated by RING_SEGMENTS thin boxes (so spheres can sit inside)."""
    seg = 2 * math.pi / RING_SEGMENTS
    half_w = (inner_r + wall) * math.tan(seg / 2) * 1.05
    for i in range(RING_SEGMENTS):
        ang = i * seg
        r = inner_r + wall / 2
        g = body.add_geom()
        g.name = f"{name}_wall{i}"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [wall / 2, half_w, height / 2]
        g.pos = [r * math.cos(ang), r * math.sin(ang), z0 + height / 2]
        g.quat = [math.cos(ang / 2), 0, 0, math.sin(ang / 2)]
        g.rgba = list(rgba)
        g.mass = mass_total / RING_SEGMENTS


def _prop_defaults(g: mujoco.MjsGeom) -> None:
    g.condim = 4
    g.friction = [1.0, 0.02, 0.001]
    g.solref = [0.004, 1.0]
    g.contype = g.conaffinity = 2


def add_plate(spec: mujoco.MjSpec, pos) -> None:
    b = spec.worldbody.add_body(name="plate", pos=[pos[0], pos[1], 0.004])
    b.add_freejoint(name="plate_free")
    disc = b.add_geom(name="plate_disc", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.055, 0.003, 0],
                      pos=[0, 0, 0], rgba=[0.95, 0.95, 0.92, 1], mass=0.06)
    _prop_defaults(disc)
    # Raised rim: the grasp feature.  12 mm thick, 18 mm tall: the jaws bottom out at ~8 mm, so a thinner wall cannot be gripped.
    _add_ring(b, "plate_rim", inner_r=0.043, wall=0.012, z0=0.003, height=0.018, rgba=[0.95, 0.95, 0.92, 1], mass_total=0.04)
    for g in b.geoms:
        _prop_defaults(g)


def add_mug(spec: mujoco.MjSpec, pos) -> None:
    b = spec.worldbody.add_body(name="mug", pos=[pos[0], pos[1], 0.003])
    b.add_freejoint(name="mug_free")
    base = b.add_geom(name="mug_base", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.018, 0.0025, 0],
                      pos=[0, 0, 0], rgba=[0.20, 0.42, 0.75, 1], mass=0.02)
    _add_ring(b, "mug_body", inner_r=0.015, wall=0.003, z0=0.0025, height=0.064, rgba=[0.20, 0.42, 0.75, 1], mass_total=0.05)
    # handle on local +y (grasps close across x, never onto the handle)
    for i, (f, t) in enumerate([((0, 0.018, 0.050), (0, 0.030, 0.050)),
                                ((0, 0.030, 0.050), (0, 0.030, 0.020)),
                                ((0, 0.030, 0.020), (0, 0.018, 0.020))]):
        g = b.add_geom(name=f"mug_handle{i}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.0035, 0, 0],
                       fromto=[*f, *t], rgba=[0.20, 0.42, 0.75, 1], mass=0.005)
    for g in b.geoms:
        _prop_defaults(g)
        g.solref = [0.004, 2.0]  # overdamped: catches falling water spheres instead of bouncing them out
    # non-colliding marker of the "inside the mug" volume, used by the poured oracle
    s = b.add_site(name="mug_inside", pos=[0, 0, 0.035], size=[0.015, 0.032, 0], type=mujoco.mjtGeom.mjGEOM_CYLINDER)  # (radius, half-height)
    s.rgba = [0, 0, 0, 0]


def add_bottle(spec: mujoco.MjSpec, pos) -> None:
    b = spec.worldbody.add_body(name="bottle", pos=[pos[0], pos[1], 0.003])
    b.add_freejoint(name="bottle_free")
    b.add_geom(name="bottle_base", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.018, 0.0025, 0],
               pos=[0, 0, 0], rgba=[0.85, 0.85, 0.95, 0.6], mass=0.02)
    # straight tube: a shouldered neck jams 3 mm spheres (hole/particle ratio < 4), a 30 mm bore pours at ~100 deg
    _add_ring(b, "bottle_body", inner_r=0.015, wall=0.003, z0=0.0025, height=0.090, rgba=[0.85, 0.85, 0.95, 0.6], mass_total=0.06)
    for g in b.geoms:
        _prop_defaults(g)
    for g in b.geoms:
        g.friction = [0.1, 0.005, 0.0001]
    s = b.add_site(name="bottle_spout", pos=[0, 0, 0.0925], size=[0.005, 0.005, 0.005])
    s.rgba = [0, 0, 0, 0]


def add_water(spec: mujoco.MjSpec, bottle_pos) -> None:
    """N_WATER free spheres stacked inside the bottle."""
    rng = np.random.default_rng(0)
    for i in range(C.N_WATER):
        layer, k = divmod(i, 3)
        ang = k * 2 * math.pi / 3 + layer * 0.7
        x = bottle_pos[0] + 0.0065 * math.cos(ang)
        y = bottle_pos[1] + 0.0065 * math.sin(ang)
        z = 0.011 + layer * 0.0065
        b = spec.worldbody.add_body(name=f"water_{i}", pos=[x, y, z])
        b.add_freejoint(name=f"water_{i}_free")
        g = b.add_geom(name=f"water_{i}_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.003, 0, 0],
                       rgba=[0.3, 0.6, 1.0, 0.9], mass=0.001)
        g.condim = 1
        g.friction = [0.05, 0.001, 0.0001]  # "water": nearly frictionless so it flows out of a tilted bottle
        g.solref = [0.004, 2.0]  # overdamped contacts: a sphere dropped into the mug must not bounce back out
        g.solref = [0.004, 1.0]
        g.contype = g.conaffinity = 2
        g.group = 1


def add_cutlery(spec: mujoco.MjSpec, name: str, kind: str, pos, yaw: float) -> None:
    """65 mm piece on a riser (VectorForge geometry): grasp bar 12 x 10 mm at 19-29 mm up."""
    b = spec.worldbody.add_body(name=name, pos=[pos[0], pos[1], pos[2]], quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    b.add_freejoint(name=f"{name}_free")
    rgba = [0.78, 0.79, 0.82, 1]
    b.add_geom(name=f"{name}_riser", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-0.026, 0, 0.0095], size=[0.006, 0.006, 0.0095], rgba=rgba, mass=0.004)
    b.add_geom(name=f"{name}_grasp", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-0.012, 0, 0.024], size=[0.016, 0.006, 0.005], rgba=rgba, mass=0.006)
    b.add_geom(name=f"{name}_neck", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0.004, 0, 0.020, 0.012, 0, 0.006], size=[0.003, 0, 0], rgba=rgba, mass=0.002)
    if kind == "spoon":
        b.add_geom(name=f"{name}_bowl", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, pos=[0.023, 0, 0.004], size=[0.010, 0.011, 0.004], rgba=rgba, mass=0.004)
    else:
        b.add_geom(name=f"{name}_base", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.015, 0, 0.004], size=[0.005, 0.010, 0.004], rgba=rgba, mass=0.002)
        for j, yy in enumerate((-0.0072, -0.0024, 0.0024, 0.0072)):
            b.add_geom(name=f"{name}_tine{j}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.027, yy, 0.004], size=[0.007, 0.0022, 0.003], rgba=rgba, mass=0.001)
    for g in b.geoms:
        _prop_defaults(g)


def add_cabinet_and_drawer(spec: mujoco.MjSpec) -> None:
    """Fixed cabinet shell with a sliding drawer (SentinelEdge's slide-joint drawer, sized for the SO-101).

    The drawer is an open tray: 4 mm floor, 18 mm walls, so a top-down gripper
    can reach the cutlery inside.  Its bar handle sits proud of the front face
    at 35 mm height so the jaws straddle it top-down.
    """
    cx, cy, _ = C.CABINET_POS
    cab = spec.worldbody.add_body(name="cabinet", pos=[cx, cy, 0.0])
    cab_rgba = [0.45, 0.30, 0.18, 1]
    # shell: back wall, two side walls, top -- open toward -y and the bottom is the table
    cab.add_geom(name="cabinet_back", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0.078, 0.03], size=[0.128, 0.005, 0.03], rgba=cab_rgba)
    cab.add_geom(name="cabinet_left", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-0.123, 0.0, 0.03], size=[0.005, 0.08, 0.03], rgba=cab_rgba)
    cab.add_geom(name="cabinet_right", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.123, 0.0, 0.03], size=[0.005, 0.08, 0.03], rgba=cab_rgba)
    # top recessed: its front edge sits 6 cm behind the handle so the moving jaw can swing closed above the bar
    cab.add_geom(name="cabinet_top", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0.02, 0.0655], size=[0.128, 0.063, 0.0025], rgba=cab_rgba)
    for g in cab.geoms:
        g.contype = g.conaffinity = 3

    dr = cab.add_body(name="drawer", pos=[0, 0, 0.0])
    dr.add_joint(name="drawer_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, -1, 0], range=[0, C.DRAWER_TRAVEL],
                 damping=2.0, frictionloss=0.3)
    dr_rgba = [0.60, 0.42, 0.24, 1]
    dr.add_geom(name="drawer_floor", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, 0.006], size=[0.11, 0.07, 0.002], rgba=dr_rgba, mass=0.15)
    dr.add_geom(name="drawer_wall_back", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0.068, 0.017], size=[0.11, 0.002, 0.011], rgba=dr_rgba, mass=0.02)
    dr.add_geom(name="drawer_wall_left", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-0.108, 0, 0.017], size=[0.002, 0.07, 0.011], rgba=dr_rgba, mass=0.02)
    dr.add_geom(name="drawer_wall_right", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.108, 0, 0.017], size=[0.002, 0.07, 0.011], rgba=dr_rgba, mass=0.02)
    dr.add_geom(name="drawer_front", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, -0.074, 0.032], size=[0.112, 0.004, 0.027], rgba=dr_rgba, mass=0.05)
    # handle: horizontal bar (12 mm dia -- the jaws cannot close below ~8 mm) 25 mm proud of the front face, at 35 mm height
    dr.add_geom(name="drawer_handle_post0", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[-0.02, -0.078, 0.035, -0.02, -0.100, 0.035], size=[0.003, 0, 0], rgba=[0.8, 0.8, 0.8, 1], mass=0.005)
    dr.add_geom(name="drawer_handle_post1", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0.02, -0.078, 0.035, 0.02, -0.100, 0.035], size=[0.003, 0, 0], rgba=[0.8, 0.8, 0.8, 1], mass=0.005)
    dr.add_geom(name="drawer_handle", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[-0.025, -0.100, 0.035, 0.025, -0.100, 0.035], size=[0.006, 0, 0], rgba=[0.85, 0.85, 0.85, 1], mass=0.01)
    for g in dr.geoms:
        g.contype = g.conaffinity = 3
        g.friction = [0.6, 0.01, 0.001]
    dr.add_site(name="drawer_handle_site", pos=[0, -0.100, 0.035], size=[0.004, 0.004, 0.004])


def build_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.modelname = "thali_dinner_table"
    spec.compiler.degree = False
    spec.option.timestep = C.PHYSICS_TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.noslip_iterations = 3
    spec.visual.global_.offwidth = 640
    spec.visual.global_.offheight = 480
    spec.visual.map.znear = 0.01

    for m in TABLE_MATERIALS + BACKDROP_MATERIALS:
        _add_texture_material(spec, *m)

    # lights (randomised per episode in randomize.py)
    spec.worldbody.add_light(name="key", pos=[0.3, -0.4, 1.4], dir=[-0.2, 0.3, -1], diffuse=[0.7, 0.7, 0.7])
    spec.worldbody.add_light(name="fill", pos=[-0.5, 0.5, 1.2], dir=[0.3, -0.3, -1], diffuse=[0.35, 0.35, 0.35])

    # floor, backdrop, table
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.05], pos=[0, 0, C.FLOOR_Z], rgba=[0.3, 0.3, 0.32, 1])
    spec.worldbody.add_geom(name="backdrop", type=mujoco.mjtGeom.mjGEOM_BOX, size=[3.0, 0.02, 1.5], pos=[0, 1.6, 0.75],
                            material="mat_room_grey", contype=0, conaffinity=0)
    tb = spec.worldbody.add_body(name="table", pos=[0, 0, 0])
    tb.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, -C.TABLE_HALF[2]], size=list(C.TABLE_HALF),
                material="mat_wood_light", friction=[0.9, 0.01, 0.001], condim=3)
    for lx, ly in ((-0.4, -0.3), (0.4, -0.3), (-0.4, 0.3), (0.4, 0.3)):
        tb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[lx, ly, (C.FLOOR_Z - 0.04) / 2], size=[0.02, 0.02, (0.04 - C.FLOOR_Z) / 2],
                    rgba=[0.3, 0.24, 0.18, 1], contype=0, conaffinity=0)

    # arms
    for arm in C.ARMS:
        arm_spec = mujoco.MjSpec.from_file(str(C.SO101_XML))
        yaw = C.ARM_BASE_YAW[arm]
        frame = spec.worldbody.add_frame(pos=list(C.ARM_BASE_POS[arm]), quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
        spec.attach(arm_spec, prefix=C.ARM_PREFIX[arm], frame=frame)

    # wrist cameras on the gripper body, looking along the approach axis past the fingers
    for arm in C.ARMS:
        body = spec.body(f"{C.ARM_PREFIX[arm]}gripper")
        cam = body.add_camera(name=f"wrist_{arm}")
        # gripper body frame: site is at z=-0.098 (approach along -z of the body); camera above the wrist looking down -z
        cam.pos = [0.0, -0.04, -0.02]
        cam.quat = [1, 0, 0, 0]
        cam.fovy = 70
    spec.worldbody.add_camera(name="overhead", pos=[0, -0.05, 0.95], quat=[1, 0, 0, 0], fovy=52)
    spec.worldbody.add_camera(name="front", pos=[0, -1.1, 0.6], xyaxes=[1, 0, 0, 0, 0.5, 0.87], fovy=45)

    # fixtures + props
    add_cabinet_and_drawer(spec)
    cx, cy, _ = C.CABINET_POS
    # pieces lie along y (yaw 90 deg): 65 mm long, 12 mm wide; 50 mm x-spacing clears the partially-open moving jaw
    add_cutlery(spec, "fork_1", "fork", (cx - 0.075, cy - 0.01, 0.008), math.pi / 2)
    add_cutlery(spec, "fork_2", "fork", (cx - 0.025, cy - 0.01, 0.008), math.pi / 2)
    add_cutlery(spec, "spoon_1", "spoon", (cx + 0.025, cy - 0.01, 0.008), math.pi / 2)
    add_cutlery(spec, "spoon_2", "spoon", (cx + 0.075, cy - 0.01, 0.008), math.pi / 2)
    add_plate(spec, (-0.12, -0.24))
    add_mug(spec, (0.15, -0.22))
    bottle_pos = (0.03, 0.11)  # >= 13 cm from the held mug (arm A's opening jaw must clear arm B's gripper), clear of the open drawer (x < -0.05)
    add_bottle(spec, bottle_pos)
    add_water(spec, bottle_pos)

    # grasp-assist welds (inactive; env.py activates one only after a physical two-pad grasp is confirmed)
    for arm in C.ARMS:
        for obj in C.OBJECTS:
            eq = spec.add_equality()
            eq.type = mujoco.mjtEq.mjEQ_WELD
            eq.name = f"weld_{arm}_{obj}"
            eq.objtype = mujoco.mjtObj.mjOBJ_BODY
            eq.name1 = f"{C.ARM_PREFIX[arm]}gripper"
            eq.name2 = obj
            eq.active = False
            eq.solref = [0.005, 1.0]

    # keyframe: home pose, jaws open
    key = spec.add_key(name="home")
    return spec


def _finish(spec: mujoco.MjSpec) -> tuple[mujoco.MjSpec, dict]:
    """Gripper fixes need a compiled probe model: pads are placed from measured mesh faces."""
    prefixes = tuple(C.ARM_PREFIX[a] for a in C.ARMS)
    probe = spec.compile()
    faces = add_jaw_pads(spec, probe, prefixes)
    torque_control_gripper(spec, prefixes)
    props = [*C.OBJECTS, *(f"water_{i}" for i in range(C.N_WATER))]
    isolate_grasp_contacts(spec, prefixes, props)
    report = {k: {"face_m": v.face, "depth_m": v.depth, "lateral_m": v.lateral, "tilt_deg": v.tilt_deg,
                  "residual_m": v.residual, "vertices": v.vertices} for k, v in faces.items()}
    return spec, report


def _fill_home_key(spec: mujoco.MjSpec, model: mujoco.MjModel) -> None:
    key = spec.keys[0]
    qpos = np.zeros(model.nq)
    ctrl = np.zeros(model.nu)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    qpos[:] = data.qpos  # free joints keep their body pos/quat
    for arm in C.ARMS:
        p = C.ARM_PREFIX[arm]
        for jn, q in zip(C.ARM_JOINTS, C.HOME_QPOS_ARM):
            jid = model.joint(p + jn).id
            qpos[model.jnt_qposadr[jid]] = q
            ctrl[model.actuator(p + jn).id] = q
        qpos[model.jnt_qposadr[model.joint(p + "gripper").id]] = C.GRIPPER_OPEN_Q
        ctrl[model.actuator(p + "gripper").id] = 1.0  # OPEN_TORQUE, N.m
    key.qpos = qpos.tolist()
    key.ctrl = ctrl.tolist()


def build_arms_only_spec() -> mujoco.MjSpec:
    """Just the two arms on their mounts (same frames as the full scene): the IK model in ik.py."""
    spec = mujoco.MjSpec()
    spec.modelname = "thali_arms_only"
    spec.compiler.degree = False
    for arm in C.ARMS:
        arm_spec = mujoco.MjSpec.from_file(str(C.SO101_XML))
        yaw = C.ARM_BASE_YAW[arm]
        frame = spec.worldbody.add_frame(pos=list(C.ARM_BASE_POS[arm]), quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
        spec.attach(arm_spec, prefix=C.ARM_PREFIX[arm], frame=frame)
    return spec


def build(out_path: Path = C.SCENE_XML, report_path: Path | None = None) -> mujoco.MjModel:
    spec = build_spec()
    spec, report = _finish(spec)
    model = spec.compile()
    _fill_home_key(spec, model)
    model = spec.compile()
    # meshdir relative to the written XML's directory
    spec.meshdir = "so101/assets"
    xml = spec.to_xml()
    out_path.write_text(xml)
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
    arms = build_arms_only_spec()
    arms.compile()
    arms.meshdir = "so101/assets"
    C.ARMS_XML.write_text(arms.to_xml())
    return model


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    m = build(report_path=root / "results" / "jaw_pads.json")
    print(f"wrote {C.SCENE_XML}: nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom}")
    m2 = mujoco.MjModel.from_xml_path(str(C.SCENE_XML))
    print(f"reloaded from disk OK: nbody={m2.nbody}")
    sys.exit(0)
