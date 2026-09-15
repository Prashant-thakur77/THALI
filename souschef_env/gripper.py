"""SO-101 gripper fixes, applied while the scene is built (see build_scene.py).

Three problems with grasping in the stock SO-101 MJCF, all found by VectorForge
(sadishihab/bimanual-vla, README "The SO-101 gripper, and what it forced") and
reproduced here on our own compile:

1. **The collision hull is not the finger.**  MuJoCo collides a mesh as its
   convex hull and both jaws are concave, so the hull is a ramp tilted ~20 deg
   off the opening axis that pushes objects down instead of squeezing them.
   Fix: measure the flat inner face of each jaw from the mesh vertices and lay
   a thin box pad exactly on it; props collide with the pads, not the meshes.
2. **The tool site sits on the fixed finger**, not mid-aperture.  Callers
   (expert primitives) offset grasps by half the object width along +z of the
   site; :func:`aperture` reports the measured gap so nothing is typed in.
3. **A position servo cannot hold a grasp** -- its force goes to zero as it
   settles.  The jaw actuator becomes a torque source: +OPEN_TORQUE runs the
   jaw to its stop, GRIP_TORQUE sweeps it closed, HOLD_TORQUE is what stays on.

Site frame convention (checked in Phase 1): x = approach (out along the
fingers), y = lateral, z = jaw-opening axis, moving jaw on +z.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

OPEN_TORQUE = 1.0      # N.m, beats 0.6 N.m.s/rad damping to retract the jaw
GRIP_TORQUE = -0.8     # N.m, closing sweep
HOLD_TORQUE = -0.20    # N.m, sustained squeeze (~2.7 N per jaw)
GRIP_TRIGGER_N = 2.0   # switch GRIP -> HOLD once pad contact force exceeds this

PAD_REFERENCE_ANGLE = 0.0  # jaw angle at which the two inner faces are parallel
PAD_THICKNESS = 0.0015
PAD_FACE_TOL = 0.0005
PAD_DEPTH_WINDOW = 0.015
PAD_LATERAL_WINDOW = 0.025

FIXED_JAW_BODY = "gripper"
MOVING_JAW_BODY = "moving_jaw_so101_v1"
SITE = "gripperframe"
JAWS = ((FIXED_JAW_BODY, +1), (MOVING_JAW_BODY, -1))  # sign: which side of the aperture the face bounds


@dataclass
class JawFace:
    face: float                 # position on the opening axis, site frame
    depth: tuple[float, float]  # extent along approach axis
    lateral: tuple[float, float]
    tilt_deg: float             # fitted-plane tilt off the opening axis (a check, not an input)
    residual: float
    vertices: int


def _mesh_vertices_world(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> np.ndarray:
    pts = []
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != body_id or model.geom_dataid[gid] < 0:
            continue
        if model.geom_group[gid] != 3:  # collision meshes only
            continue
        mesh = model.geom_dataid[gid]
        a, n = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        verts = model.mesh_vert[a : a + n].astype(float)
        pts.append(data.geom_xpos[gid] + verts @ data.geom_xmat[gid].reshape(3, 3).T)
    if not pts:
        raise ValueError(f"body {body_id} has no collision mesh")
    return np.vstack(pts)


def measure_jaw_face(model: mujoco.MjModel, data: mujoco.MjData, prefix: str, body: str, sign: int) -> JawFace:
    """Fit a plane to the flat inner face of one jaw, expressed in the gripper site frame."""
    site = model.site(prefix + SITE).id
    origin = data.site_xpos[site]
    rot = data.site_xmat[site].reshape(3, 3)
    approach, lateral, opening = rot[:, 0], rot[:, 1], rot[:, 2]
    local = _mesh_vertices_world(model, data, model.body(prefix + body).id) - origin
    depth, lat, opn = local @ approach, local @ lateral, local @ opening
    keep = (np.abs(depth) < PAD_DEPTH_WINDOW) & (np.abs(lat) < PAD_LATERAL_WINDOW)
    local, depth, lat, opn = local[keep], depth[keep], lat[keep], opn[keep]
    face_at = float(opn.max() if sign > 0 else opn.min())
    on_face = np.abs(opn - face_at) < PAD_FACE_TOL
    if on_face.sum() < 3:
        raise RuntimeError(f"{prefix}{body}: no flat inner face within {PAD_FACE_TOL * 1e3:.1f} mm")
    centred = local[on_face] - local[on_face].mean(axis=0)
    normal = np.linalg.svd(centred)[2][-1]
    return JawFace(
        face=face_at,
        depth=(float(depth[on_face].min()), float(depth[on_face].max())),
        lateral=(float(lat[on_face].min()), float(lat[on_face].max())),
        tilt_deg=float(np.degrees(np.arccos(min(1.0, abs(normal @ opening))))),
        residual=float(np.abs(centred @ normal).max()),
        vertices=int(on_face.sum()),
    )


def jaw_pad_poses(model: mujoco.MjModel, prefix: str) -> dict[str, dict]:
    """Pad box pose for each jaw of one arm, in that jaw body's own frame."""
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[model.joint(prefix + "gripper").id]] = PAD_REFERENCE_ANGLE
    mujoco.mj_kinematics(model, data)
    site = model.site(prefix + SITE).id
    origin = data.site_xpos[site].copy()
    rot = data.site_xmat[site].reshape(3, 3).copy()
    out: dict[str, dict] = {}
    for body, sign in JAWS:
        face = measure_jaw_face(model, data, prefix, body, sign)
        (d0, d1), (l0, l1) = face.depth, face.lateral
        half = [(d1 - d0) / 2, (l1 - l0) / 2, PAD_THICKNESS / 2]
        centre_site = np.array([(d0 + d1) / 2, (l0 + l1) / 2, face.face - sign * PAD_THICKNESS / 2])
        world = origin + rot @ centre_site
        bid = model.body(prefix + body).id
        body_rot = data.xmat[bid].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (body_rot.T @ rot).ravel())
        out[prefix + body] = {
            "pos": (body_rot.T @ (world - data.xpos[bid])).tolist(),
            "quat": quat.tolist(),
            "size": half,
            "face": face,
        }
    return out


def aperture(model: mujoco.MjModel, prefix: str, jaw_q: float) -> float:
    """Gap between the two pad faces (m) at jaw angle ``jaw_q``, measured on the compiled model."""
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[model.joint(prefix + "gripper").id]] = jaw_q
    mujoco.mj_kinematics(model, data)
    site = model.site(prefix + SITE).id
    opening = data.site_xmat[site].reshape(3, 3)[:, 2]
    gaps = []
    for body, sign in JAWS:
        gid = model.geom(f"{prefix}{body}_pad").id
        face = data.geom_xpos[gid] + sign * opening * PAD_THICKNESS / 2
        gaps.append(float((face - data.site_xpos[site]) @ opening))
    return gaps[1] - gaps[0]


# ---------------------------------------------------------------------------
# Spec-level edits, called by build_scene.py
# ---------------------------------------------------------------------------

def add_jaw_pads(spec: mujoco.MjSpec, probe: mujoco.MjModel, prefixes: tuple[str, ...]) -> dict[str, JawFace]:
    """Add a massless pad box on the inner face of every jaw. ``probe`` is the same spec compiled once."""
    faces: dict[str, JawFace] = {}
    for prefix in prefixes:
        for body, pad in jaw_pad_poses(probe, prefix).items():
            g = spec.body(body).add_geom()
            g.name = f"{body}_pad"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = pad["size"]
            g.pos = pad["pos"]
            g.quat = pad["quat"]
            g.group = 3
            g.density = 0.0
            g.condim = 4
            g.friction = [1.5, 0.02, 0.001]
            g.contype = 3
            g.conaffinity = 3
            faces[g.name] = pad["face"]
    return faces


def torque_control_gripper(spec: mujoco.MjSpec, prefixes: tuple[str, ...]) -> None:
    """Turn each arm's jaw position servo into a direct torque actuator (ctrl in N.m)."""
    for prefix in prefixes:
        act = next(a for a in spec.actuators if a.name == prefix + "gripper")
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.gainprm = [1.0] + [0.0] * 9
        act.biastype = mujoco.mjtBias.mjBIAS_NONE
        act.biasprm = [0.0] * 10
        act.ctrlrange = list(act.forcerange)
        act.ctrllimited = 1


def isolate_grasp_contacts(spec: mujoco.MjSpec, prefixes: tuple[str, ...], prop_bodies: list[str]) -> None:
    """Props collide with the jaw pads (bit 2) and structure (bit 1) but never with the jaw meshes.

    Everything that collides gets bits 1|2 = 3.  Prop geoms drop to 2, jaw
    collision meshes drop to 1, so the only excluded pair is prop-vs-jaw-mesh.
    Pads keep 3 (added in add_jaw_pads) so they still stop on the table.
    """
    jaw_bodies = {prefix + body for prefix in prefixes for body, _ in JAWS}
    prop_set = set(prop_bodies)

    def visit(body: mujoco.MjsBody) -> None:
        for g in body.geoms:
            if g.contype == 0 and g.conaffinity == 0:
                continue
            if g.name.endswith("_pad"):
                continue
            if body.name in prop_set:
                g.contype = g.conaffinity = 2
            elif body.name in jaw_bodies and g.meshname:
                g.contype = g.conaffinity = 1
            else:
                g.contype = g.conaffinity = 3
        for child in body.bodies:
            visit(child)

    visit(spec.worldbody)
