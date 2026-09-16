"""Differential IK for one SO-101 arm with mink (the plan's IK source: mink examples/arm_aloha.py).

Runs on the arms-only model so the QP never sees the props' free joints.  The
arm is 5-DoF, so orientation gets a low cost and the residual is reported --
callers decide whether a pose is "reached" from ``pos_err`` / ``rot_err``.

Grasp frame convention (gripper.py): site x = approach, y = lateral (along the
grasp bar), z = jaw opening.  ``top_down_rotation(yaw)`` builds a frame with
the approach axis pointing straight down and the jaws closing across ``yaw``.
"""

from __future__ import annotations

from dataclasses import dataclass

import mink
import mujoco
import numpy as np

from souschef_env import constants as C

POS_TOL = 0.004   # m
ROT_TOL = 0.35    # rad (20 deg); the 5-DoF wrist cannot tilt laterally, so orientation is approximate on purpose


@dataclass
class IKResult:
    q: np.ndarray          # 5 joint angles for the arm
    pos_err: float
    rot_err: float
    iters: int

    @property
    def ok(self) -> bool:
        return self.pos_err <= POS_TOL and self.rot_err <= ROT_TOL


def top_down_rotation(yaw: float, tilt: float = 0.0) -> np.ndarray:
    """Site rotation matrix: approach (x) down, lateral (y) along ``yaw``, optionally tilted back toward the arm."""
    x = np.array([0.0, 0.0, -1.0])
    y = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    if tilt:
        # rotate the approach axis about the lateral axis: a tilted top-down grasp reaches further
        c, s = np.cos(tilt), np.sin(tilt)
        z0 = np.cross(x, y)
        x = c * x + s * z0
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)


class ArmIK:
    def __init__(self, model: mujoco.MjModel | None = None):
        self.model = model or mujoco.MjModel.from_xml_path(str(C.ARMS_XML))
        self.configuration = mink.Configuration(self.model)
        self.tasks = {
            a: mink.FrameTask(frame_name=f"{C.ARM_PREFIX[a]}gripperframe", frame_type="site",
                              position_cost=1.0, orientation_cost=0.05, lm_damping=1.0)
            for a in C.ARMS
        }
        self.posture = mink.PostureTask(self.model, cost=1e-3)
        self.limits = [mink.ConfigurationLimit(model=self.model)]
        self._qadr = {a: np.array([self.model.jnt_qposadr[self.model.joint(C.ARM_PREFIX[a] + j).id] for j in C.ARM_JOINTS]) for a in C.ARMS}
        self._jaw_qadr = {a: self.model.jnt_qposadr[self.model.joint(C.ARM_PREFIX[a] + "gripper").id] for a in C.ARMS}
        self.home = np.zeros(self.model.nq)
        for a in C.ARMS:
            self.home[self._qadr[a]] = C.HOME_QPOS_ARM
            self.home[self._jaw_qadr[a]] = C.GRIPPER_OPEN_Q

    def site_pose(self, arm: str, q5: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics: (site position, site rotation matrix) for arm joints ``q5``."""
        q = self.home.copy()
        q[self._qadr[arm]] = q5
        self.configuration.update(q)
        T = self.configuration.get_transform_frame_to_world(f"{C.ARM_PREFIX[arm]}gripperframe", "site")
        return T.translation().copy(), T.rotation().as_matrix().copy()

    def solve(self, arm: str, pos: np.ndarray, rot: np.ndarray | None = None, q_init: np.ndarray | None = None,
              other_q: np.ndarray | None = None, max_iters: int = 60, dt: float = 0.05, ori_cost: float = 0.05) -> IKResult:
        """Joint angles that put ``arm``'s site at ``pos`` (and ``rot`` if given), starting from ``q_init``."""
        q = self.home.copy()
        q[self._qadr[arm]] = C.HOME_QPOS_ARM if q_init is None else q_init
        other = "b" if arm == "a" else "a"
        if other_q is not None:
            q[self._qadr[other]] = other_q
        self.configuration.update(q)
        self.posture.set_target(q)
        # the other arm holds still: its task target is where it is now
        other_T = self.configuration.get_transform_frame_to_world(f"{C.ARM_PREFIX[other]}gripperframe", "site")
        self.tasks[other].set_target(other_T)
        if rot is None:
            rot = self.configuration.get_transform_frame_to_world(f"{C.ARM_PREFIX[arm]}gripperframe", "site").rotation().as_matrix()
        target = mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(np.asarray(rot, dtype=float)), np.asarray(pos, dtype=float))
        self.tasks[arm].set_target(target)
        self.tasks[arm].set_orientation_cost(ori_cost)
        tasks = [self.tasks[arm], self.tasks[other], self.posture]
        it = 0
        for it in range(1, max_iters + 1):
            vel = mink.solve_ik(self.configuration, tasks, dt, "daqp", limits=self.limits, damping=1e-4)
            self.configuration.integrate_inplace(vel, dt)
            err = self.tasks[arm].compute_error(self.configuration)
            if np.linalg.norm(err[:3]) <= POS_TOL * 0.5 and np.linalg.norm(err[3:]) <= ROT_TOL:
                break
        err = self.tasks[arm].compute_error(self.configuration)
        return IKResult(q=self.configuration.q[self._qadr[arm]].copy(), pos_err=float(np.linalg.norm(err[:3])),
                        rot_err=float(np.linalg.norm(err[3:])), iters=it)

    def solve_top_down(self, arm: str, pos: np.ndarray, yaw: float, tilt: float = 0.0, **kw) -> IKResult:
        return self.solve(arm, pos, top_down_rotation(yaw, tilt), **kw)
