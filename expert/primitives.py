"""Scripted expert: mink-IK primitives that drive ThaliEnv through its action space.

Every skill is a sequence of Cartesian legs.  A leg interpolates the gripper
site pose, solves IK for each control step (warm-started), and steps the env
with the resulting joint targets -- so recorded demos contain exactly the
actions a policy has to reproduce.  Grasps follow VectorForge's findings:
top-down, jaws aligned with the grasp feature's narrow axis, the site offset by
half the feature width + clearance because the fixed jaw sits on the site.

Skills: open_drawer(arm) · pick_place(obj, arm, zone) · handoff(obj, from, to)
        · hold_mug(arm) · pour(arm)
Shared-workspace rule (plan 2.2): the handoff/pour zone is a reservation --
``Workspace`` refuses a leg that enters the zone while the other arm owns it.
"""

from __future__ import annotations

import threading

import math
from dataclasses import dataclass, field
from typing import Callable

import mujoco
import numpy as np

from souschef_env import constants as C
from souschef_env import oracles
from souschef_env.env import ThaliEnv
from souschef_env.ik import ArmIK, top_down_rotation

JAW_CLEARANCE = 0.004
OPEN_MARGIN = 0.028   # m of aperture beyond the object width when opening for a grasp
OPEN_MARGIN_NARROW = 0.022  # for features under 2 cm (cutlery bars, drawer handle): neighbours sit 5 cm apart
APERTURE_AT_ZERO, APERTURE_PER_RAD, JAW_MAX = 0.0158, 0.070, 1.745  # aperture(q) ~ 15.8 mm + 70 mm/rad (measured, Phase 1)
HOVER = 0.06          # m above the grasp point
LIFT = 0.07           # carry height above the table for the object's lowest point
TOOL_SPEED = 0.12     # m/s along a leg
MIN_LEG_T, MAX_LEG_T = 0.2, 2.0
CLOSE_T = 1.4         # s allowed for the jaw to sweep in and load
OPEN_T = 0.5
SETTLE_T = 0.3
POUR_ROLL = math.radians(132)
POUR_EXIT_ROLL = math.radians(110)  # roll at which a straight-tube bottle starts to dump (~92 deg tilt with a 30 deg-pitched axis)
POUR_LIP_Z = 0.0
MUG_GRASP_Z = 0.034   # side-grasp height above the mug base (body is 6.4 cm tall; handle spans 2-5 cm on +y)   # spout lip this far *below* the rim plane, inside the opening: spheres leave the tube at ~0.5 m/s with a
                      # sideways component and bounce off the rim when dropped onto it from above
POUR_PITCH = math.radians(30)  # shallower approach for the bottle: the roll axis is closer to horizontal, so 132 deg of roll tips it ~105 deg
SIDE_PITCH = math.radians(45)  # approach angle below horizontal for side grasps
HANDOFF_XY = None     # filled from results/reach_envelope.json by Workspace


@dataclass
class Grasp:
    pos: np.ndarray       # world grasp point (between the pads)
    yaw: float            # lateral axis direction (jaws close perpendicular to it)
    width: float          # feature width across the jaws
    tilt: float = 0.0
    side: bool = False    # side grasp: approach horizontal along +yaw direction


@dataclass
class SkillResult:
    skill: str
    ok: bool
    steps: int
    detail: dict = field(default_factory=dict)


class Interrupted(RuntimeError):
    """Raised from Expert.step when the runtime's barge-in predicate fires (voice "stop" / "other arm")."""


class Workspace:
    """Reservation of the shared zone between the arms (plan 2.2)."""

    def __init__(self, centre_xy: tuple[float, float] = (0.0, 0.0), radius: float = 0.09):
        self.centre = np.array(centre_xy)
        self.radius = radius
        self.owner: str | None = None

    def inside(self, xy: np.ndarray) -> bool:
        return bool(np.hypot(*(np.asarray(xy)[:2] - self.centre)) < self.radius)

    def reserve(self, arm: str) -> bool:
        if self.owner not in (None, arm):
            return False
        self.owner = arm
        return True

    def release(self, arm: str) -> None:
        if self.owner == arm:
            self.owner = None


class StepBarrier:
    """Lets two skill threads (one per arm) share one simulator.

    Each thread owns one arm's joint targets; ``Expert.step`` from a registered thread parks at the barrier until
    every active arm has submitted its targets for this control step, then exactly one thread advances the physics
    once with the merged 12-D action and releases the others.  A thread that finishes its skill unregisters, which
    also releases anyone waiting on it.  With fewer than two arms registered the barrier is a no-op.
    """

    def __init__(self) -> None:
        self.cv = threading.Condition()
        self.active: set[str] = set()
        self.arrived: set[str] = set()
        self.generation = 0
        self.error: BaseException | None = None

    def register(self, arm: str) -> None:
        with self.cv:
            self.active.add(arm)

    def unregister(self, arm: str, do_step: Callable[[], None]) -> None:
        with self.cv:
            self.active.discard(arm)
            self.arrived.discard(arm)
            if self.active and self.arrived >= self.active:  # the other arm was already waiting on us
                self._advance(do_step)

    def _advance(self, do_step: Callable[[], None]) -> None:
        try:
            do_step()
        except BaseException as e:  # Interrupted (barge-in) must reach every thread
            self.error = e
        self.arrived.clear()
        self.generation += 1
        self.cv.notify_all()

    def step(self, arm: str, do_step: Callable[[], None]) -> None:
        with self.cv:
            if len(self.active) < 2 or arm not in self.active:
                do_step()
                return
            self.arrived.add(arm)
            gen = self.generation
            if self.arrived >= self.active:
                self.error = None
                self._advance(do_step)
            else:
                while gen == self.generation:
                    self.cv.wait(timeout=5.0)
            if self.error is not None:
                raise self.error


class Expert:
    def __init__(self, env: ThaliEnv, on_step: Callable[[np.ndarray, dict], None] | None = None):
        self.env = env
        self.m, self.d = env.model, env.data
        self.ik = ArmIK()
        self.ik_lock = threading.Lock()      # mink Configuration is shared; two arm threads solve one at a time
        self.barrier = StepBarrier()
        self._thread_arm = threading.local()  # which arm the calling thread drives (set by begin_concurrent)
        self.on_step = on_step
        self.interrupt: Callable[[], bool] | None = None  # polled every control step; True -> Interrupted
        self.workspace = Workspace()
        self.q = {a: np.array(C.HOME_QPOS_ARM, dtype=float) for a in C.ARMS}
        self.jaw = {a: 1.0 for a in C.ARMS}
        self.steps = 0
        self.last_obs = None
        self.sync_from_env()

    # ------------------------------------------------------------ low level
    def sync_from_env(self) -> None:
        for a in C.ARMS:
            self.q[a] = self.d.qpos[self.env._arm_qadr[a]].copy()
            self.jaw[a] = 1.0 if self.env.jaw_state(a) == "open" else 0.0

    def action(self) -> np.ndarray:
        out = []
        for a in C.ARMS:
            out.extend(self.q[a])
            out.append(self.jaw[a])
        return np.asarray(out, dtype=np.float64)

    def _raw_step(self) -> None:
        if self.interrupt is not None and self.interrupt():
            raise Interrupted()
        act = self.action()
        obs, _, _, _, info = self.env.step(act)
        self.last_obs = obs
        self.steps += 1
        if self.on_step:
            self.on_step(act, obs)

    def step(self, n: int = 1) -> None:
        arm = getattr(self._thread_arm, "arm", None)
        for _ in range(n):
            if arm is None:
                self._raw_step()
            else:
                self.barrier.step(arm, self._raw_step)

    # ------------------------------------------------------------ concurrency (both arms at once)
    def begin_concurrent(self, arm: str) -> None:
        """Call at the start of a skill thread: this thread now drives ``arm`` and steps through the barrier."""
        self._thread_arm.arm = arm
        self.barrier.register(arm)

    def end_concurrent(self, arm: str) -> None:
        self.barrier.unregister(arm, self._raw_step)
        self._thread_arm.arm = None

    def hold_still(self, n: int = 1) -> None:
        """Step the sim with the current targets and the interrupt check disabled (used while paused)."""
        for _ in range(n):
            self.env.step(self.action())
            self.steps += 1

    def site(self, arm: str) -> tuple[np.ndarray, np.ndarray]:
        sid = self.m.site(C.ARM_PREFIX[arm] + "gripperframe").id
        return self.d.site_xpos[sid].copy(), self.d.site_xmat[sid].reshape(3, 3).copy()

    def move(self, arm: str, pos: np.ndarray, rot: np.ndarray | None = None, t: float | None = None,
             stop: Callable[[], bool] | None = None) -> float:
        """Straight-line leg of the site to ``pos`` (and ``rot``); returns final position error (m)."""
        p0, R0 = self.site(arm)
        pos = np.asarray(pos, dtype=float)
        if rot is None:
            rot = R0
        dist = float(np.linalg.norm(pos - p0))
        t = t if t is not None else float(np.clip(dist / TOOL_SPEED, MIN_LEG_T, MAX_LEG_T))
        n = max(1, int(round(t / C.DT)))
        q0, q1 = mujoco.MjData(self.m), None  # noqa: F841 (placeholder to keep structure simple)
        other = "b" if arm == "a" else "a"
        for k in range(1, n + 1):
            s = k / n
            s = 3 * s * s - 2 * s * s * s  # smoothstep velocity profile
            p = p0 + (pos - p0) * s
            R = _slerp_mat(R0, rot, s)
            with self.ik_lock:
                res = self.ik.solve(arm, p, R, q_init=self.q[arm], other_q=self.q[other], max_iters=15)
            self.q[arm] = res.q
            self.step()
            if stop and stop():
                break
        p_end, _ = self.site(arm)
        err = float(np.linalg.norm(p_end - pos))
        if err > 0.005 and not (stop and stop()):
            # The local per-step IK can wedge in a joint limit while the orientation slerps.  Solve the end pose
            # globally (two seeds), and blend the joints to it -- a curved finish beats a stalled arm.
            best = None
            for seed in (self.q[arm], np.array(C.HOME_QPOS_ARM)):
                with self.ik_lock:
                    r = self.ik.solve(arm, pos, rot, q_init=seed, other_q=self.q[other], max_iters=100)
                if best is None or r.pos_err + 0.02 * r.rot_err < best.pos_err + 0.02 * best.rot_err:
                    best = r
            q0 = self.q[arm].copy()
            n2 = max(1, int(0.6 / C.DT))
            for k in range(1, n2 + 1):
                s = k / n2
                self.q[arm] = q0 + (best.q - q0) * (3 * s * s - 2 * s * s * s)
                self.step()
            p_end, _ = self.site(arm)
            err = float(np.linalg.norm(p_end - pos))
        return err

    def settle(self, t: float = SETTLE_T) -> None:
        self.step(int(t / C.DT))

    def open_jaw(self, arm: str, t: float = OPEN_T, width: float | None = None) -> None:
        """Open fully, or (given an object ``width``) just wide enough: width + OPEN_MARGIN of aperture."""
        if width is None:
            self.jaw[arm] = 1.0
        else:
            margin = OPEN_MARGIN_NARROW if width < 0.02 else OPEN_MARGIN
            q = (width + margin - APERTURE_AT_ZERO) / APERTURE_PER_RAD
            self.jaw[arm] = float(np.clip(q / JAW_MAX, 0.15, 1.0))
        self.step(int(t / C.DT))

    def close_jaw(self, arm: str, t: float = CLOSE_T) -> bool:
        self.jaw[arm] = 0.0
        for _ in range(int(t / C.DT)):
            self.step()
            if self.env.jaw_state(arm) == "holding":
                self.step(int(0.2 / C.DT))
                return True
        return self.env.jaw_state(arm) == "holding"

    def park(self, arm: str) -> None:
        p, R = self.site(arm)
        with self.ik_lock:
            home_p, home_R = self.ik.site_pose(arm, np.array(C.HOME_QPOS_ARM))
        self.move(arm, np.array([p[0], p[1], max(p[2], 0.10)]), R)
        self.move(arm, home_p, home_R)
        self.workspace.release(arm)

    # ------------------------------------------------------------ grasp geometry
    def obj_pose(self, obj: str) -> tuple[np.ndarray, np.ndarray]:
        bid = self.m.body(obj).id
        return self.d.xpos[bid].copy(), self.d.xmat[bid].reshape(3, 3).copy()

    def obj_lowest_z(self, obj: str) -> float:
        """Lowest world z of the body's geoms, from each geom's AABB corners rotated into the world."""
        bid = self.m.body(obj).id
        zs = []
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
        for g in range(self.m.ngeom):
            if self.m.geom_bodyid[g] == bid:
                c, h = self.m.geom_aabb[g, :3], self.m.geom_aabb[g, 3:]
                R = self.d.geom_xmat[g].reshape(3, 3)
                pts = self.d.geom_xpos[g] + (c + corners * h) @ R.T
                zs.append(pts[:, 2].min())
        return float(min(zs))

    def grasp_for(self, obj: str, arm: str, rim_offset: float = 0.0) -> Grasp:
        """``rim_offset`` (rad) rotates the plate's rim grasp point away from the nearest-to-arm point (retries)."""
        p, R = self.obj_pose(obj)
        yaw_obj = math.atan2(R[1, 0], R[0, 0])
        base = np.array(C.ARM_BASE_POS[arm])
        if obj == "plate":
            # grasp the rim at the point nearest the arm; jaws straddle the 3 mm wall (lateral = tangent)
            rim_r = self.m.geom_size[self.m.geom("plate_rim_wall0").id][0] + 0.052 * self.m.geom_size[self.m.geom("plate_rim_wall0").id][0] / 0.0015 * 0  # radial centre ~ inner_r + wall/2
            rim_r = float(np.hypot(*self.m.geom_pos[self.m.geom("plate_rim_wall0").id][:2]))
            d = base[:2] - p[:2]
            ang = math.atan2(d[1], d[0]) + rim_offset
            pos = np.array([p[0] + rim_r * math.cos(ang), p[1] + rim_r * math.sin(ang), p[2] + 0.014])
            return Grasp(pos, ang + math.pi / 2, width=0.003)
        if obj == "mug":
            # barrel, jaws closing perpendicular to the handle (handle on local +y -> lateral along y)
            h = self.m.geom_size[self.m.geom("mug_body_wall0").id][2]
            width = 2 * float(np.hypot(*self.m.geom_pos[self.m.geom("mug_body_wall0").id][:2])) + 0.003
            return Grasp(np.array([p[0], p[1], p[2] + 0.004 + h]), yaw_obj + math.pi / 2, width=width)
        if obj == "bottle":
            h = self.m.geom_size[self.m.geom("bottle_body_wall0").id][2]
            width = 2 * float(np.hypot(*self.m.geom_pos[self.m.geom("bottle_body_wall0").id][:2])) + 0.003
            d = p[:2] - base[:2]
            return Grasp(np.array([p[0], p[1], p[2] + 0.004 + h]), math.atan2(d[1], d[0]) + math.pi / 2, width=width)
        # cutlery: the grasp bar, lateral along the piece's long (local x) axis
        g = self.m.geom(f"{obj}_grasp").id
        pos = p + R @ self.m.geom_pos[g]
        return Grasp(pos, yaw_obj, width=2 * float(self.m.geom_size[g][1]))

    def drawer_grasp(self) -> Grasp:
        sid = self.m.site("drawer_handle_site").id
        return Grasp(self.d.site_xpos[sid].copy(), 0.0, width=2 * float(self.m.geom_size[self.m.geom("drawer_handle").id][0]))

    def site_target(self, g: Grasp, flip: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Site pose that puts the grasp point between the pads: offset back along the opening axis."""
        yaw = g.yaw + (math.pi if flip else 0.0)
        R = top_down_rotation(yaw, g.tilt)
        opening = R[:, 2]
        pos = g.pos - opening * (g.width / 2 + JAW_CLEARANCE)
        return pos, R

    def _sweep_blocked(self, obj: str, site: np.ndarray, opening: np.ndarray, reach: float, z: float) -> bool:
        """Does another object sit where the moving jaw swings (segment site -> site + opening*reach)?"""
        for other in C.OBJECTS:
            if other == obj:
                continue
            p, _ = self.obj_pose(other)
            if p[2] > z + 0.06 or p[2] < z - 0.05:
                continue
            v = p[:2] - site[:2]
            along = float(v @ opening[:2])
            perp = float(abs(v[0] * opening[1] - v[1] * opening[0]))
            if -0.01 < along < reach + 0.012 and perp < 0.03:
                return True
        # the drawer's side walls (a piece near a wall must be grasped with the jaw sweeping away from it)
        if z < 0.05:
            for wall in ("drawer_wall_left", "drawer_wall_right", "drawer_wall_back", "drawer_front"):
                gid = self.m.geom(wall).id
                v = self.d.geom_xpos[gid][:2] - site[:2]
                along = float(v @ opening[:2])
                perp = float(abs(v[0] * opening[1] - v[1] * opening[0]))
                half = float(max(self.m.geom_size[gid][:2]))
                if -0.005 < along < reach + 0.015 and perp < half + 0.02 and self.d.geom_xpos[gid][2] + self.m.geom_size[gid][2] > z - 0.02:
                    return True
        return False

    def _choose_flip(self, arm: str, g: Grasp, obj: str | None = None) -> bool:
        """Of the two equivalent grasps (yaw, yaw+pi) pick the one IK reaches better whose jaw sweep is clear."""
        best, best_err = False, np.inf
        margin = OPEN_MARGIN_NARROW if g.width < 0.02 else OPEN_MARGIN
        for flip in (False, True):
            pos, R = self.site_target(g, flip)
            with self.ik_lock:
                r = self.ik.solve(arm, pos, R, q_init=self.q[arm], max_iters=40)
            err = r.pos_err + 0.05 * r.rot_err
            if obj is not None and self._sweep_blocked(obj, pos, R[:, 2], g.width + margin, g.pos[2]):
                err += 0.05
            if err < best_err:
                best, best_err = flip, err
        return best

    # ------------------------------------------------------------ skills
    def pick(self, obj: str, arm: str, miss_offset: float = 0.0, rim_offset: float = 0.0) -> SkillResult:
        """Top-down pick.  ``miss_offset`` shifts the first attempt so it misses (recovery demos); ``rim_offset`` picks
        another point on the plate's rim (a plate at the table edge cannot be gripped at the point nearest the arm)."""
        start = self.steps
        g = self.grasp_for(obj, arm, rim_offset=rim_offset)
        z0 = self.obj_pose(obj)[0][2]
        flip = self._choose_flip(arm, g, obj)
        pos, R = self.site_target(g, flip)
        if miss_offset:
            pos = pos + np.array([miss_offset, 0, 0])
        self.open_jaw(arm, width=g.width)
        p_now, _ = self.site(arm)
        self.move(arm, np.array([p_now[0], p_now[1], max(p_now[2], pos[2] + HOVER)]))
        self.move(arm, pos + np.array([0, 0, HOVER]), R)
        self.move(arm, pos, R)
        held = self.close_jaw(arm)
        # straight up, then to carry height
        self.move(arm, pos + np.array([0, 0, 0.02]), R)
        self.move(arm, pos + np.array([0, 0, HOVER]), R)
        self.settle()
        z1 = self.obj_pose(obj)[0][2]
        ok = oracles.held_by(self.m, self.d, obj) == arm and (z1 - z0) > 0.025
        return SkillResult("pick", ok, self.steps - start, {"obj": obj, "arm": arm, "rise": z1 - z0, "jaw_holding": held, "flip": flip})

    def reachable(self, arm: str, pos: np.ndarray, rot: np.ndarray | None = None, rot_tol: float = 0.5) -> bool:
        """IK converges in position *and* orientation (a 39 cm stretch 'reaches' in position with the wrist pinned
        at both limits and 40 deg of error -- the arm then drags its forearm through the table)."""
        best = None
        for seed in (self.q[arm], np.array(C.HOME_QPOS_ARM)):
            with self.ik_lock:
                r = self.ik.solve(arm, np.asarray(pos, dtype=float), rot, q_init=seed, max_iters=80)
            if best is None or r.pos_err + 0.02 * r.rot_err < best.pos_err + 0.02 * best.rot_err:
                best = r
        return best.pos_err <= 0.008 and best.rot_err <= rot_tol

    def place(self, obj: str, arm: str, xy: tuple[float, float], yaw: float | None = None, floor_z: float = 0.0,
              extra_lift: float = 0.0) -> SkillResult:
        """Set the held object down at ``xy``; ``floor_z`` is the height of the surface it lands on (drawer tray floor),
        ``extra_lift`` raises the carry so the hanging piece clears obstacles on the way (the drawer front)."""
        start = self.steps
        p_site, R = self.site(arm)
        p_obj, _ = self.obj_pose(obj)
        hold = p_obj - p_site                      # where the object hangs relative to the site
        drop = p_site[2] - self.obj_lowest_z(obj)  # how far below the site its lowest point is
        target_site_xy = np.array(xy) - hold[:2]
        carry_z = LIFT + drop + extra_lift
        R_level0 = self._levelling_rotation(obj) @ R  # a 5-DoF grasp can hold the piece up to 20 deg tilted
        # the object's final yaw is free (zones are discs), so try yaw offsets until the set-down pose is reachable
        R_level = None
        for dyaw in (0.0, 0.5, -0.5, 1.0, -1.0, 1.57, -1.57):
            c_, s_ = math.cos(dyaw), math.sin(dyaw)
            R_try = np.array([[c_, -s_, 0], [s_, c_, 0], [0, 0, 1.0]]) @ R_level0
            # a tilted set-down leaves cutlery leaning on the fixed pad, so its pose must be near-level (rot < 0.25);
            # side-grasped mug/bottle frames never converge that tightly and do not need to
            tol = 0.25 if obj in C.CUTLERY else 0.5
            if self.reachable(arm, np.array([target_site_xy[0], target_site_xy[1], floor_z + drop + 0.004]), R_try, rot_tol=tol):
                R_level = R_try
                break
        if R_level is None:
            return SkillResult("place", False, 0, {"obj": obj, "arm": arm, "reason": "unreachable", "err": 1.0})
        self.move(arm, np.array([p_site[0], p_site[1], carry_z]), R)
        self.move(arm, np.array([target_site_xy[0], target_site_xy[1], carry_z]), R_level)
        # re-measure the hang after levelling, then set down 3 mm above the table
        p_site2, _ = self.site(arm)
        hold = self.obj_pose(obj)[0] - p_site2
        drop = p_site2[2] - self.obj_lowest_z(obj)
        target_site_xy = np.array(xy) - hold[:2]
        self.move(arm, np.array([target_site_xy[0], target_site_xy[1], floor_z + drop + 0.003]), R_level)
        self.settle(0.2)
        self.open_jaw(arm, width=0.04)
        # detach: back the fixed pad away from the piece along the opening axis before rising, so a piece
        # resting against the pad is not carried back up
        p_rel, R_rel = self.site(arm)
        self.move(arm, p_rel - R_rel[:, 2] * 0.012, R_rel, t=0.4)
        if obj == "mug":
            # the handle can hook over the opening jaw: slide the gripper away from the handle before rising
            handle_dir = self.obj_pose("mug")[1][:, 1]
            self.open_jaw(arm, width=0.06)
            p_rel, R_rel = self.site(arm)
            self.move(arm, p_rel - np.array([handle_dir[0], handle_dir[1], 0.0]) * 0.03, R_rel, t=0.5)
        self.move(arm, np.array([target_site_xy[0], target_site_xy[1], floor_z + drop + HOVER]), R_level)
        self.settle()
        # a light piece can ride up on the open jaw instead of staying put: if it is still above the surface, go back down,
        # open wide, back the jaw off sideways and rise again (twice at most)
        for _ in range(2):
            if self.obj_lowest_z(obj) - floor_z > 0.02:
                self.move(arm, np.array([target_site_xy[0], target_site_xy[1], floor_z + drop + 0.006]), R_level, t=0.5)
                self.open_jaw(arm, width=0.06)
                self.settle(0.3)
                p_rel, R_rel = self.site(arm)
                self.move(arm, p_rel - R_rel[:, 2] * 0.02 + R_rel[:, 1] * 0.01, R_rel, t=0.4)
                self.move(arm, np.array([target_site_xy[0], target_site_xy[1], floor_z + drop + HOVER]), R_level)
                self.settle()
            else:
                break
        p_final = self.obj_pose(obj)[0]
        err = float(np.hypot(*(p_final[:2] - np.array(xy))))
        return SkillResult("place", err < 0.03 and oracles.held_by(self.m, self.d, obj) is None, self.steps - start,
                           {"obj": obj, "arm": arm, "err": err})

    def zone_reachable(self, arm: str, zone: str) -> bool:
        (zx, zy), _ = C.ZONES[zone]
        return self.reachable(arm, np.array([zx, zy, 0.03]), top_down_rotation(0.0))

    def zone_target(self, obj: str, zone: str) -> tuple[float, float]:
        """Where inside the zone to set ``obj`` down: the centre, nudged away from whatever already sits too close.

        The plate's randomised start can overlap the fork zone and a plate placed after the fork can land on its tines;
        cutlery is moved away from a plate within 9 cm (plate radius 5.5 cm + a fork's half-length) and the plate away
        from cutlery within 8 cm, by at most 3 cm so the piece stays inside the zone (radius 5 cm)."""
        (zx, zy), radius = C.ZONES[zone]
        tgt = np.array([zx, zy])
        others = [("plate", 0.09)] if obj in C.CUTLERY else ([(c, 0.08) for c in C.CUTLERY] if obj == "plate" else [])
        for name, clear in others:
            if name == obj:
                continue
            try:
                q = self.obj_pose(name)[0][:2]
            except Exception:
                continue
            if oracles.held_by(self.m, self.d, name) is not None:
                continue
            v = tgt - q
            dist = float(np.linalg.norm(v))
            if 1e-6 < dist < clear:
                tgt = tgt + v / dist * min(0.03, clear - dist)
        if float(np.hypot(*(tgt - np.array([zx, zy])))) > radius - 0.01:
            tgt = np.array([zx, zy]) + (tgt - np.array([zx, zy])) / np.hypot(*(tgt - np.array([zx, zy]))) * (radius - 0.01)
        return float(tgt[0]), float(tgt[1])

    def pick_place(self, obj: str, arm: str, zone: str, miss_offset: float = 0.0) -> SkillResult:
        start = self.steps
        (zx, zy) = self.zone_target(obj, zone)
        if not self.zone_reachable(arm, zone):
            return SkillResult("pick_place", False, 0, {"obj": obj, "arm": arm, "zone": zone, "reason": "zone unreachable for this arm"})
        r1 = self.pick(obj, arm, miss_offset=miss_offset)
        retried = False
        # retries: the same grasp once (a slip), then -- for the plate -- other points on the rim
        for rim in ([0.0, 0.7, -0.7] if obj == "plate" else [0.0]):
            if r1.ok:
                break
            self.open_jaw(arm)
            r1 = self.pick(obj, arm, rim_offset=rim)
            retried = True
        if not r1.ok:
            self.open_jaw(arm)
            self.park(arm)
            return SkillResult("pick_place", False, self.steps - start, {"stage": "pick", "obj": obj, "arm": arm, "retried": retried})
        r2 = self.place(obj, arm, (zx, zy))
        self.park(arm)
        ok = oracles.object_in_zone(self.m, self.d, obj, zone)
        attempts = 1
        while not ok and attempts < 3 and oracles.held_by(self.m, self.d, obj) is None and self.obj_pose(obj)[0][2] < 0.05:
            # the piece landed off-zone or was dragged: pick it up from where it is and place again
            attempts += 1
            retried = True
            r1 = self.pick(obj, arm)
            if not r1.ok:
                self.open_jaw(arm)
                self.park(arm)
                break
            r2 = self.place(obj, arm, (zx, zy))
            self.park(arm)
            ok = oracles.object_in_zone(self.m, self.d, obj, zone)
        return SkillResult("pick_place", ok, self.steps - start, {"obj": obj, "arm": arm, "zone": zone, "retried": retried, "attempts": attempts, "place_err": r2.detail["err"]})

    def open_drawer(self, arm: str = "a") -> SkillResult:
        start = self.steps
        g = self.drawer_grasp()
        # the bar runs along x. Site (fixed jaw) on the cabinet side (+y) so the moving jaw swings closed on the
        # open side, and the pull (-y) pushes the bar with the fixed pad's face rather than relying on friction.
        pos, R = self.site_target(g, flip=False)
        pos2, R2 = self.site_target(g, flip=True)
        if pos2[1] > pos[1]:
            pos, R = pos2, R2
        self.open_jaw(arm, width=g.width)
        p_now, _ = self.site(arm)
        self.move(arm, np.array([p_now[0], p_now[1], max(p_now[2], pos[2] + HOVER)]))
        self.move(arm, pos + np.array([0, 0, HOVER]), R)
        self.move(arm, pos, R)
        self.close_jaw(arm)
        self.move(arm, pos + np.array([0, -C.DRAWER_TRAVEL * 0.95, 0.0]), R, t=1.8)
        self.settle()
        self.open_jaw(arm)
        p, _ = self.site(arm)
        self.move(arm, p + np.array([0, -0.02, HOVER]), R)
        self.park(arm)
        ok = oracles.drawer_open(self.m, self.d)
        return SkillResult("open_drawer", ok, self.steps - start, {"arm": arm, "drawer_qpos": oracles.drawer_qpos(self.m, self.d)})

    # ------------------------------------------------------------ clearing the table
    def drawer_slot(self, obj: str) -> tuple[float, float]:
        """Where ``obj`` goes inside the (open) drawer tray: the spot it occupied at reset, following the slide."""
        cx, cy, _ = C.CABINET_POS
        q = oracles.drawer_qpos(self.m, self.d)
        dx = -0.075 if obj.startswith("fork") else 0.025
        return (cx + dx, cy - q - 0.01)

    def put_in_drawer(self, obj: str, arm: str) -> SkillResult:
        """Return a piece of cutlery to the open drawer: pick it (unless the arm already holds it), set it on the tray floor."""
        start = self.steps
        if not oracles.drawer_open(self.m, self.d):
            return SkillResult("put_in_drawer", False, 0, {"obj": obj, "arm": arm, "reason": "drawer closed"})
        if oracles.held_by(self.m, self.d, obj) != arm:
            r = self.pick(obj, arm)
            if not r.ok:
                self.open_jaw(arm)
                self.park(arm)
                return SkillResult("put_in_drawer", False, self.steps - start, {"obj": obj, "arm": arm, "stage": "pick", **r.detail})
        floor = float(self.m.geom_pos[self.m.geom("drawer_floor").id][2] + self.m.geom_size[self.m.geom("drawer_floor").id][2])
        # the tray front stands 6 cm above the table: carry the piece well over it or its tip shoves the drawer shut
        r = self.place(obj, arm, self.drawer_slot(obj), floor_z=floor, extra_lift=0.06)
        self.park(arm)
        ok = oracles.in_drawer(self.m, self.d, obj)
        return SkillResult("put_in_drawer", ok, self.steps - start, {"obj": obj, "arm": arm, "stage": "place", **r.detail})

    def close_drawer(self, arm: str = "a") -> SkillResult:
        """Grasp the handle bar and push the drawer home (+y); the mirror of open_drawer."""
        start = self.steps
        q0 = oracles.drawer_qpos(self.m, self.d)
        if q0 < 0.02:
            return SkillResult("close_drawer", True, 0, {"arm": arm, "drawer_qpos": q0, "already": True})
        g = self.drawer_grasp()
        pos, R = self.site_target(g, flip=False)
        pos2, R2 = self.site_target(g, flip=True)
        if pos2[1] < pos[1]:  # fixed jaw on the open (-y) side: the push (+y) goes through the fixed pad's face
            pos, R = pos2, R2
        self.open_jaw(arm, width=g.width)
        p_now, _ = self.site(arm)
        self.move(arm, np.array([p_now[0], p_now[1], max(p_now[2], pos[2] + HOVER)]))
        self.move(arm, pos + np.array([0, 0, HOVER]), R)
        self.move(arm, pos, R)
        self.close_jaw(arm)
        self.move(arm, pos + np.array([0, q0 * 0.97, 0.0]), R, t=1.8)
        self.settle()
        self.open_jaw(arm)
        p, _ = self.site(arm)
        self.move(arm, p + np.array([0, -0.03, HOVER]), R)
        self.park(arm)
        ok = oracles.drawer_closed(self.m, self.d)
        return SkillResult("close_drawer", ok, self.steps - start, {"arm": arm, "drawer_qpos": oracles.drawer_qpos(self.m, self.d)})

    def handoff(self, obj: str, from_arm: str, to_arm: str) -> SkillResult:
        """Via-table handoff: from_arm places the object at the handoff pose, to_arm picks it up."""
        start = self.steps
        hx, hy = self.workspace.centre if HANDOFF_XY is None else HANDOFF_XY
        if not self.workspace.reserve(from_arm):
            return SkillResult("handoff", False, 0, {"reason": "workspace owned by other arm"})
        if oracles.held_by(self.m, self.d, obj) != from_arm:
            r = self.pick(obj, from_arm)
            if not r.ok:
                self.workspace.release(from_arm)
                return SkillResult("handoff", False, self.steps - start, {"stage": "pick", "obj": obj})
        self.place(obj, from_arm, (hx, hy))
        self.park(from_arm)
        self.workspace.release(from_arm)
        self.workspace.reserve(to_arm)
        r = self.pick(obj, to_arm)
        ok = r.ok and oracles.held_by(self.m, self.d, obj) == to_arm
        return SkillResult("handoff", ok, self.steps - start, {"obj": obj, "from": from_arm, "to": to_arm, "handoff_xy": [hx, hy]})

    # ------------------------------------------------------------ side grasps (mug hold, bottle pour)
    def side_grasp_target(self, obj: str, arm: str, z_above_base: float, pitch: float = SIDE_PITCH,
                          lateral_down: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Diagonal grasp of a vertical cylinder: approach along the bearing from the base, ``pitch`` rad down,
        jaws closing tangentially so the object's opening stays clear.

        Returns (site_pos, site_rot, grasp_point).  The SO-101's approach axis always lies in the arm's vertical
        plane, so the approach must point along the bearing base -> object; a fully horizontal approach at
        table height is unreachable (the wrist would sit on the table), hence the 45 deg default.
        """
        p, _ = self.obj_pose(obj)
        base = np.array(C.ARM_BASE_POS[arm])
        bearing = p[:2] - base[:2]
        bearing = bearing / np.linalg.norm(bearing)
        a = np.array([bearing[0] * math.cos(pitch), bearing[1] * math.cos(pitch), -math.sin(pitch)])
        opening = np.array([bearing[1], -bearing[0], 0.0]) * (-1.0 if lateral_down else 1.0)  # tangential
        lateral = np.cross(opening, a)
        R = np.stack([a, lateral, opening], axis=1)
        wall = f"{obj}_body_wall0"
        width = 2 * float(np.hypot(*self.m.geom_pos[self.m.geom(wall).id][:2])) + 0.003
        grasp = np.array([p[0], p[1], p[2] + z_above_base])
        pos = grasp - opening * (width / 2 + JAW_CLEARANCE)
        return pos, R, grasp

    def pick_side(self, obj: str, arm: str, z_above_base: float, pitch: float = SIDE_PITCH, lateral_down: bool = False) -> SkillResult:
        start = self.steps
        pos, R, _ = self.side_grasp_target(obj, arm, z_above_base, pitch, lateral_down)
        a = R[:, 0]
        wall = f"{obj}_body_wall0"
        self.open_jaw(arm, width=2 * float(np.hypot(*self.m.geom_pos[self.m.geom(wall).id][:2])) + 0.003)
        p_now, _ = self.site(arm)
        self.move(arm, np.array([p_now[0], p_now[1], max(p_now[2], 0.12)]))
        # come straight down onto the grasp pose: a horizontal slide-in at 3 cm height sweeps the open drawer
        # front and the other arm's gripper, and a contact there forces the partially-open jaw wide open
        self.move(arm, pos + np.array([0, 0, 0.08]), R)
        self.move(arm, pos, R, t=1.0)
        self.close_jaw(arm)
        z0 = self.obj_pose(obj)[0][2]
        self.move(arm, pos + np.array([0, 0, 0.03]), R)
        self.settle()
        ok = oracles.held_by(self.m, self.d, obj) == arm and self.obj_pose(obj)[0][2] - z0 > 0.005
        return SkillResult("pick_side", ok, self.steps - start, {"obj": obj, "arm": arm, "rise": float(self.obj_pose(obj)[0][2] - z0)})

    def handoff_place(self, obj: str, from_arm: str, to_arm: str, zone: str) -> SkillResult:
        """Via-table handoff followed by the receiving arm placing the object in ``zone``."""
        start = self.steps
        r = self.handoff(obj, from_arm, to_arm)
        if not r.ok:
            return SkillResult("handoff_place", False, self.steps - start, {"stage": "handoff", **r.detail})
        (zx, zy), _ = C.ZONES[zone]
        r2 = self.place(obj, to_arm, (zx, zy))
        self.park(to_arm)
        self.workspace.release(to_arm)
        ok = oracles.object_in_zone(self.m, self.d, obj, zone)
        return SkillResult("handoff_place", ok, self.steps - start, {"obj": obj, "from": from_arm, "to": to_arm, "zone": zone, "place_err": r2.detail.get("err")})

    def hold_mug(self, arm: str = "b") -> SkillResult:
        """Side-grasp the mug (opening stays clear) and hold it steady with its base at POUR_POSE."""
        start = self.steps
        if oracles.held_by(self.m, self.d, "mug") != arm:
            # the moving jaw (which sticks out along the opening axis) must point away from the bottle
            away = self.obj_pose("mug")[0][:2] - self.obj_pose("bottle")[0][:2]
            # grasp at mid-body (centre of mass height), not 2 cm above the base: a low grasp lets the mug's weight and
            # handle torque it 15-20 deg in the jaw, and a tilted mug spills the pour and tips over on set-down
            _, R_try, _ = self.side_grasp_target("mug", arm, MUG_GRASP_Z, SIDE_PITCH, False)
            ld = bool(R_try[:2, 2] @ away < 0)
            r = self.pick_side("mug", arm, z_above_base=MUG_GRASP_Z, lateral_down=ld)
            if not r.ok:
                self.open_jaw(arm)
                self.park(arm)
                r = self.pick_side("mug", arm, z_above_base=MUG_GRASP_Z, lateral_down=not ld)
            if not r.ok:
                self.open_jaw(arm)
                self.park(arm)
                return SkillResult("hold_mug", False, self.steps - start, {"stage": "pick"})
        p_site, R = self.site(arm)
        p_mug, _ = self.obj_pose("mug")
        hold = p_mug - p_site
        # the hold offset rotates with the frame, so solve the target with the re-yawed offset
        R2 = self._reyaw(arm, R, p_site[:2], np.array(C.POUR_POSE[:2]))
        d = self._bearing(arm, np.array(C.POUR_POSE[:2])) - self._bearing(arm, p_site[:2])
        Rz = np.array([[math.cos(d), -math.sin(d), 0], [math.sin(d), math.cos(d), 0], [0, 0, 1.0]])
        target = np.array(C.POUR_POSE) - Rz @ hold
        self.workspace.reserve(arm)
        transit_z = target[2] + 0.08  # the mug must clear a placed plate's rim (24 mm) on the way
        self.move(arm, np.array([p_site[0], p_site[1], transit_z]), R)
        self.move(arm, np.array([target[0], target[1], transit_z]), R2, t=1.5)
        self.move(arm, target, R2)
        self.settle()
        p_final = self.obj_pose("mug")[0]
        ok = oracles.mug_held(self.m, self.d) == arm and np.linalg.norm(p_final - np.array(C.POUR_POSE)) < 0.03
        return SkillResult("hold_mug", ok, self.steps - start, {"arm": arm, "mug_pos": p_final.tolist()})

    def _levelling_rotation(self, obj: str) -> np.ndarray:
        """World rotation that would bring ``obj``'s up-axis back to vertical (applied to the site frame)."""
        up = self.obj_pose(obj)[1][:, 2]
        z = np.array([0.0, 0.0, 1.0])
        axis = np.cross(up, z)
        s_ = np.linalg.norm(axis)
        if s_ < 1e-6:
            return np.eye(3)
        axis /= s_
        ang = math.atan2(s_, float(up @ z))
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K

    def _bearing(self, arm: str, xy: np.ndarray) -> float:
        base = np.array(C.ARM_BASE_POS[arm])
        return math.atan2(xy[1] - base[1], xy[0] - base[0])

    def _reyaw(self, arm: str, R: np.ndarray, from_xy: np.ndarray, to_xy: np.ndarray) -> np.ndarray:
        """Rotate a site frame about vertical by the change in bearing base->position.

        The SO-101's approach axis has to lie in the arm's vertical plane, so carrying an object from one
        bearing to another means yawing the whole grasp frame; a yaw keeps a held object level, whereas
        commanding the old frame makes the IK trade the unreachable part for a real tilt.
        """
        d = self._bearing(arm, np.asarray(to_xy)) - self._bearing(arm, np.asarray(from_xy))
        c, sn = math.cos(d), math.sin(d)
        return np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1.0]]) @ R

    def _roll_wrist(self, arm: str, delta: float, t: float, stop: Callable[[], bool] | None = None) -> float:
        """Roll the wrist joint by ``delta`` over ``t`` seconds; with ``stop`` the roll ends early (checked every
        control step) and the roll actually applied is returned."""
        n = max(1, int(t / C.DT))
        q0 = float(self.q[arm][4])
        applied = 0.0
        for k in range(1, n + 1):
            s = k / n
            applied = delta * (3 * s * s - 2 * s * s * s)
            self.q[arm][4] = q0 + applied
            self.step()
            if stop is not None and stop():
                break
        return applied

    def water_in_bottle(self) -> int:
        """Spheres inside the bottle tube (diagnostics for the pour)."""
        return int(oracles.water_in_bottle_mask(self.m, self.d).sum())

    def pour(self, arm: str = "a", amount: str | None = None) -> SkillResult:
        """Side-grasp the bottle, carry it beside the held mug, roll the wrist so the spout tips over the rim.

        ``amount`` ("little" / "normal" / "full") sets the sphere count to deliver (constants.POUR_TARGET_SPHERES);
        the roll stops and reverses as soon as that many spheres are in the mug (closed loop on the oracle)."""
        start = self.steps
        n0 = oracles.water_in_mug(self.m, self.d)
        target = C.POUR_TARGET_SPHERES.get(amount or "normal", C.POURED_MIN_SPHERES)
        if oracles.mug_held(self.m, self.d) == arm:
            return SkillResult("pour", False, 0, {"reason": "pouring arm holds the mug"})
        p_b0 = self.obj_pose("bottle")[0].copy()
        # The wrist roll axis is (nearly) the radial approach axis, so the spout can only tip sideways
        # (tangentially).  Which side, and which grasp frame (lateral up/down = wrist rolled by pi), is decided
        # by reachability: the carry pose that puts the tipped spout over the mug must solve, and wrist_roll
        # must stay inside its range through the whole roll.
        GZ = 0.030
        # Grasp frame (lateral up/down = wrist rolled by pi): the first one whose grasp solves and keeps
        # wrist_roll inside its range through a full roll.  The tip side is chosen later by carry reachability.
        chosen = None
        for lateral_down in (False, True):
            pos_c, R_c, _ = self.side_grasp_target("bottle", arm, GZ, POUR_PITCH, lateral_down)
            r_ik = self.ik.solve(arm, pos_c, R_c, q_init=self.q[arm], max_iters=60)
            if not r_ik.ok:
                continue
            for sign in (1.0, -1.0):
                if -2.6 < r_ik.q[4] + sign * POUR_ROLL < 2.7:
                    chosen = (lateral_down, sign)
                    break
            if chosen:
                break
        if chosen is None:
            return SkillResult("pour", False, self.steps - start, {"stage": "no_feasible_roll"})
        lateral_down, roll_sign = chosen
        phase: dict[str, int] = {"start": self.water_in_bottle()}
        r = self.pick_side("bottle", arm, z_above_base=GZ, pitch=POUR_PITCH, lateral_down=lateral_down)
        phase["after_pick"] = self.water_in_bottle()
        if not r.ok:
            self.open_jaw(arm)
            self.park(arm)
            r = self.pick_side("bottle", arm, z_above_base=GZ, pitch=POUR_PITCH, lateral_down=lateral_down)
        if not r.ok:
            self.open_jaw(arm)
            self.park(arm)
            return SkillResult("pour", False, self.steps - start, {"stage": "grasp_bottle"})
        p_site, R0 = self.site(arm)
        grasp_now = self.obj_pose("bottle")[0] + np.array([0, 0, GZ])
        site_to_grasp0 = grasp_now - p_site
        spout_len = float(self.m.site_pos[self.m.site("bottle_spout").id][2]) - GZ
        mug_p = self.obj_pose("mug")[0]
        rim_z = mug_p[2] + 2 * float(self.m.geom_size[self.m.geom("mug_body_wall0").id][2]) + 0.004
        # frame at the mug's bearing: everything (approach, roll axis, hold offset) is yawed by the bearing change
        R = self._reyaw(arm, R0, p_site[:2], mug_p[:2])
        d = self._bearing(arm, mug_p[:2]) - self._bearing(arm, p_site[:2])
        Rz = np.array([[math.cos(d), -math.sin(d), 0], [math.sin(d), math.cos(d), 0], [0, 0, 1.0]])
        site_to_grasp = Rz @ site_to_grasp0
        a = R[:, 0]
        # rolling the wrist by phi rotates the bottle axis (z) about the approach axis a (Rodrigues)
        z = np.array([0, 0, 1.0])
        ax = -a  # the wrist_roll joint axis is the gripper body +z; the approach (site x) is body -z

        def rolled(phi_: float) -> np.ndarray:
            return z * math.cos(phi_) + np.cross(ax, z) * math.sin(phi_) + ax * (ax @ z) * (1 - math.cos(phi_))

        # aim the spout 2 cm over the rim centre at the *exit* roll (where the water leaves), not at full roll
        best = None
        for sign in (roll_sign, -roll_sign):
            phi_c = sign * POUR_ROLL
            q_end = float(self.q[arm][4]) + phi_c
            if not (-2.6 < q_end < 2.7):
                continue
            u_exit = rolled(sign * POUR_EXIT_ROLL)
            grasp_exit = np.array([mug_p[0], mug_p[1], rim_z + POUR_LIP_Z]) - spout_len * u_exit
            site_c = grasp_exit - site_to_grasp
            r_ik = self.ik.solve(arm, site_c, R, q_init=self.q[arm], max_iters=80)
            other = "b" if arm == "a" else "a"
            moving_pad = self.d.geom_xpos[self.env._pads[other][1]]  # the other arm's moving jaw sticks out ~6 cm
            bottle_c = site_c + (grasp_now - p_site)
            clearance = float(np.linalg.norm((bottle_c - moving_pad)[:2]))
            score = r_ik.pos_err + 0.02 * r_ik.rot_err + 0.5 * max(0.0, 0.07 - clearance)
            if best is None or score < best[0]:
                best = (score, sign, site_c)
        if best is None or best[0] > 0.06:
            self.open_jaw(arm)
            return SkillResult("pour", False, self.steps - start, {"stage": "carry_unreachable"})
        _, roll_sign, site_target = best
        phi = roll_sign * POUR_ROLL
        self.workspace.reserve(arm)
        # transit high: the hanging bottle must clear the other arm's wrist (~0.12 m) on its way over the mug
        z_hi = max(site_target[2], 0.21)
        self.move(arm, np.array([p_site[0], p_site[1], z_hi]), R0)
        phase["after_lift"] = self.water_in_bottle()
        self.move(arm, np.array([site_target[0], site_target[1], z_hi]), R, t=2.0)
        phase["after_transit"] = self.water_in_bottle()
        self.move(arm, site_target, R, t=1.0)
        self.settle()
        phase["at_pour_pose"] = self.water_in_bottle()

        # Roll-correct-roll.  The wrist joint does the rolling (exact, no IK orientation fight); between roll
        # chunks the site is moved so the *measured* spout stays over the rim centre, because the spout swings
        # by spout_len * d(u) as the roll proceeds and no single pre-correction tracks that across shapes.
        spout_sid = self.m.site("bottle_spout").id
        r_bottle = float(np.hypot(*self.m.geom_pos[self.m.geom("bottle_body_wall0").id][:2]))
        want = np.array([mug_p[0], mug_p[1], rim_z + POUR_LIP_Z])
        aborted = False

        def exit_point() -> np.ndarray:
            """Where the water actually leaves: the lowest point of the spout rim, not the spout centre."""
            u_now = self.d.xmat[self.m.body("bottle").id].reshape(3, 3)[:, 2]
            down = -z - (-z @ u_now) * u_now
            n_ = np.linalg.norm(down)
            return self.d.site_xpos[spout_sid] + (r_bottle * down / n_ if n_ > 1e-6 else 0.0)

        def correct(max_iter: int, tol: float = 0.004) -> float:
            for _ in range(max_iter):
                err = want - exit_point()
                if np.linalg.norm(err[:2]) < tol and abs(err[2]) < 0.008:
                    break
                p_now, R_now = self.site(arm)
                self.move(arm, p_now + np.clip(err, -0.04, 0.04), R_now, t=0.5)
            return float(np.linalg.norm((want - exit_point())[:2]))

        phi1 = roll_sign * math.radians(70)   # fast to a tilt at which nothing can leave yet
        self._roll_wrist(arm, phi1, 1.2)
        spout_err = correct(3)
        chunk_err: list[float] = []            # exit-point error after each roll chunk (diagnostics)
        if spout_err > 0.03:
            aborted = True                     # do not dump water on the table
            n1 = oracles.water_in_mug(self.m, self.d)
            self._roll_wrist(arm, -phi1, 1.0)
        else:
            chunks = 16                      # ~4 deg per chunk: the spheres leave one by one instead of as a slug
            dphi = (phi - phi1) / chunks
            applied = phi1

            def enough() -> bool:
                return oracles.water_in_mug(self.m, self.d) - n0 >= target

            def enough_soon() -> bool:   # stop the roll one sphere early: what is already in the air still lands
                return oracles.water_in_mug(self.m, self.d) - n0 >= max(1, target - 1)

            for _ in range(chunks):
                if enough_soon():
                    break
                applied += self._roll_wrist(arm, dphi, 0.5, stop=enough_soon)
                if enough_soon():
                    break
                chunk_err.append(round(correct(1), 4))
            # hold the tilt while water keeps arriving: the spheres trickle through the spout, so wait until the count has
            # not changed for a second (at most 4 s); if the stream never starts, tip a further 20 deg (still inside the
            # wrist range) and wait again
            def hold(max_s: float) -> int:
                last, quiet = oracles.water_in_mug(self.m, self.d), 0
                for _ in range(int(max_s / C.DT)):
                    if enough():
                        break
                    self.step()
                    now = oracles.water_in_mug(self.m, self.d)
                    quiet = quiet + 1 if now == last else 0
                    last = now
                    if quiet > int(1.0 / C.DT):
                        break
                return last

            got = hold(4.0)
            if got - n0 < max(2, target // 2) and not enough():
                extra = roll_sign * math.radians(20)
                q_next = float(self.q[arm][4]) + extra
                if -2.6 < q_next < 2.7:
                    applied += self._roll_wrist(arm, extra, 0.6, stop=enough_soon)
                    correct(1)
                    hold(4.0)
            n1 = oracles.water_in_mug(self.m, self.d)
            phase["after_roll"] = self.water_in_bottle()
            self._roll_wrist(arm, -applied, 1.5)
        self.settle(0.3)
        # put the bottle back where it was and let go
        back_site = np.array([p_b0[0], p_b0[1], 0.0]) + np.array([0, 0, GZ]) - site_to_grasp0
        p_now, _ = self.site(arm)
        self.move(arm, np.array([p_now[0], p_now[1], z_hi]), R)
        self.move(arm, np.array([back_site[0], back_site[1], z_hi]), R0, t=2.0)
        self.move(arm, back_site + np.array([0, 0, 0.004]), R0, t=1.5)
        self.settle()
        self.open_jaw(arm, width=0.04)
        self.move(arm, back_site - R0[:, 0] * 0.06 + np.array([0, 0, 0.05]), R0)
        self.park(arm)
        ok = (n1 - n0) >= target
        return SkillResult("pour", ok, self.steps - start, {"arm": arm, "amount": amount or "normal", "target_spheres": target, "poured": n1 - n0,
                                                          "spheres_before": n0, "spheres_after": n1, "phi": phi, "aborted": aborted,
                                                          "spout_err": round(float(spout_err), 4), "chunk_err": chunk_err, "roll_sign": roll_sign, "lateral_down": lateral_down,
                                                          "in_bottle_by_phase": phase})


def _slerp_mat(R0: np.ndarray, R1: np.ndarray, s: float) -> np.ndarray:
    if s >= 1.0:
        return R1
    q0, q1 = np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(q0, R0.ravel())
    mujoco.mju_mat2Quat(q1, R1.ravel())
    if q0 @ q1 < 0:
        q1 = -q1
    dot = float(np.clip(q0 @ q1, -1, 1))
    if dot > 0.9995:
        q = q0 + s * (q1 - q0)
    else:
        th = math.acos(dot)
        q = (math.sin((1 - s) * th) * q0 + math.sin(s * th) * q1) / math.sin(th)
    q /= np.linalg.norm(q)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)
