"""Deterministic plan verifier (plan Phase 5.1).

Input: a skill plan (planner/schema.json) and a symbolic world state (from the sim oracles, the scene
description, or the camera-derived state).  The plan is simulated step by step against a copy of that
state; every step is checked against the rules below and the world is advanced by the step's effects.

Rules (each yields a coded reason):
  SCHEMA            plan does not match planner/schema.json
  UNKNOWN_OBJECT    step names an object / zone the scene does not have
  REACH             the target is outside the arm's top-down reach (IK on the arms-only model)
  GRASP_PRECOND     arm already holds something / object held by the other arm / drawer closed for cutlery
  WORKSPACE         two consecutive steps of different arms both enter the shared zone with it reserved
  POUR_PRECOND      pour without the mug held by the *other* arm (under the spout)
  ORDER             drawer-open-before-cutlery or hold-before-pour appears later in the plan -> REORDER
  VELOCITY          a commanded joint step exceeds the per-joint velocity limit (gentle mode halves it)
  SELF_HANDOFF      handoff to the same arm
  ZONE_MISMATCH     object placed in a zone meant for another kind of object

Verdicts: ALLOW (plan runs as given), REORDER (a fixed plan is returned), BLOCK (refused, with reasons).
Borrowed shape: TaskForge's PlanValidator (issue codes per step), RePlanTable/Sovereign's allow/slow/stop loop.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from souschef_env import constants as C

SCHEMA = json.loads((Path(__file__).resolve().parent.parent / "planner" / "schema.json").read_text())
SKILLS = SCHEMA["properties"]["steps"]["items"]["properties"]["skill"]["enum"]
ZONE_FOR = {"plate": "plate", "mug": "mug", "fork_1": "fork", "fork_2": "fork", "spoon_1": "spoon", "spoon_2": "spoon"}
SHARED_ZONE_SKILLS = {"handoff", "hold_mug", "pour"}
# per-joint velocity limits (rad/s): STS3215 no-load ~ 6 rad/s; we allow 3, gentle mode 1.5
JOINT_VEL_LIMIT = 3.0
GENTLE_FACTOR = 0.5
JAW_MAX_Q = 1.745

ALLOW, REORDER, BLOCK = "ALLOW", "REORDER", "BLOCK"


@dataclass
class Issue:
    code: str
    step: int | None
    message: str

    def as_dict(self) -> dict:
        return {"code": self.code, "step": self.step, "message": self.message}


@dataclass
class Verdict:
    verdict: str
    issues: list[Issue] = field(default_factory=list)
    plan: dict | None = None  # the (possibly reordered) plan that may run

    def as_dict(self) -> dict:
        return {"verdict": self.verdict, "issues": [i.as_dict() for i in self.issues], "plan": self.plan}


@dataclass
class World:
    """Symbolic world: object positions (m), who holds what, drawer state."""
    objects: dict[str, tuple[float, float, float]]
    held: dict[str, str | None]        # obj -> arm
    drawer_open: bool
    in_drawer: dict[str, bool]
    zones: dict[str, tuple[float, float]] = field(default_factory=lambda: {k: v[0] for k, v in C.ZONES.items()})

    @classmethod
    def from_scene_state(cls, st: dict) -> "World":
        objs = {k: (v["x_cm"] / 100, v["y_cm"] / 100, v["z_cm"] / 100) for k, v in st["objects"].items()}
        return cls(objects=objs, held={k: v["held_by"] for k, v in st["objects"].items()},
                   drawer_open=bool(st["drawer"]["open"]), in_drawer={k: bool(v["in_drawer"]) for k, v in st["objects"].items()})

    def holding(self, arm: str) -> str | None:
        for o, a in self.held.items():
            if a == arm:
                return o
        return None


def validate_schema(plan: Any) -> list[Issue]:
    """Minimal JSON-schema check without a dependency: types, enums, required keys, no extra keys."""
    issues: list[Issue] = []
    if not isinstance(plan, dict):
        return [Issue("SCHEMA", None, "plan is not an object")]
    extra = set(plan) - set(SCHEMA["properties"])
    if extra:
        issues.append(Issue("SCHEMA", None, f"unexpected keys {sorted(extra)}"))
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        return issues + [Issue("SCHEMA", None, "steps must be a non-empty list")]
    if len(steps) > SCHEMA["properties"]["steps"]["maxItems"]:
        issues.append(Issue("SCHEMA", None, f"too many steps ({len(steps)})"))
    if plan.get("mode", "normal") not in ("normal", "gentle"):
        issues.append(Issue("SCHEMA", None, f"bad mode {plan.get('mode')!r}"))
    props = SCHEMA["properties"]["steps"]["items"]["properties"]
    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            issues.append(Issue("SCHEMA", i, "step is not an object"))
            continue
        for k in ("skill", "arm"):
            if k not in s:
                issues.append(Issue("SCHEMA", i, f"missing {k}"))
        for k, v in s.items():
            if k not in props:
                issues.append(Issue("SCHEMA", i, f"unexpected key {k!r}"))
            elif v not in props[k]["enum"]:
                issues.append(Issue("SCHEMA", i, f"{k}={v!r} not in {props[k]['enum']}"))
    return issues


class Verifier:
    def __init__(self, reach_check: bool = True):
        self._ik = None
        self.reach_check = reach_check
        self._reach_cache: dict[tuple, bool] = {}

    # ------------------------------------------------------------ reach
    def reachable(self, arm: str, xy: tuple[float, float], z: float = 0.03) -> bool:
        if not self.reach_check:
            return True
        key = (arm, round(xy[0], 2), round(xy[1], 2), round(z, 2))
        if key not in self._reach_cache:
            from souschef_env.ik import ArmIK, top_down_rotation
            if self._ik is None:
                self._ik = ArmIK()
            ok = False
            for yaw in (0.0, np.pi / 2):
                r = self._ik.solve(arm, np.array([xy[0], xy[1], z]), top_down_rotation(yaw), max_iters=60)
                if r.ok:  # POS_TOL 4 mm, ROT_TOL 20 deg -- the same bar the expert's primitives use
                    ok = True
                    break
            self._reach_cache[key] = ok
        return self._reach_cache[key]

    # ------------------------------------------------------------ velocity
    @staticmethod
    def check_velocity(q_prev: np.ndarray, q_next: np.ndarray, dt: float, mode: str = "normal") -> Issue | None:
        """Joint-space step check for the runtime: BLOCK a command that would exceed the velocity limit."""
        lim = JOINT_VEL_LIMIT * (GENTLE_FACTOR if mode == "gentle" else 1.0)
        q_prev, q_next = np.asarray(q_prev, dtype=float), np.asarray(q_next, dtype=float)
        idx = [i for i in range(len(q_prev)) if i % 6 != 5]  # skip the two jaw entries
        v = np.abs(q_next[idx] - q_prev[idx]) / dt
        if v.max() > lim:
            j = idx[int(v.argmax())]
            return Issue("VELOCITY", None, f"joint {C.JOINTS[j]} would move at {v.max():.1f} rad/s > {lim:.1f}")
        return None

    # ------------------------------------------------------------ plan
    def check_step(self, i: int, s: dict, w: World, owner: str | None, later: list[dict] | None = None) -> tuple[list[Issue], str | None]:
        """Issues for step ``s`` given world ``w`` (``later`` = the remaining steps, for ORDER detection);
        returns (issues, new shared-zone owner)."""
        issues: list[Issue] = []
        later = later or []
        skill, arm, obj, zone, to_arm = s["skill"], s["arm"], s.get("obj"), s.get("zone"), s.get("to_arm")
        other = "b" if arm == "a" else "a"

        if obj is not None and obj not in w.objects:
            return [Issue("UNKNOWN_OBJECT", i, f"no object {obj!r} in the scene")], owner
        if zone is not None and zone not in w.zones:
            return [Issue("UNKNOWN_OBJECT", i, f"no zone {zone!r}")], owner

        if skill == "open_drawer":
            if w.holding(arm):
                issues.append(Issue("GRASP_PRECOND", i, f"arm {arm} is holding {w.holding(arm)}"))
            cx, cy, _ = C.CABINET_POS
            if not self.reachable(arm, (cx, cy - 0.10), 0.035):
                issues.append(Issue("REACH", i, f"drawer handle out of reach of arm {arm}"))
            if w.drawer_open:
                issues.append(Issue("ORDER", i, "drawer is already open"))

        elif skill == "pick_place":
            if obj is None or zone is None:
                return [Issue("SCHEMA", i, "pick_place needs obj and zone")], owner
            if ZONE_FOR.get(obj) != zone:
                issues.append(Issue("ZONE_MISMATCH", i, f"{obj} does not belong in zone {zone!r}"))
            if obj in C.CUTLERY and w.in_drawer.get(obj, False) and not w.drawer_open:
                issues.append(Issue("ORDER", i, f"{obj} is in the closed drawer: open_drawer must come first"))
            if w.held.get(obj) == other:
                issues.append(Issue("GRASP_PRECOND", i, f"{obj} is held by arm {other}"))
            if w.holding(arm) not in (None, obj):
                issues.append(Issue("GRASP_PRECOND", i, f"arm {arm} already holds {w.holding(arm)}"))
            p = w.objects[obj]
            if not self.reachable(arm, (p[0], p[1])):
                issues.append(Issue("REACH", i, f"{obj} at ({p[0]:.2f},{p[1]:.2f}) is out of reach of arm {arm}"))
            if not self.reachable(arm, w.zones[zone]):
                issues.append(Issue("REACH", i, f"zone {zone} is out of reach of arm {arm} (needs a handoff)"))

        elif skill == "handoff":
            if obj is None or to_arm is None:
                return [Issue("SCHEMA", i, "handoff needs obj and to_arm")], owner
            if to_arm == arm:
                issues.append(Issue("SELF_HANDOFF", i, "handoff to the same arm"))
            if obj in C.CUTLERY and w.in_drawer.get(obj, False) and not w.drawer_open:
                issues.append(Issue("ORDER", i, f"{obj} is in the closed drawer: open_drawer must come first"))
            if w.held.get(obj) not in (None, arm):
                issues.append(Issue("GRASP_PRECOND", i, f"{obj} is held by arm {w.held[obj]}"))
            if w.holding(arm) not in (None, obj):
                issues.append(Issue("GRASP_PRECOND", i, f"arm {arm} already holds {w.holding(arm)}"))
            if w.holding(to_arm):
                issues.append(Issue("GRASP_PRECOND", i, f"receiving arm {to_arm} holds {w.holding(to_arm)}"))
            p = w.objects[obj]
            if w.held.get(obj) != arm and not self.reachable(arm, (p[0], p[1])):
                issues.append(Issue("REACH", i, f"{obj} is out of reach of arm {arm}"))
            if owner not in (None, arm):
                issues.append(Issue("WORKSPACE", i, f"shared zone reserved by arm {owner}"))
            if zone is not None and not self.reachable(to_arm, w.zones[zone]):
                issues.append(Issue("REACH", i, f"zone {zone} is out of reach of the receiving arm {to_arm}"))

        elif skill == "hold_mug":
            if w.held.get("mug") == other:
                issues.append(Issue("GRASP_PRECOND", i, f"mug is held by arm {other}"))
            if w.holding(arm) not in (None, "mug"):
                issues.append(Issue("GRASP_PRECOND", i, f"arm {arm} holds {w.holding(arm)}"))
            p = w.objects["mug"]
            if w.held.get("mug") != arm and not self.reachable(arm, (p[0], p[1])):
                issues.append(Issue("REACH", i, f"mug is out of reach of arm {arm}"))
            if owner not in (None, arm):
                issues.append(Issue("WORKSPACE", i, f"shared zone reserved by arm {owner}"))

        elif skill == "pour":
            if w.held.get("mug") != other:
                if any(t.get("skill") == "hold_mug" and t.get("arm") == other for t in later):
                    issues.append(Issue("ORDER", i, f"hold_mug by arm {other} must come before the pour"))
                else:
                    issues.append(Issue("POUR_PRECOND", i, f"pour by arm {arm} requires the mug held by arm {other} under the spout"))
            if w.holding(arm) not in (None, "bottle"):
                issues.append(Issue("GRASP_PRECOND", i, f"arm {arm} holds {w.holding(arm)}"))
            p = w.objects["bottle"]
            if not self.reachable(arm, (p[0], p[1])):
                issues.append(Issue("REACH", i, f"bottle is out of reach of arm {arm}"))
            if owner not in (None, other, arm):
                issues.append(Issue("WORKSPACE", i, f"shared zone reserved by arm {owner}"))

        elif skill == "place_mug":
            if w.held.get("mug") != arm:
                issues.append(Issue("GRASP_PRECOND", i, f"arm {arm} does not hold the mug"))
            if not self.reachable(arm, w.zones["mug"]):
                issues.append(Issue("REACH", i, f"mug zone is out of reach of arm {arm}"))
        elif skill == "put_in_drawer":
            if obj is None or obj not in C.CUTLERY:
                return [Issue("SCHEMA", i, "put_in_drawer needs a fork or spoon as obj")], owner
            if not w.drawer_open:
                if any(t.get("skill") == "open_drawer" for t in later):
                    issues.append(Issue("ORDER", i, "open_drawer must come before put_in_drawer"))
                else:
                    issues.append(Issue("GRASP_PRECOND", i, "the drawer is closed: open_drawer first"))
            if w.held.get(obj) == other:
                issues.append(Issue("GRASP_PRECOND", i, f"{obj} is held by arm {other}"))
            if w.holding(arm) not in (None, obj):
                issues.append(Issue("GRASP_PRECOND", i, f"arm {arm} already holds {w.holding(arm)}"))
            p = w.objects[obj]
            if w.held.get(obj) != arm and not self.reachable(arm, (p[0], p[1])):
                issues.append(Issue("REACH", i, f"{obj} at ({p[0]:.2f},{p[1]:.2f}) is out of reach of arm {arm} (needs a handoff)"))
            cx, cy, _ = C.CABINET_POS
            if not self.reachable(arm, (cx - 0.075, cy - 0.13)):
                issues.append(Issue("REACH", i, f"the drawer tray is out of reach of arm {arm}"))

        elif skill == "close_drawer":
            if w.holding(arm):
                issues.append(Issue("GRASP_PRECOND", i, f"arm {arm} is holding {w.holding(arm)}"))
            cx, cy, _ = C.CABINET_POS
            if not self.reachable(arm, (cx, cy - 0.10), 0.035):
                issues.append(Issue("REACH", i, f"drawer handle out of reach of arm {arm}"))
            if not w.drawer_open:
                issues.append(Issue("ORDER", i, "drawer is already closed"))
            if any(t.get("skill") in ("put_in_drawer", "pick_place", "handoff") and t.get("obj") in C.CUTLERY for t in later):
                issues.append(Issue("ORDER", i, "close_drawer must come after the cutlery steps"))
        else:
            issues.append(Issue("SCHEMA", i, f"unknown skill {skill!r}"))

        # effects (only applied when the step is legal)
        if not issues:
            if skill == "open_drawer":
                w.drawer_open = True
            elif skill == "put_in_drawer":
                cx, cy, _ = C.CABINET_POS
                w.held[obj] = None
                w.in_drawer[obj] = True
                w.objects[obj] = (cx, cy - 0.12, 0.0)
            elif skill == "close_drawer":
                w.drawer_open = False
            elif skill == "pick_place":
                w.held[obj] = None
                w.in_drawer[obj] = False
                w.objects[obj] = (*w.zones[zone], 0.0)
            elif skill == "handoff":
                w.held[obj] = to_arm
                w.in_drawer[obj] = False
                w.objects[obj] = (0.0, 0.0, 0.0)
                owner = to_arm
                if zone is not None:
                    w.held[obj] = None
                    w.objects[obj] = (*w.zones[zone], 0.0)
                    owner = None
            elif skill == "hold_mug":
                w.held["mug"] = arm
                w.objects["mug"] = tuple(C.POUR_POSE)
                owner = arm
            elif skill == "pour":
                w.held["bottle"] = None
            elif skill == "place_mug":
                w.held["mug"] = None
                w.objects["mug"] = (*w.zones["mug"], 0.0)
                owner = None
        return issues, owner

    def verify(self, plan: dict, world: World) -> Verdict:
        schema_issues = validate_schema(plan)
        if schema_issues:
            return Verdict(BLOCK, schema_issues)
        plan = copy.deepcopy(plan)
        issues = self._simulate(plan, world)
        if not issues:
            return Verdict(ALLOW, [], plan)
        # ORDER issues may be fixable by moving prerequisite steps forward
        if all(i.code == "ORDER" for i in issues):
            fixed = self._reorder(plan, world)
            if fixed is not None and not self._simulate(fixed, world):
                return Verdict(REORDER, issues, fixed)
        return Verdict(BLOCK, issues)

    def _simulate(self, plan: dict, world: World) -> list[Issue]:
        w = copy.deepcopy(world)
        owner: str | None = None
        issues: list[Issue] = []
        steps = plan["steps"]
        for i, s in enumerate(steps):
            step_issues, owner = self.check_step(i, s, w, owner, later=steps[i + 1 :])
            issues.extend(step_issues)
        return issues

    @staticmethod
    def _reorder(plan: dict, world: World) -> dict | None:
        """Pull open_drawer before the first cutlery step and hold_mug before pour; drop a redundant open_drawer."""
        steps = list(plan["steps"])
        opens = [s for s in steps if s["skill"] == "open_drawer"]
        rest = [s for s in steps if s["skill"] != "open_drawer"]
        if world.drawer_open:
            opens = []
        elif not opens and any(s.get("obj") in C.CUTLERY for s in rest):
            opens = [{"skill": "open_drawer", "arm": "a"}]
        out: list[dict] = list(opens[:1])
        holds = [s for s in rest if s["skill"] == "hold_mug"]
        pours = [s for s in rest if s["skill"] == "pour"]
        if pours and not holds:
            holds = [{"skill": "hold_mug", "arm": "b" if pours[0]["arm"] == "a" else "a"}]
        seen_hold = False
        for s in rest:
            if s["skill"] == "hold_mug":
                if not seen_hold:
                    out.append(s)
                    seen_hold = True
                continue
            if s["skill"] == "pour" and not seen_hold:
                out.append(holds[0])
                seen_hold = True
            out.append(s)
        fixed = dict(plan)
        fixed["steps"] = out
        return fixed
