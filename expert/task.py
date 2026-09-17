"""The full dinner-table task as a fixed skill program, and the per-skill catalogue the demo recorder uses.

Full task (plan §3 Phase 9 command): open the drawer (A) -> fork to the left of the plate (A) -> spoon to the
right of the plate (A hands to B, B places: the spoon zone is out of A's reach) -> plate (A) -> B holds the
mug -> A pours -> B sets the mug down on its spot.
"""

from __future__ import annotations

from dataclasses import dataclass

from souschef_env import constants as C
from souschef_env import oracles
from expert.primitives import Expert, SkillResult


@dataclass(frozen=True)
class Step:
    skill: str
    arm: str
    obj: str | None = None
    zone: str | None = None
    arm2: str | None = None
    amount: str | None = None   # pour only: "little" / "normal" / "full"

    def run(self, ex: Expert) -> SkillResult:
        if self.skill == "open_drawer":
            return ex.open_drawer(self.arm)
        if self.skill == "pick_place":
            return ex.pick_place(self.obj, self.arm, self.zone)
        if self.skill == "handoff":
            if self.zone is None:   # plain handoff: the receiving arm ends up holding the piece (e.g. before put_in_drawer)
                return ex.handoff(self.obj, self.arm, self.arm2)
            return ex.handoff_place(self.obj, self.arm, self.arm2, self.zone)
        if self.skill == "hold_mug":
            return ex.hold_mug(self.arm)
        if self.skill == "pour":
            return ex.pour(self.arm, self.amount)
        if self.skill == "put_in_drawer":
            return ex.put_in_drawer(self.obj, self.arm)
        if self.skill == "close_drawer":
            return ex.close_drawer(self.arm)
        if self.skill == "place_mug":
            (zx, zy), _ = C.ZONES["mug"]
            r = ex.place("mug", self.arm, (zx, zy))
            ex.park(self.arm)
            ex.workspace.release(self.arm)
            return SkillResult("place_mug", oracles.object_in_zone(ex.m, ex.d, "mug", "mug"), r.steps, r.detail)
        raise ValueError(self.skill)


FULL_TASK: tuple[Step, ...] = (
    Step("open_drawer", "a"),
    Step("pick_place", "a", "fork_1", "fork"),
    Step("handoff", "a", "spoon_1", "spoon", arm2="b"),
    Step("pick_place", "a", "plate", "plate"),
    Step("hold_mug", "b"),
    Step("pour", "a"),
    Step("place_mug", "b"),
)


def run_full_task(ex: Expert, steps: tuple[Step, ...] = FULL_TASK, stop_on_fail: bool = False) -> list[SkillResult]:
    out = []
    for st in steps:
        r = st.run(ex)
        out.append(r)
        if stop_on_fail and not r.ok:
            break
    return out
