"""Per-arm skill queues with dependencies (plan Phase 7, Flip & Ship pattern).

Each verified plan step is assigned to the arm that performs it and appended to that arm's queue.  Steps that
need both arms (handoff, pour, and anything after a hold) carry dependencies on the other arm's earlier steps.
``next_ready`` returns the first step whose dependencies are done, preferring the arm that has been idle
longer, so arm B's ``hold_mug`` is dispatched while arm A still has nothing runnable.  Execution itself is one
skill at a time (the scripted expert and the policies move one arm's chain per skill); true simultaneous
motion of both arms is not implemented and is stated as such in the README.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from souschef_env import constants as C


@dataclass
class QueuedStep:
    idx: int
    step: dict
    arm: str
    deps: list[int] = field(default_factory=list)
    status: str = "queued"       # queued | running | done | failed | skipped


class ArmQueues:
    def __init__(self) -> None:
        self.queues: dict[str, deque[QueuedStep]] = {a: deque() for a in C.ARMS}
        self.all: list[QueuedStep] = []
        self.last_ran: dict[str, int] = {a: -1 for a in C.ARMS}

    def load(self, plan: dict) -> None:
        self.queues = {a: deque() for a in C.ARMS}
        self.all = []
        last_by_arm: dict[str, int | None] = {a: None for a in C.ARMS}
        drawer_idx: int | None = None
        hold_idx: dict[str, int | None] = {a: None for a in C.ARMS}
        for i, s in enumerate(plan["steps"]):
            arm = s["arm"]
            q = QueuedStep(i, s, arm)
            # same-arm steps stay ordered
            if last_by_arm[arm] is not None:
                q.deps.append(last_by_arm[arm])
            other = "b" if arm == "a" else "a"
            if s["skill"] in ("pick_place", "handoff") and s.get("obj") in C.CUTLERY and drawer_idx is not None:
                q.deps.append(drawer_idx)
            if s["skill"] == "handoff" and last_by_arm[s.get("to_arm", other)] is not None:
                q.deps.append(last_by_arm[s.get("to_arm", other)])   # the receiving arm must be free
            if s["skill"] == "pour":
                if hold_idx[other] is not None:
                    q.deps.append(hold_idx[other])                    # mug held by the other arm first
            if s["skill"] == "place_mug" and last_by_arm[other] is not None:
                q.deps.append(last_by_arm[other])                     # after the pour
            if s["skill"] == "open_drawer":
                drawer_idx = i
            if s["skill"] == "hold_mug":
                hold_idx[arm] = i
            last_by_arm[arm] = i
            if s["skill"] == "handoff":
                last_by_arm[s.get("to_arm", other)] = i               # the receiver ends up busy with it too
            self.queues[arm].append(q)
            self.all.append(q)

    def next_ready(self) -> QueuedStep | None:
        done = {q.idx for q in self.all if q.status in ("done", "skipped")}
        candidates = []
        for arm, q in self.queues.items():
            if q and q[0].status == "queued" and all(d in done for d in q[0].deps):
                candidates.append((self.last_ran[arm], arm, q[0]))
        if not candidates:
            return None
        candidates.sort(key=lambda t: (t[0], t[1]))
        return candidates[0][2]

    def mark(self, q: QueuedStep, status: str) -> None:
        q.status = status
        if status in ("done", "skipped", "failed"):
            self.queues[q.arm].popleft()
            self.last_ran[q.arm] = max(self.last_ran.values()) + 1

    def pending(self) -> list[QueuedStep]:
        return [q for q in self.all if q.status == "queued"]

    def deadlocked(self) -> bool:
        return self.next_ready() is None and bool(self.pending())

    def snapshot(self) -> dict:
        return {a: [(q.idx, q.step["skill"], q.status) for q in self.queues[a]] for a in C.ARMS}
