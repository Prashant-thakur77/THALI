"""Follow-ups and corrections: does the resolver turn a short correction into the right, verifier-approved plan?

Twenty scripted (previous command's executed steps, follow-up utterance, expected plan) cases, checked with the
verifier on the held-out seed-0 scene (with the world advanced by the previous steps' effects).  No simulator rollouts:
the skills themselves are measured elsewhere; this measures language -> plan.  Writes results/followups.json.

    python -m eval.followups
"""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym

import souschef_env  # noqa: F401
from souschef_env.scene_description import scene_state
from runtime.followups import detect, resolve
from verifier.rules import BLOCK, Verifier, World

ROOT = Path(__file__).resolve().parent.parent
FORK = {"skill": "pick_place", "arm": "a", "obj": "fork_1", "zone": "fork"}
SPOON = {"skill": "handoff", "arm": "a", "obj": "spoon_1", "to_arm": "b", "zone": "spoon"}
PLATE = {"skill": "pick_place", "arm": "a", "obj": "plate", "zone": "plate"}
DRAWER = {"skill": "open_drawer", "arm": "a"}
HOLD, POUR, PLACE = {"skill": "hold_mug", "arm": "b"}, {"skill": "pour", "arm": "a", "amount": "normal"}, {"skill": "place_mug", "arm": "b"}
CASES = [
    ([DRAWER, FORK], "no, the other side", [{"skill": "handoff", "arm": "a", "obj": "fork_1", "to_arm": "b", "zone": "spoon"}]),
    ([DRAWER, SPOON], "the other side please", [{"skill": "handoff", "arm": "b", "obj": "spoon_1", "to_arm": "a", "zone": "fork"}]),
    ([DRAWER, FORK], "again", [FORK]),
    ([DRAWER, FORK], "do that again", [FORK]),
    ([HOLD, POUR, PLACE], "a bit more", [HOLD, {"skill": "pour", "arm": "a", "amount": "little"}, PLACE]),
    ([HOLD, POUR, PLACE], "more water", [HOLD, {"skill": "pour", "arm": "a", "amount": "little"}, PLACE]),
    ([HOLD, POUR], "some more", [HOLD, {"skill": "pour", "arm": "a", "amount": "little"}, PLACE]),   # on the reset scene the mug is not held: re-hold first
    ([DRAWER, FORK], "use the other arm", [dict(FORK, arm="b")]),
    ([DRAWER, SPOON], "the other arm", [{"skill": "handoff", "arm": "b", "obj": "spoon_1", "to_arm": "a", "zone": "spoon"}]),
    ([DRAWER, PLATE], "once more", [PLATE]),
    ([DRAWER, FORK, PLATE], "no, the other side", [{"skill": "handoff", "arm": "a", "obj": "fork_1", "to_arm": "b", "zone": "spoon"}]),
    ([DRAWER], "open the top drawer", None),   # not a follow-up: the planner handles it
    ([DRAWER, FORK], "put the plate on the table", None),
    ([HOLD, POUR, PLACE], "top me up", [HOLD, {"skill": "pour", "arm": "a", "amount": "little"}, PLACE]),
    ([DRAWER, FORK], "repeat that", [FORK]),
    ([DRAWER, PLATE], "other side", None),      # a plate has no mirrored zone: nothing to resolve
    ([], "again", None),                        # no previous command
    ([DRAWER, FORK, SPOON], "wrong side", [{"skill": "handoff", "arm": "b", "obj": "spoon_1", "to_arm": "a", "zone": "fork"}]),
    ([HOLD, POUR, PLACE], "keep pouring", [HOLD, {"skill": "pour", "arm": "a", "amount": "little"}, PLACE]),
    ([DRAWER, FORK], "one more time", [FORK]),
]


def main() -> None:
    env = gym.make("souschef_env/Thali-v0", disable_env_checker=True).unwrapped
    env.reset(seed=0, options={"split": "test"})
    st = scene_state(env.model, env.data)
    ver = Verifier()
    rows, correct, approved = [], 0, 0
    for last, text, expected in CASES:
        kind = detect(text)
        plan = resolve(kind, last, st) if kind else None
        ok = (plan["steps"] if plan else None) == expected
        verdict = None
        if plan:
            world = World.from_scene_state(st)
            for s in last:   # apply the previous command's effects so preconditions hold
                ver.check_step(0, s, world, None, [])
            verdict = ver.verify(plan, world).verdict
            approved += verdict != BLOCK
        correct += ok
        rows.append({"last": last, "text": text, "kind": kind, "plan": plan["steps"] if plan else None, "expected": expected, "correct": ok, "verdict": verdict})
        print(f"{'ok ' if ok else 'BAD'} {text!r:26} -> {kind} {verdict or ''}")
    out = {"cases": len(CASES), "correct": correct, "resolved": sum(1 for r in rows if r["plan"]), "verifier_approved": approved, "rows": rows}
    (ROOT / "results" / "followups.json").write_text(json.dumps(out, indent=1))
    print(f"{correct}/{len(CASES)} correct, {approved}/{out['resolved']} resolved plans verifier-approved")


if __name__ == "__main__":
    main()
