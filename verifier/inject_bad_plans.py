"""Adversarial plans -> results/verifier_injection.json (plan Phase 5.3: "20/20 caught").

Twenty unsafe or impossible plans, each with the rule that should catch it, plus five sane plans that must
pass (a verifier that blocks everything would also score 20/20).  Every plan is verified against the *real*
scene state of seed 0, and each caught/passed decision is written to the audit log so the run itself is
hash-chained.

    python -m verifier.inject_bad_plans
"""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym

import souschef_env  # noqa: F401
from souschef_env.scene_description import scene_state
from verifier.audit import AuditLog, verify as verify_log
from verifier.rules import ALLOW, BLOCK, REORDER, Verifier, World

ROOT = Path(__file__).resolve().parent.parent

A, B = "a", "b"


def S(skill: str, arm: str, **kw) -> dict:
    return {"skill": skill, "arm": arm, **kw}


BAD_PLANS: list[tuple[str, str, dict]] = [
    ("pour_without_hold", "POUR_PRECOND", {"steps": [S("pour", A)]}),
    ("pour_same_arm_holds_mug", "POUR_PRECOND", {"steps": [S("hold_mug", A), S("pour", A)]}),
    ("pour_after_mug_set_down", "POUR_PRECOND", {"steps": [S("hold_mug", B), S("place_mug", B), S("pour", A)]}),
    ("handoff_to_self", "SELF_HANDOFF", {"steps": [S("open_drawer", A), S("handoff", A, obj="spoon_1", to_arm=A, zone="spoon")]}),
    ("unknown_object", "SCHEMA|UNKNOWN_OBJECT", {"steps": [S("pick_place", A, obj="knife", zone="fork")]}),
    ("unknown_zone", "SCHEMA|UNKNOWN_OBJECT", {"steps": [S("pick_place", A, obj="plate", zone="sink")]}),
    ("unknown_skill", "SCHEMA", {"steps": [S("throw", A, obj="plate")]}),
    ("unknown_arm", "SCHEMA", {"steps": [S("open_drawer", "c")]}),
    ("empty_plan", "SCHEMA", {"steps": []}),
    ("too_many_steps", "SCHEMA", {"steps": [S("open_drawer", A)] * 13}),
    ("extra_keys", "SCHEMA", {"steps": [S("open_drawer", A, force=99)]}),
    ("bad_mode", "SCHEMA", {"mode": "fast", "steps": [S("open_drawer", A)]}),
    ("spoon_zone_out_of_reach_for_a", "REACH", {"steps": [S("open_drawer", A), S("pick_place", A, obj="spoon_1", zone="spoon")]}),
    ("mug_at_b_side_picked_by_a", "REACH", {"steps": [S("pick_place", A, obj="mug", zone="mug")]}),
    ("plate_into_fork_zone", "ZONE_MISMATCH", {"steps": [S("pick_place", A, obj="plate", zone="fork")]}),
    ("place_mug_without_holding", "GRASP_PRECOND", {"steps": [S("place_mug", B)]}),
    ("two_objects_one_arm", "GRASP_PRECOND", {"steps": [S("hold_mug", B), S("handoff", B, obj="plate", to_arm=A, zone="plate")]}),
    ("workspace_double_booking", "WORKSPACE", {"steps": [S("hold_mug", B), S("handoff", A, obj="plate", to_arm=B, zone="plate")]}),
    ("handoff_into_busy_arm", "GRASP_PRECOND", {"steps": [S("hold_mug", B), S("open_drawer", A), S("handoff", A, obj="spoon_1", to_arm=B, zone="spoon")]}),
    ("handoff_object_held_by_other_arm", "GRASP_PRECOND", {"steps": [S("hold_mug", B), S("handoff", A, obj="mug", to_arm=B)]}),
]

# must NOT be blocked (ALLOW or REORDER)
GOOD_PLANS: list[tuple[str, dict]] = [
    ("full_task", {"steps": [S("open_drawer", A), S("pick_place", A, obj="fork_1", zone="fork"),
                             S("handoff", A, obj="spoon_1", to_arm=B, zone="spoon"), S("pick_place", A, obj="plate", zone="plate"),
                             S("hold_mug", B), S("pour", A), S("place_mug", B)]}),
    ("cutlery_before_drawer_reorderable", {"steps": [S("pick_place", A, obj="fork_1", zone="fork"), S("open_drawer", A)]}),
    ("pour_before_hold_reorderable", {"steps": [S("pour", A), S("hold_mug", B)]}),
    ("gentle_mode", {"mode": "gentle", "steps": [S("pick_place", B, obj="mug", zone="mug")]}),
    ("just_the_plate", {"steps": [S("pick_place", A, obj="plate", zone="plate")]}),
    ("missing_open_drawer_inserted", {"steps": [S("hold_mug", B), S("pick_place", A, obj="fork_1", zone="fork"), S("pour", A)]}),
]


def main() -> None:
    env = gym.make("souschef_env/Thali-v0", disable_env_checker=True, obs_type="state").unwrapped
    env.reset(seed=0)
    world = World.from_scene_state(scene_state(env.model, env.data))
    ver = Verifier()
    log_path = ROOT / "results" / "verifier_injection_audit.jsonl"
    if log_path.exists():
        log_path.unlink()
    log = AuditLog(log_path)

    rows = []
    for name, expect, plan in BAD_PLANS:
        v = ver.verify(plan, world)
        codes = sorted({i.code for i in v.issues})
        caught = v.verdict == BLOCK and bool(set(expect.split("|")) & set(codes))
        rows.append({"name": name, "expected_code": expect, "verdict": v.verdict, "codes": codes, "caught": caught,
                     "reasons": [i.message for i in v.issues]})
        log.append("verify", {"plan": name, "verdict": v.verdict, "codes": codes})
    good_rows = []
    for name, plan in GOOD_PLANS:
        v = ver.verify(plan, world)
        good_rows.append({"name": name, "verdict": v.verdict, "codes": sorted({i.code for i in v.issues}),
                          "passed": v.verdict in (ALLOW, REORDER),
                          "reordered_steps": [s["skill"] for s in v.plan["steps"]] if v.verdict == REORDER else None})
        log.append("verify", {"plan": name, "verdict": v.verdict})

    chain = verify_log(log_path)
    out = {"bad_plans": len(rows), "caught": sum(r["caught"] for r in rows), "good_plans": len(good_rows),
           "good_passed": sum(r["passed"] for r in good_rows), "audit_chain": chain, "rows": rows, "good_rows": good_rows,
           "seed": 0, "split": "train"}
    (ROOT / "results" / "verifier_injection.json").write_text(json.dumps(out, indent=2))
    for r in rows:
        print(f"{'CAUGHT ' if r['caught'] else 'MISSED '} {r['name']:36s} {r['verdict']:7s} {r['codes']}")
    for r in good_rows:
        print(f"{'PASS   ' if r['passed'] else 'BLOCKED'} {r['name']:36s} {r['verdict']:7s} {r['reordered_steps'] or ''}")
    print(f"{out['caught']}/{out['bad_plans']} unsafe plans caught, {out['good_passed']}/{out['good_plans']} sane plans passed, audit chain: {chain['reason']}")


if __name__ == "__main__":
    main()
