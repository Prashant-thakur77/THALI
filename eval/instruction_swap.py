"""Instruction-swap confusion matrix (plan Phase 9.2).

Same seed, same scene; the command is varied along three axes -- which ARM, which OBJECT, and the ORDER of two
skills -- and the planner (VLM with verifier-in-the-loop, then rule fallback) must produce a plan whose (skill,
arm, obj) tuples reflect the swap.  The matrix counts, for each requested variant, which variant the plan
actually encodes.  A planner that ignores the instruction lands on the diagonal's neighbours; a language-
following one stays on the diagonal.  Writes results/instruction_swap.json.

    python -m eval.instruction_swap --seed 0 --planner auto
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym

import souschef_env  # noqa: F401
from souschef_env.scene_description import describe, scene_state
from planner.plan import Planner
from verifier.rules import Verifier, World

ROOT = Path(__file__).resolve().parent.parent

# (variant name, command, expected (skill, arm, obj) signature of the *first* relevant step)
ARM_SWAPS = [
    ("plate_arm_A", "pick up the plate with arm A and place it on its spot", ("pick_place", "a", "plate")),
    ("mug_arm_B", "pick up the mug with arm B and place it on its spot", ("pick_place", "b", "mug")),
    ("hold_B_pour_A", "hold the mug with arm B and pour water with arm A", ("pour", "a", None)),
    ("hold_A_pour_B", "hold the mug with arm A and pour water with arm B", ("pour", "b", None)),
]
OBJECT_SWAPS = [
    ("fork", "open the drawer and put a fork on the left of the plate", ("pick_place", "a", "fork_1")),
    ("spoon", "open the drawer and put a spoon on the right of the plate", ("handoff", "a", "spoon_1")),
    ("plate", "put the plate on the plate spot", ("pick_place", "a", "plate")),
    ("mug", "put the mug on the mug spot", ("pick_place", "b", "mug")),
]
ORDER_SWAPS = [
    ("plate_then_mug", "first put the plate on its spot, then put the mug on its spot", ["plate", "mug"]),
    ("mug_then_plate", "first put the mug on its spot, then put the plate on its spot", ["mug", "plate"]),
]


def first_match(plan: dict, skill: str) -> tuple | None:
    for s in plan["steps"]:
        if s["skill"] == skill:
            return (s["skill"], s["arm"], s.get("obj"))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--planner", default="auto")
    ap.add_argument("--device", default="CPU")
    args = ap.parse_args()
    env = gym.make("souschef_env/Thali-v0", disable_env_checker=True).unwrapped
    obs, _ = env.reset(seed=args.seed)
    img, st, txt = obs["pixels"]["overhead"], scene_state(env.model, env.data), describe(env.model, env.data)
    world = World.from_scene_state(st)
    ver = Verifier()
    pl = Planner(backend=args.planner, device=args.device)
    results = {"seed": args.seed, "planner_backend": pl.backend, "arm": [], "object": [], "order": [], "matrix": {}}
    for group, cases in (("arm", ARM_SWAPS), ("object", OBJECT_SWAPS)):
        names = [c[0] for c in cases]
        sigs = {c[0]: c[2] for c in cases}
        matrix = {n: {m: 0 for m in names} for n in names}
        for name, cmd, expected in cases:
            r = pl.plan(cmd, st, txt, img, verifier=ver, world=world)
            got = first_match(r.plan, expected[0])
            # which variant does the plan encode?  the one whose signature matches; else "none"
            landed = next((m for m in names if got == sigs[m]), None)
            if landed:
                matrix[name][landed] += 1
            results[group].append({"variant": name, "command": cmd, "expected": expected, "got": got, "source": r.source,
                                   "verdict": r.verdict, "correct": got == expected, "plan": [(s["skill"], s["arm"], s.get("obj")) for s in r.plan["steps"]]})
            print(f"{group:6s} {name:14s} {'OK ' if got == expected else 'X  '} {r.source:9s} {got}")
        results["matrix"][group] = matrix
    for name, cmd, order in ORDER_SWAPS:
        r = pl.plan(cmd, st, txt, img, verifier=ver, world=world)
        seq = [s.get("obj") for s in r.plan["steps"] if s["skill"] == "pick_place" and s.get("obj") in ("plate", "mug")]
        ok = seq == order
        results["order"].append({"variant": name, "command": cmd, "expected": order, "got": seq, "source": r.source, "verdict": r.verdict, "correct": ok})
        print(f"order  {name:14s} {'OK ' if ok else 'X  '} {r.source:9s} {seq}")
    n = sum(len(results[g]) for g in ("arm", "object", "order"))
    results["correct"] = sum(x["correct"] for g in ("arm", "object", "order") for x in results[g])
    results["total"] = n
    results["accuracy"] = results["correct"] / n
    results["from_vlm"] = sum(x["source"].startswith("vlm") for g in ("arm", "object", "order") for x in results[g])
    (ROOT / "results" / "instruction_swap.json").write_text(json.dumps(results, indent=2))
    print(f"instruction swap: {results['correct']}/{n} correct ({results['from_vlm']} plans from the VLM)")


if __name__ == "__main__":
    main()
