"""Clear the table: the reverse task.

For each held-out seed the scripted expert first sets the table (drawer, fork, spoon handed to arm B), then the runtime
receives "clear the table": plan (rule planner: handoff spoon B->A, put_in_drawer x2, close_drawer), verify, execute,
check.  Success = fork and spoon back in the drawer and the drawer closed (oracles.CLEAR_TASK).  Writes results/clear_table.json.

    python -m eval.clear_table --seeds 10 --split test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym

import souschef_env  # noqa: F401
from souschef_env import oracles
from expert.primitives import Expert
from planner.plan import Planner
from runtime.state_machine import Runtime
from verifier.rules import Verifier

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--split", default="test")
    ap.add_argument("--planner", default="rules")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "clear_table.json")
    a = ap.parse_args()
    env = gym.make("souschef_env/Thali-v0", disable_env_checker=True).unwrapped
    rows = []
    for seed in range(a.seeds):
        t0 = time.time()
        rt = Runtime(env, Planner(backend=a.planner), Verifier(), camera_check=False, audit_path=ROOT / "results" / "audit_clear.jsonl")
        # the runtime resets the scene itself; hook the set-up into its reset by running it before the command on the same seed
        env.reset(seed=seed, options={"split": a.split})
        env.render_enabled = False
        ex = Expert(env)
        setup_ok = ex.open_drawer("a").ok and ex.pick_place("fork_1", "a", "fork").ok and ex.handoff_place("spoon_1", "a", "b", "spoon").ok
        if not setup_ok:
            rows.append({"seed": seed, "setup_ok": False, "success": False}); print(f"seed {seed}: setup failed", flush=True); continue
        log = rt.run_command("clear the table", seed=seed, split=a.split, reset=False)
        sg = oracles.subgoals(env.model, env.data)
        rows.append({"seed": seed, "setup_ok": True, "success": all(sg[g] for g in oracles.CLEAR_TASK), "state": rt.state,
                     "subgoals": {g: bool(sg[g]) for g in oracles.CLEAR_TASK}, "plan": [(s.step["skill"], s.arm, bool(s.oracle_ok)) for s in log.steps],
                     "replans": log.replans, "sim_steps": log.sim_steps, "wall_s": round(time.time() - t0, 1)})
        print(f"seed {seed}: {'ok' if rows[-1]['success'] else 'fail'} {rows[-1]['subgoals']} ({rows[-1]['wall_s']}s)", flush=True)
    out = {"split": a.split, "seeds": a.seeds, "planner": a.planner, "setup_ok": sum(r["setup_ok"] for r in rows),
           "successes": sum(r["success"] for r in rows), "success_rate": round(sum(r["success"] for r in rows) / max(1, len(rows)), 3),
           "per_subgoal_rate": {g: round(sum(r.get("subgoals", {}).get(g, False) for r in rows) / max(1, len(rows)), 2) for g in oracles.CLEAR_TASK}, "rows": rows}
    a.out.write_text(json.dumps(out, indent=1))
    print(f"{out['successes']}/{a.seeds} cleared -> {a.out}")


if __name__ == "__main__":
    main()
