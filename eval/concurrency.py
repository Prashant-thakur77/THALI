"""Sequential vs concurrent two-arm execution (plan Phase 7 extension).

The same commands run on the same seeds twice: one skill at a time (the default) and with ``Runtime(concurrent=True)``,
where independent steps for the two arms (see ``ArmQueues.ready_pair``) are driven simultaneously through the
expert's step barrier.  Reports success and the deterministic simulator step count per mode; the wall-clock is
also recorded but depends on machine load.  Writes results/concurrency.json.

    python -m eval.concurrency --seeds 0 1 2 3 4 --split test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym

import souschef_env  # noqa: F401
from planner.plan import Planner
from runtime.state_machine import Runtime
from verifier.rules import Verifier

ROOT = Path(__file__).resolve().parent.parent
COMMANDS = {
    "drawer_and_mug": "open the top drawer with arm A and put the mug on its spot with arm B",
    "plate_and_mug": "put the plate on its spot with arm A and put the mug on its spot with arm B",
}


def run(command: str, seed: int, split: str, concurrent: bool, planner: str) -> dict:
    env = gym.make("souschef_env/Thali-v0", disable_env_checker=True).unwrapped
    rt = Runtime(env, Planner(backend=planner), Verifier(), camera_check=False, concurrent=concurrent,
                 audit_path=ROOT / "results" / "audit_concurrency.jsonl")
    t0 = time.time()
    log = rt.run_command(command, seed=seed, split=split)
    return {"seed": seed, "concurrent": concurrent, "success": bool(log.success), "state": rt.state, "sim_steps": int(log.sim_steps),
            "wall_s": round(time.time() - t0, 1), "steps_concurrent": sum(1 for s in log.steps if getattr(s, "concurrent", False)),
            "skills": [(s.step["skill"], s.arm, bool(s.oracle_ok)) for s in log.steps]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--split", default="test")
    ap.add_argument("--planner", default="rules")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "concurrency.json")
    a = ap.parse_args()
    out = {"split": a.split, "seeds": a.seeds, "planner": a.planner, "commands": {}}
    for name, cmd in COMMANDS.items():
        rows = {"sequential": [], "concurrent": []}
        for seed in a.seeds:
            for conc in (False, True):
                r = run(cmd, seed, a.split, conc, a.planner)
                rows["concurrent" if conc else "sequential"].append(r)
                print(f"{name} seed {seed} {'concurrent' if conc else 'sequential'}: success={r['success']} sim_steps={r['sim_steps']} wall={r['wall_s']}s", flush=True)
        summ = {}
        for mode, rs in rows.items():
            summ[mode] = {"successes": sum(r["success"] for r in rs), "total": len(rs),
                          "mean_sim_steps": round(sum(r["sim_steps"] for r in rs) / len(rs), 1),
                          "mean_wall_s": round(sum(r["wall_s"] for r in rs) / len(rs), 1)}
        summ["sim_step_reduction"] = round(1 - summ["concurrent"]["mean_sim_steps"] / summ["sequential"]["mean_sim_steps"], 3)
        summ["pairs_run_concurrently"] = sum(r["steps_concurrent"] > 0 for r in rows["concurrent"])
        out["commands"][name] = {"command": cmd, "summary": summ, "rows": rows}
    a.out.write_text(json.dumps(out, indent=1))
    for name, c in out["commands"].items():
        s = c["summary"]
        print(f"{name}: sequential {s['sequential']['successes']}/{s['sequential']['total']} ({s['sequential']['mean_sim_steps']} steps) · "
              f"concurrent {s['concurrent']['successes']}/{s['concurrent']['total']} ({s['concurrent']['mean_sim_steps']} steps) · "
              f"{100 * s['sim_step_reduction']:.0f}% fewer sim steps")


if __name__ == "__main__":
    main()
