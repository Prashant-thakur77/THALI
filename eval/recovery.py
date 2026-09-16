"""Mid-task perturbation recovery (self-correction evidence).

The runtime executes ``open the top drawer, put the plate on the table with arm A, then put the fork on the table``.
Right after the plate step passes its check, the plate is knocked 10 cm off its zone (freejoint teleport, as if a
person bumped it) while the fork step proceeds.  The final-state verification in ``Runtime.run_command`` must notice
the plate goal no longer holds, redo exactly that step, and end with every goal satisfied.  Writes results/recovery.json.

    python -m eval.recovery --seeds 0 1 2 3 --out results/recovery.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

import souschef_env  # noqa: F401
from souschef_env import oracles
from planner.plan import Planner
from runtime.state_machine import Runtime
from verifier.rules import Verifier

ROOT = Path(__file__).resolve().parent.parent
COMMAND = "open the top drawer, put the plate on the table with arm A, then put the fork on the table"
KNOCK_TO = (-0.08, 0.0)  # 10 cm from the plate zone (0, -0.10), still inside arm A's reach


def knock_plate(env) -> dict:
    m, d = env.model, env.data
    adr = m.jnt_qposadr[m.joint("plate_free").id]
    before = d.qpos[adr:adr + 3].copy()
    d.qpos[adr:adr + 2] = KNOCK_TO
    d.qpos[adr + 2] = 0.006
    vadr = m.jnt_dofadr[m.joint("plate_free").id]
    d.qvel[vadr:vadr + 6] = 0.0
    mujoco.mj_forward(m, d)
    return {"from": [round(float(x), 3) for x in before], "to": [*KNOCK_TO, 0.006],
            "plate_in_zone_after_knock": oracles.object_in_zone(m, d, "plate", "plate")}


def run_seed(seed: int, split: str, planner: str) -> dict:
    env = gym.make("souschef_env/Thali-v0", disable_env_checker=True).unwrapped
    rt = Runtime(env, Planner(backend=planner), Verifier(), audit_path=ROOT / "results" / "audit_recovery.jsonl")
    knock: dict = {}

    def after_check(idx: int, step: dict) -> None:
        if not knock and step["skill"] == "pick_place" and step.get("obj") == "plate":
            knock.update(knock_plate(env) | {"after_step": idx})

    rt.after_check = after_check
    t0 = time.time()
    log = rt.run_command(COMMAND, seed=seed, split=split)
    sg = oracles.subgoals(env.model, env.data)
    return {"seed": seed, "split": split, "knock": knock, "detected": bool(log.final_checks),
            "final_checks": log.final_checks, "replans": log.replans, "state": rt.state,
            "plate_placed_final": bool(sg["plate_placed"]), "fork_placed_final": bool(sg["fork_placed"]), "drawer_open_final": bool(sg["drawer_open"]),
            "recovered": bool(knock) and bool(log.final_checks) and bool(sg["plate_placed"]) and bool(sg["fork_placed"]),
            "steps": [{"skill": s.step["skill"], "arm": s.arm, "oracle_ok": bool(s.oracle_ok)} for s in log.steps], "wall_s": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--split", default="test")
    ap.add_argument("--planner", default="rules")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "recovery.json")
    a = ap.parse_args()
    rows = [run_seed(s, a.split, a.planner) for s in a.seeds]
    out = {"command": COMMAND, "perturbation": f"plate teleported to {KNOCK_TO} after its step passed its check",
           "seeds": a.seeds, "split": a.split, "planner": a.planner,
           "knocked": sum(bool(r["knock"]) for r in rows), "detected": sum(r["detected"] for r in rows),
           "recovered": sum(r["recovered"] for r in rows), "total": len(rows), "rows": rows}
    a.out.parent.mkdir(exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(f"knocked {out['knocked']}/{out['total']}  detected {out['detected']}/{out['total']}  recovered {out['recovered']}/{out['total']}")


if __name__ == "__main__":
    main()
