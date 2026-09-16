"""Run the scripted expert on the full task over N seeds and write results/expert_full_task.json.

This is the expert's own success rate -- the ceiling for every learned policy in this repo and the
"scripted fallback" row of the policy-only / policy+retry / policy+fallback table.

    python -m expert.sweep --seeds 10 --split train
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
from expert.task import FULL_TASK, run_full_task

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or ROOT / "results" / f"expert_full_task_{args.split}.json"
    env = gym.make("souschef_env/Thali-v0", disable_env_checker=True, obs_type="state").unwrapped
    rows = []
    t0 = time.time()
    for seed in range(args.seeds):
        env.reset(seed=seed, options={"split": args.split})
        ex = Expert(env)
        t = time.time()
        res = run_full_task(ex)
        sg = oracles.subgoals(env.model, env.data)
        rows.append({"seed": seed, "success": bool(all(sg.values())), "subgoals": sg,
                     "skills": {f"{i}_{s.skill}": bool(r.ok) for i, (s, r) in enumerate(zip(FULL_TASK, res))},
                     "steps": ex.steps, "seconds": round(time.time() - t, 1)})
        print(f"seed {seed}: {'SUCCESS' if rows[-1]['success'] else 'fail'} {sum(sg.values())}/6 ({rows[-1]['seconds']}s)", flush=True)
    n_ok = sum(r["success"] for r in rows)
    per_skill = {k: sum(r["skills"][k] for r in rows) / len(rows) for k in rows[0]["skills"]}
    per_subgoal = {k: sum(r["subgoals"][k] for r in rows) / len(rows) for k in rows[0]["subgoals"]}
    summary = {"split": args.split, "seeds": args.seeds, "successes": n_ok, "success_rate": n_ok / args.seeds,
               "per_skill_rate": per_skill, "per_subgoal_rate": per_subgoal, "elapsed_s": round(time.time() - t0, 1), "rows": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"{n_ok}/{args.seeds} full-task successes on {args.split} -> {out}")


if __name__ == "__main__":
    main()
