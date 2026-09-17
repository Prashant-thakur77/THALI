"""Full-task success over seeds (plan Phase 9.1 + Phase 3.3 policy-only / +retry / +fallback).

    python -m eval.run_seeds --policy expert  --seeds 10 --split test
    python -m eval.run_seeds --policy act     --mode policy_only|policy_retry|policy_fallback --seeds 10 --split test
    python -m eval.run_seeds --policy smolvla --mode policy_fallback   (needs Prashant-77/thali_smolvla on the Hub)

Each run executes the 7-step full-task program (expert/task.FULL_TASK) through the runtime's executor on
``test_ranges`` seeds 0-9, records per-skill outcome (and which stage won it), sub-goals, sim steps and the
randomisation sample of the seed.  Writes results/seeds_<policy>_<mode>_<split>.json and refreshes the
aggregate results/seeds.json that the README table is generated from.  ``--axes`` restricts randomisation to a
subset (used by eval/heatmap.py).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

import souschef_env  # noqa: F401
from souschef_env import oracles
from souschef_env.randomize import AXES
from expert.primitives import Expert
from expert.task import FULL_TASK
from runtime.executors import PolicyExecutor
from runtime.state_machine import ExpertExecutor

ROOT = Path(__file__).resolve().parent.parent
SMOLVLA_REPO = "Prashant-77/thali_smolvla"


def smolvla_path() -> Path | None:
    """Local checkpoint dir for the Hub model, or None if it does not exist (yet)."""
    local = ROOT / "outputs" / "smolvla_mt" / "checkpoints" / "last" / "pretrained_model"
    if (local / "config.json").exists():
        return local
    try:
        from huggingface_hub import snapshot_download
        p = Path(snapshot_download(SMOLVLA_REPO))
        return p if (p / "config.json").exists() else None
    except Exception:
        return None


def make_executor(policy: str, mode: str, device: str):
    if policy == "expert":
        return lambda ex: ExpertExecutor(ex)
    if policy == "act":
        return lambda ex: PolicyExecutor(ex, kind="act", mode=mode, device=device)
    if policy == "smolvla":
        p = smolvla_path()
        if p is None:
            return None
        return lambda ex: PolicyExecutor(ex, kind="smolvla", mode=mode, device=device, smolvla_path=p)
    raise ValueError(policy)


def run(policy: str, mode: str, seeds: int, split: str, axes: tuple[str, ...], device: str, tag: str | None = None) -> dict | None:
    factory = make_executor(policy, mode, device)
    if factory is None:
        return None
    env = gym.make("souschef_env/Thali-v0", disable_env_checker=True, obs_type="pixels_agent_pos" if policy != "expert" else "state").unwrapped
    rows = []
    t_all = time.time()
    for seed in range(seeds):
        env.reset(seed=seed, options={"split": split, "axes": axes})
        sample = env.last_sample.as_dict()
        ex = Expert(env)
        if policy == "expert":
            env.render_enabled = False
        executor = factory(ex)
        t0 = time.time()
        skills = []
        for st in FULL_TASK:
            step = {"skill": st.skill, "arm": st.arm, "obj": st.obj, "zone": st.zone, "to_arm": st.arm2}
            r = executor.run(step)
            skills.append({"skill": st.skill, "obj": st.obj, "ok": bool(r.ok), "won_by": r.detail.get("won_by", "expert" if policy == "expert" else None),
                           "stages": r.detail.get("stages"), "steps": r.steps})
        sg = oracles.subgoals(env.model, env.data)
        rows.append({"seed": seed, "success": bool(all(sg[g] for g in oracles.FULL_TASK)), "subgoals": sg, "skills": skills, "sim_steps": ex.steps,
                     "seconds": round(time.time() - t0, 1), "sample": {k: sample[k] for k in ("shape", "mass", "friction", "background") if sample.get(k)}})
        print(f"[{policy}/{mode}/{split}{'/' + '+'.join(axes) if axes != AXES else ''}] seed {seed}: {'SUCCESS' if rows[-1]['success'] else 'fail'} "
              f"{sum(sg.values())}/6 | " + " ".join(f"{s['skill'][:5]}:{('' if s['ok'] else 'X')}{(s['won_by'] or '-')[:4]}" for s in skills) + f" ({rows[-1]['seconds']}s)", flush=True)
        env.render_enabled = True
    env.close()
    n = len(rows)
    per_skill = {f"{i}_{st.skill}": sum(r["skills"][i]["ok"] for r in rows) / n for i, st in enumerate(FULL_TASK)}
    per_subgoal = {k: sum(r["subgoals"][k] for r in rows) / n for k in rows[0]["subgoals"]}
    won = {}
    for r in rows:
        for s in r["skills"]:
            if s["ok"]:
                won[s["won_by"] or "expert"] = won.get(s["won_by"] or "expert", 0) + 1
    out = {"policy": policy, "mode": mode, "split": split, "axes": list(axes), "seeds": n, "successes": sum(r["success"] for r in rows),
           "success_rate": sum(r["success"] for r in rows) / n, "per_skill_rate": per_skill, "per_subgoal_rate": per_subgoal,
           "skill_successes_won_by": won, "elapsed_s": round(time.time() - t_all, 1), "rows": rows}
    name = tag or f"seeds_{policy}_{mode}_{split}" + ("" if axes == AXES else "_" + "+".join(axes))
    (ROOT / "results" / f"{name}.json").write_text(json.dumps(out, indent=2, default=_jsonable))
    print(f"{out['successes']}/{n} on {split} -> results/{name}.json")
    return out


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    raise TypeError(type(v))


def aggregate() -> None:
    """results/seeds.json: one row per (policy, mode) for the README table, plus 'pending' rows for missing runs."""
    rows = []
    for policy, mode in (("expert", "expert"), ("act", "policy_only"), ("act", "policy_retry"), ("act", "policy_fallback"),
                         ("smolvla", "policy_only"), ("smolvla", "policy_retry"), ("smolvla", "policy_fallback")):
        for split in ("train", "test"):
            f = ROOT / "results" / f"seeds_{policy}_{mode}_{split}.json"
            if f.exists():
                d = json.loads(f.read_text())
                rows.append({"policy": policy, "mode": mode, "split": split, "successes": d["successes"], "seeds": d["seeds"],
                             "success_rate": d["success_rate"], "per_subgoal_rate": d["per_subgoal_rate"], "file": f.name})
            elif policy == "smolvla" and split == "test":
                rows.append({"policy": policy, "mode": mode, "split": split, "status": "pending SmolVLA run (docs/KAGGLE_TODO.md)"})
    (ROOT / "results" / "seeds.json").write_text(json.dumps({"rows": rows}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="expert", choices=["expert", "act", "smolvla"])
    ap.add_argument("--mode", default="policy_fallback", choices=["policy_only", "policy_retry", "policy_fallback"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--split", default="test")
    ap.add_argument("--axes", nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    axes = tuple(args.axes) if args.axes else AXES
    mode = "expert" if args.policy == "expert" else args.mode
    out = run(args.policy, mode, args.seeds, args.split, axes, args.device, args.tag)
    if out is None:
        print(f"{args.policy}: checkpoint not available -> rows stay 'pending' (see docs/KAGGLE_TODO.md)")
    aggregate()


if __name__ == "__main__":
    main()
