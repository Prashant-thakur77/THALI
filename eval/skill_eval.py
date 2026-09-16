"""Per-skill learned-policy evaluation from task-consistent start states.

For each skill the scene is reset on a held-out seed, the prerequisite skills are run by the scripted expert
(unrendered, exactly as the demos were recorded), and then the learned policy alone (no retry, no fallback) must
finish the skill within its step budget.  Success is the simulator oracle; for pick/place skills the final distance
from the zone centre is reported too, so a near-miss is visible.  Writes results/skill_eval_<tag>.json.

    python -m eval.skill_eval --act-root outputs --tag 1050ep --seeds 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

import souschef_env  # noqa: F401
from souschef_env import constants as C
from souschef_env import oracles
from expert.make_demos import SKILLS, run_skill
from expert.primitives import Expert
from runtime.executors import PolicyExecutor, skill_done

ROOT = Path(__file__).resolve().parent.parent


def step_for(skill: str) -> dict:
    k = SKILLS[skill]
    if skill == "open_drawer":
        return {"skill": "open_drawer", "arm": k["arm"]}
    if skill.startswith("pick_place"):
        return {"skill": "pick_place", "arm": k["arm"], "obj": k["obj"], "zone": k["zone"]}
    if skill == "handoff_spoon":
        return {"skill": "handoff", "arm": k["arm"], "to_arm": k["arm2"], "obj": k["obj"], "zone": k["zone"]}
    if skill == "hold_mug":
        return {"skill": "hold_mug", "arm": k["arm"]}
    return {"skill": "pour", "arm": k["arm"]}


def zone_error_cm(env, step: dict) -> float | None:
    if not step.get("zone"):
        return None
    (zx, zy), _ = C.ZONES[step["zone"]]
    p = oracles.object_pos(env.model, env.data, step["obj"])
    return round(float(np.hypot(p[0] - zx, p[1] - zy)) * 100, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-root", type=Path, default=ROOT / "outputs")
    ap.add_argument("--kind", default="act", choices=["act", "smolvla"])
    ap.add_argument("--smolvla-path", type=Path, default=None)
    ap.add_argument("--tag", default="act")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed-base", type=int, default=500)
    ap.add_argument("--split", default="test")
    ap.add_argument("--skills", nargs="+", default=list(SKILLS))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    env = gym.make("souschef_env/Thali-v0", disable_env_checker=True, obs_type="pixels_agent_pos").unwrapped
    out = {"tag": a.tag, "kind": a.kind, "act_root": str(a.act_root), "split": a.split, "seeds": a.seeds, "skills": {}}
    for skill in a.skills:
        rows = []
        step = step_for(skill)
        for i in range(a.seeds):
            seed = a.seed_base + i
            env.reset(seed=seed, options={"split": a.split})
            ex = Expert(env)
            env.render_enabled = False
            if not all(run_skill(ex, p).ok for p in SKILLS[skill]["prereq"]):
                rows.append({"seed": seed, "prereq_failed": True, "ok": False})
                continue
            env.render_enabled = True
            pe = PolicyExecutor(ex, kind=a.kind, mode="policy_only", device=a.device, act_root=a.act_root, smolvla_path=a.smolvla_path)
            pol = pe._policy_for(skill)
            if pol is None:
                rows.append({"seed": seed, "no_policy": True, "ok": False})
                continue
            t0 = time.time()
            ok, n, dt = pe._rollout(pol, step, skill)
            rows.append({"seed": seed, "ok": bool(ok), "steps": int(n), "policy_s": round(dt, 1), "zone_error_cm": zone_error_cm(env, step),
                         "oracle": bool(skill_done(ex, step)), "wall_s": round(time.time() - t0, 1)})
            print(f"{skill} seed {seed}: {'ok' if ok else 'fail'} in {n} steps  err={rows[-1]['zone_error_cm']} cm", flush=True)
        n_ok = sum(r["ok"] for r in rows)
        errs = [r["zone_error_cm"] for r in rows if r.get("zone_error_cm") is not None]
        out["skills"][skill] = {"successes": n_ok, "total": len(rows), "rate": round(n_ok / max(len(rows), 1), 3),
                                "within_1_5cm": sum(e < 1.5 for e in errs) if errs else None, "median_zone_error_cm": round(float(np.median(errs)), 2) if errs else None,
                                "rows": rows}
        print(f"== {skill}: {n_ok}/{len(rows)}", flush=True)
    path = ROOT / "results" / f"skill_eval_{a.tag}.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
