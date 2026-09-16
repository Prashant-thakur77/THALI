"""Planner evaluation: commands -> VLM plan -> verifier, with retry and rule fallback (results/planner_eval.json).

Records, per command: which source produced the accepted plan (vlm / vlm_retry / rules), the verifier verdict
at each attempt, latency and tokens/s of the local Qwen2-VL-2B INT4 on the chosen OpenVINO device, and whether
the accepted plan matches the reference (rule-planner) skill sequence.  This is the number behind "local VLM
planner" in the README; the instruction-swap matrix (eval/instruction_swap.py) builds on the same call.

    python -m eval.planner_eval --device CPU
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym

import souschef_env  # noqa: F401
from souschef_env.scene_description import describe, scene_state
from planner.plan import Planner, rule_plan
from verifier.rules import Verifier, World

ROOT = Path(__file__).resolve().parent.parent
COMMANDS = [
    "Open the top drawer, pick up the plate with arm A, place it on the table, pick up the mug with arm B, pour water into the mug with arm A",
    "set the table",
    "open the drawer and put a fork next to the plate",
    "pour a little water, gently",
    "hand the spoon to arm B and put it right of the plate",
    "put the plate on the table with arm A",
    "arm B, put the mug on its spot",
    "hold the mug with arm B and pour with arm A",
]


def skills(plan: dict) -> list[str]:
    return [s["skill"] for s in plan["steps"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="auto")
    args = ap.parse_args()
    env = gym.make("souschef_env/Thali-v0", disable_env_checker=True).unwrapped
    obs, _ = env.reset(seed=args.seed)
    img, st, txt = obs["pixels"]["overhead"], scene_state(env.model, env.data), describe(env.model, env.data)
    world = World.from_scene_state(st)
    ver = Verifier()
    t0 = time.time()
    pl = Planner(backend=args.backend, device=args.device)
    load_s = time.time() - t0
    rows = []
    for cmd in COMMANDS:
        r = pl.plan(cmd, st, txt, img, verifier=ver, world=world)
        ref = skills(ver.verify(rule_plan(cmd, st), world).plan or rule_plan(cmd, st))
        requested = [k for k in ref if k != "place_mug"]  # setting the mug down is a courtesy, not a request
        got = skills(r.plan)
        rows.append({"command": cmd, "source": r.source, "verdict": r.verdict, "latency_s": round(r.latency_s, 2),
                     "vlm_metrics": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.vlm_metrics.items()},
                     "attempts": [{k: v for k, v in a.items() if k != "raw"} for a in r.attempts],
                     "skills": got, "reference_skills": ref, "covers_requested": all(k in got for k in requested),
                     "matches_reference": got == ref, "mode": r.plan.get("mode")})
        print(f"{r.source:9s} {r.verdict or '-':7s} {r.latency_s:5.1f}s {'covers' if rows[-1]['covers_requested'] else 'MISSES'} | {cmd[:50]}")
    n = len(rows)
    vlm_rows = [r for r in rows if r["source"].startswith("vlm")]
    out = {"device": args.device, "backend": pl.backend, "model": "Qwen/Qwen2-VL-2B-Instruct INT4 (optimum-cli export, group 128)",
           "hardware": "Intel Core i7-13650HX CPU (Raptor Lake UHD iGPU unstable for VLM: see docs/BLOCKERS.md); no NPU",
           "load_s": round(load_s, 1), "commands": n,
           "accepted_from_vlm": len(vlm_rows), "accepted_from_rules": n - len(vlm_rows),
           "vlm_first_try_accepted": sum(r["source"] == "vlm" for r in rows),
           "match_reference_rate": sum(r["matches_reference"] for r in rows) / n,
           "covers_requested_rate": sum(r["covers_requested"] for r in rows) / n,
           "verifier_approved_rate": sum(r["verdict"] in ("ALLOW", "REORDER") for r in rows) / n,
           "mean_latency_s": round(sum(r["latency_s"] for r in rows) / n, 2),
           "vlm_tokens_per_s": round(sum(r["vlm_metrics"].get("tokens_per_s") or 0 for r in rows if r["vlm_metrics"].get("tokens_per_s")) / max(1, sum(1 for r in rows if r["vlm_metrics"].get("tokens_per_s"))), 1),
           "vlm_ttft_ms": round(sum(r["vlm_metrics"].get("ttft_ms") or 0 for r in rows if r["vlm_metrics"].get("ttft_ms")) / max(1, sum(1 for r in rows if r["vlm_metrics"].get("ttft_ms")))),
           "rows": rows}
    (ROOT / "results" / "planner_eval.json").write_text(json.dumps(out, indent=2))
    print(f"accepted from VLM {out['accepted_from_vlm']}/{n} (first try {out['vlm_first_try_accepted']}), rules {out['accepted_from_rules']}/{n}; "
          f"covers requested {out['covers_requested_rate']:.2f}, verifier-approved {out['verifier_approved_rate']:.2f}; {out['vlm_tokens_per_s']} tok/s, TTFT {out['vlm_ttft_ms']} ms on {args.device}")


if __name__ == "__main__":
    main()
