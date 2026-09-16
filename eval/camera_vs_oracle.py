"""Camera-based state judgement vs the sim oracle (plan Phase 9.3).

The runtime asks a yes/no question about the overhead frame after every skill (planner/state_check.py: the local
VLM when loaded, else the pixel heuristic).  This script runs the full task on seeds 0-9 (test split) with the
scripted expert, asks every sub-goal question after every skill for BOTH backends, and scores each against the
oracle: agreement, and precision/recall of "yes".  Writes results/camera_vs_oracle.json.

    python -m eval.camera_vs_oracle --seeds 10 [--no-vlm]
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
from expert.task import FULL_TASK
from planner.state_check import QUESTIONS, PixelHeuristic, VLMStateCheck

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--split", default="test")
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--device", default="CPU")
    args = ap.parse_args()
    backends = {"pixels": PixelHeuristic()}
    if not args.no_vlm:
        from planner.plan import MODEL_DIR, OpenVinoVLM
        if (MODEL_DIR / "openvino_language_model.xml").exists():
            backends["vlm"] = VLMStateCheck(OpenVinoVLM(MODEL_DIR, args.device))
    env = gym.make("souschef_env/Thali-v0", disable_env_checker=True).unwrapped
    judgements = []
    for seed in range(args.seeds):
        env.reset(seed=seed, options={"split": args.split})
        for b in backends.values():
            if hasattr(b, "set_reference"):
                b.set_reference(env.render_camera("overhead"))
        ex = Expert(env)
        env.render_enabled = False
        for i, st in enumerate([None, *FULL_TASK]):
            if st is not None:
                st.run(ex)
            env.render_enabled = True
            frame = env.render_camera("overhead")
            env.render_enabled = False
            truth = oracles.subgoals(env.model, env.data)
            for key in QUESTIONS:
                for name, b in backends.items():
                    t0 = time.perf_counter()
                    ans = bool(b.ask(frame, key))
                    judgements.append({"seed": seed, "after_step": i, "subgoal": key, "backend": name, "camera": ans, "oracle": bool(truth[key]),
                                       "latency_s": round(time.perf_counter() - t0, 3)})
        print(f"seed {seed}: {len(judgements)} judgements so far", flush=True)
    summary = {}
    for name in backends:
        js = [j for j in judgements if j["backend"] == name]
        tp = sum(j["camera"] and j["oracle"] for j in js)
        fp = sum(j["camera"] and not j["oracle"] for j in js)
        fn = sum((not j["camera"]) and j["oracle"] for j in js)
        agree = sum(j["camera"] == j["oracle"] for j in js)
        per_goal = {k: round(sum(j["camera"] == j["oracle"] for j in js if j["subgoal"] == k) / max(1, sum(1 for j in js if j["subgoal"] == k)), 3) for k in QUESTIONS}
        summary[name] = {"judgements": len(js), "agreement": round(agree / len(js), 3), "precision": round(tp / max(1, tp + fp), 3),
                         "recall": round(tp / max(1, tp + fn), 3), "per_subgoal_agreement": per_goal,
                         "mean_latency_s": round(sum(j["latency_s"] for j in js) / len(js), 3)}
    out = {"seeds": args.seeds, "split": args.split, "backends": summary, "judgements": judgements}
    (ROOT / "results" / "camera_vs_oracle.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
