"""Record scripted-expert demonstrations into a LeRobotDataset v3.0 (plan Phase 2.3).

Per skill: N episodes, each from a task-consistent start state (prerequisite skills run unrecorded),
seeded under ``train_ranges``, with a random instruction paraphrase (expert/instructions.py) and, for
10 % of pick-based episodes, a deliberate first-grasp miss followed by the expert's own retry (recovery demos).
Only oracle-verified successes are written.  Every frame carries the three camera views, the 12-D joint
state and the 12-D action the expert actually sent.  Counts and the expert's own success rate per skill go
to results/demos.json; episode-index lists per skill let per-skill ACT baselines select their subset.

    python -m expert.make_demos --episodes 60 --root data/lerobot/thali_all --repo-id Prashant-77/thali_all [--push]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from dotenv import load_dotenv

import souschef_env  # noqa: F401
from souschef_env import constants as C
from souschef_env import oracles
from expert.instructions import instruction
from expert.primitives import Expert, SkillResult

ROOT = Path(__file__).resolve().parent.parent
FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (C.N_ACTIONS,), "names": list(C.JOINTS)},
    "action": {"dtype": "float32", "shape": (C.N_ACTIONS,), "names": list(C.ACTIONS)},
    **{f"observation.images.{cam}": {"dtype": "video", "shape": (C.IMAGE_HEIGHT, C.IMAGE_WIDTH, 3), "names": ["height", "width", "channels"]}
       for cam in C.CAMERAS},
}

# skill -> (arm, obj, zone, arm2, prerequisites)
SKILLS: dict[str, dict] = {
    "open_drawer": {"arm": "a", "prereq": []},
    "pick_place_fork": {"arm": "a", "obj": "fork_1", "zone": "fork", "prereq": ["open_drawer"]},
    "pick_place_plate": {"arm": "a", "obj": "plate", "zone": "plate", "prereq": []},
    "pick_place_mug": {"arm": "b", "obj": "mug", "zone": "mug", "prereq": []},
    "handoff_spoon": {"arm": "a", "arm2": "b", "obj": "spoon_1", "zone": "spoon", "prereq": ["open_drawer"]},
    "hold_mug": {"arm": "b", "prereq": []},
    "pour": {"arm": "a", "prereq": ["hold_mug"]},
}
RECOVERY_FRACTION = 0.10


def run_skill(ex: Expert, name: str, miss: float = 0.0) -> SkillResult:
    k = SKILLS[name]
    if name == "open_drawer":
        return ex.open_drawer(k["arm"])
    if name.startswith("pick_place"):
        return ex.pick_place(k["obj"], k["arm"], k["zone"], miss_offset=miss)
    if name == "handoff_spoon":
        return ex.handoff_place(k["obj"], k["arm"], k["arm2"], k["zone"])
    if name == "hold_mug":
        return ex.hold_mug(k["arm"])
    if name == "pour":
        return ex.pour(k["arm"])
    raise ValueError(name)


def skill_instruction(name: str, rng: random.Random) -> str:
    k = SKILLS[name]
    base = "pick_place" if name.startswith("pick_place") else ("handoff" if name == "handoff_spoon" else name)
    return instruction(base, rng, k["arm"], k.get("obj"), k.get("zone"), k.get("arm2"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60, help="target successful episodes per skill")
    ap.add_argument("--skills", nargs="*", default=list(SKILLS))
    ap.add_argument("--root", type=Path, default=ROOT / "data" / "lerobot" / "thali_all")
    ap.add_argument("--repo-id", default=f"{os.environ.get('HF_USER', 'Prashant-77')}/thali_all")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "demos.json")
    ap.add_argument("--seed-base", type=int, default=100000)
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if args.root.exists():
        shutil.rmtree(args.root)
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=C.FPS, features=FEATURES, root=args.root,
                               robot_type="so101_bimanual_sim", use_videos=True, image_writer_threads=4)

    env = gym.make("souschef_env/Thali-v0", disable_env_checker=True, obs_type="pixels_agent_pos").unwrapped
    stats = {s: {"attempts": 0, "successes": 0, "frames": 0, "recovery": 0, "episodes": []} for s in args.skills}
    t0 = time.time()
    ep_index = 0
    for skill in args.skills:
        si = list(SKILLS).index(skill)  # stable per-skill seed stream regardless of sharding
        rng = random.Random(1000 + si)
        seed = args.seed_base + si * 10000
        while stats[skill]["successes"] < args.episodes:
            seed += 1
            stats[skill]["attempts"] += 1
            env.reset(seed=seed, options={"split": "train"})
            # prerequisites, unrecorded and unrendered
            ex = Expert(env)
            env.render_enabled = False
            ok_pre = all(run_skill(ex, p).ok for p in SKILLS[skill]["prereq"])
            env.render_enabled = True
            if not ok_pre:
                continue
            text = skill_instruction(skill, rng)
            recovery = skill.startswith("pick_place") and rng.random() < RECOVERY_FRACTION
            frames: list[dict] = []

            def record(action: np.ndarray, obs: dict) -> None:
                frames.append({
                    "observation.state": obs["agent_pos"].astype(np.float32),
                    "action": action.astype(np.float32),
                    **{f"observation.images.{cam}": obs["pixels"][cam] for cam in C.CAMERAS},
                    "task": text,
                })

            ex.on_step = record
            r = run_skill(ex, skill, miss=0.02 if recovery else 0.0)
            ex.on_step = None
            if not r.ok or len(frames) < 20:
                continue
            for f in frames:
                ds.add_frame(f)
            ds.save_episode()
            st = stats[skill]
            st["successes"] += 1
            st["frames"] += len(frames)
            st["recovery"] += int(recovery)
            st["episodes"].append(ep_index)
            ep_index += 1
            print(f"[{time.time() - t0:6.0f}s] {skill:17s} seed {seed} ok  {len(frames):4d} frames  "
                  f"({st['successes']}/{args.episodes}, expert rate {st['successes'] / st['attempts']:.2f}) '{text}'", flush=True)
    ds.finalize()
    summary = {
        "repo_id": args.repo_id, "root": str(args.root), "fps": C.FPS, "cameras": list(C.CAMERAS),
        "image_hw": [C.IMAGE_HEIGHT, C.IMAGE_WIDTH], "split": "train", "recovery_fraction": RECOVERY_FRACTION,
        "total_episodes": ep_index, "total_frames": sum(s["frames"] for s in stats.values()),
        "elapsed_s": round(time.time() - t0, 1),
        "skills": {k: {**v, "expert_success_rate": round(v["successes"] / max(1, v["attempts"]), 3)} for k, v in stats.items()},
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "episodes"} for k, v in summary["skills"].items()}, indent=1))
    if args.push:
        if not (os.environ.get("HF_TOKEN") or (Path.home() / ".cache" / "huggingface" / "token").exists()):
            print("No HF token: dataset NOT pushed (see docs/KAGGLE_TODO.md)")
        else:
            ds.push_to_hub(tags=["thali", "so101", "bimanual", "mujoco"], private=False)
            print("pushed", args.repo_id)


if __name__ == "__main__":
    main()
