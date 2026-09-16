"""Nominal / disturbed overhead frames from *real* expert runs (the states the runtime actually checks).

For each seed the scene is reset (the reference frame), then the scripted expert runs drawer → plate → fork → mug,
parking after each skill; the frame after every successful skill is a nominal sample, saved as the
|frame - reference| table crop (anomaly/crop.py).  Disturbances (spill, tipped mug, knocked plate, fallen bottle,
dropped cutlery) are applied on top of one of those real post-skill states for the held-out test seeds.  Compared with
make_data.py (teleported objects, arms never moved) this covers the real drawer travel, arm rest poses and jaw states.

    python -m anomaly.make_data_real --split train --seeds 60 --shard 0 2
    python -m anomaly.make_data_real --split test  --seeds 15 --bad 15
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

import souschef_env  # noqa: F401
from anomaly.crop import diff_from_reference
from anomaly.make_data import ANOMALIES, anomaly_variant
from expert.make_demos import SKILLS, run_skill
from expert.primitives import Expert

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "anomaly" / "data_diffreal"
SEQUENCE = ("open_drawer", "pick_place_plate", "pick_place_fork", "pick_place_mug")


def run_prefix(env, ex: Expert, k: int) -> bool:
    """Run the first k skills of the sequence; False if any failed (that frame is not a clean nominal sample)."""
    for skill in SEQUENCE[:k]:
        r = run_skill(ex, skill)
        ex.park(SKILLS[skill]["arm"])
        ex.settle(0.3)
        if not r.ok:
            return False
    return True


def save(env, ref: np.ndarray, split: str, cls: str, name: str) -> None:
    img = diff_from_reference(env.render_camera("overhead"), ref)
    p = DATA / split / cls
    p.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(p / f"{name}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--bad", type=int, default=0, help="test only: disturbed frames per anomaly type")
    ap.add_argument("--shard", type=int, nargs=2, default=(0, 1), metavar=("I", "N"))
    a = ap.parse_args()
    env = gym.make("souschef_env/Thali-v0", disable_env_checker=True).unwrapped
    m, d = env.model, env.data
    base = 400_000 if a.split == "train" else 500_000
    n_good = n_bad = 0
    for i in range(a.seeds):
        if i % a.shard[1] != a.shard[0]:
            continue
        seed = base + i
        env.reset(seed=seed, options={"split": a.split})
        ref = env.render_camera("overhead")
        ex = Expert(env)
        env.render_enabled = False
        for k, skill in enumerate(SEQUENCE, start=1):
            r = run_skill(ex, skill)
            ex.park(SKILLS[skill]["arm"])
            ex.settle(0.3)
            if not r.ok:
                break
            save(env, ref, a.split, "good", f"s{seed}_k{k}")
            n_good += 1
    if a.split == "test" and a.bad:
        for t, kind in enumerate(ANOMALIES):
            for i in range(a.bad):
                if i % a.shard[1] != a.shard[0]:
                    continue
                seed = 600_000 + t * 1000 + i
                rng = random.Random(seed)
                env.reset(seed=seed, options={"split": "test"})
                ref = env.render_camera("overhead")
                ex = Expert(env)
                env.render_enabled = False
                k = rng.randint(1, len(SEQUENCE))
                if not run_prefix(env, ex, k):
                    k = 0  # disturbance on the untouched layout is still a valid disturbed sample
                anomaly_variant(m, d, kind, rng)
                save(env, ref, "test", kind, f"s{seed}_k{k}")
                n_bad += 1
    print(json.dumps({"split": a.split, "shard": a.shard, "good": n_good, "bad": n_bad}))


if __name__ == "__main__":
    main()
