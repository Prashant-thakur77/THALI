"""Render an MVTec-style folder dataset of overhead table states for Anomalib.

Nominal ("good") frames are physically settled table states the runtime expects to see: the randomised initial layout
(train split), and the same layout with objects settled into their zones with the drawer open or closed.  Anomalous
frames (test only) are the disturbances a person or a failed grasp produces: water spilled on the table, the mug
tipped over, the plate knocked off its zone toward the table edge, the bottle fallen, cutlery dropped on the table.
Nominal frames use the train split's randomisation ranges; every test frame (good and anomalous) uses held-out seeds
and the test ranges, so the detector is scored on layouts, lighting and backgrounds it never saw.

    python -m anomaly.make_data --train 240 --test-good 60 --test-bad 15   # 15 per anomaly type
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from PIL import Image

import souschef_env  # noqa: F401
from souschef_env import constants as C

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "anomaly" / "data"
ANOMALIES = ("spill", "tipped_mug", "knocked_plate", "fallen_bottle", "dropped_cutlery")


def _set_free(m, d, joint: str, xy=None, z=None, quat=None) -> None:
    adr = m.jnt_qposadr[m.joint(joint).id]
    if xy is not None:
        d.qpos[adr:adr + 2] = xy
    if z is not None:
        d.qpos[adr + 2] = z
    if quat is not None:
        d.qpos[adr + 3:adr + 7] = quat
    v = m.jnt_dofadr[m.joint(joint).id]
    d.qvel[v:v + 6] = 0.0


def _settle(m, d, n: int = 300) -> None:
    mujoco.mj_forward(m, d)
    for _ in range(n):
        mujoco.mj_step(m, d)


def _drawer(m, d, open_: bool) -> None:
    adr = m.jnt_qposadr[m.joint("drawer_slide").id]
    d.qpos[adr] = C.DRAWER_OPEN_Q if open_ and hasattr(C, "DRAWER_OPEN_Q") else (0.12 if open_ else 0.0)


def nominal_variant(m, d, rng: random.Random) -> str:
    """Move objects into zones like a completed step would leave them; drawer open or closed."""
    kind = rng.choice(["initial", "plate_placed", "plate_mug_placed", "all_placed"])
    if kind != "initial":
        (zx, zy), _ = C.ZONES["plate"]
        _set_free(m, d, "plate_free", xy=(zx + rng.uniform(-0.01, 0.01), zy + rng.uniform(-0.01, 0.01)), z=0.006)
    if kind in ("plate_mug_placed", "all_placed"):
        (zx, zy), _ = C.ZONES["mug"]
        _set_free(m, d, "mug_free", xy=(zx + rng.uniform(-0.01, 0.01), zy + rng.uniform(-0.01, 0.01)), z=0.03)
    if kind == "all_placed":
        for obj, zone in (("fork_1", "fork"), ("spoon_1", "spoon")):
            (zx, zy), _ = C.ZONES[zone]
            _set_free(m, d, f"{obj}_free", xy=(zx + rng.uniform(-0.01, 0.01), zy + rng.uniform(-0.01, 0.01)), z=0.02,
                      quat=(np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)))
    _drawer(m, d, rng.random() < 0.5)
    _settle(m, d)
    return kind


def anomaly_variant(m, d, kind: str, rng: random.Random) -> None:
    if kind == "spill":
        (zx, zy), _ = C.ZONES["mug"]
        for i in rng.sample(range(C.N_WATER), 12):
            _set_free(m, d, f"water_{i}_free", xy=(zx + rng.uniform(-0.06, 0.06), zy + rng.uniform(-0.06, 0.06)), z=0.004)
    elif kind == "tipped_mug":
        (zx, zy), _ = C.ZONES["mug"]
        _set_free(m, d, "mug_free", xy=(zx + rng.uniform(-0.02, 0.02), zy + rng.uniform(-0.02, 0.02)), z=0.05,
                  quat=(np.cos(np.pi / 4), np.sin(np.pi / 4) * rng.choice([-1, 1]), 0, 0))
    elif kind == "knocked_plate":
        _set_free(m, d, "plate_free", xy=(rng.uniform(-0.05, 0.05), rng.uniform(-0.30, -0.25)), z=0.006)
    elif kind == "fallen_bottle":
        _set_free(m, d, "bottle_free", z=0.06, quat=(np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0))
    elif kind == "dropped_cutlery":
        for obj in ("fork_1", "spoon_1"):
            _set_free(m, d, f"{obj}_free", xy=(rng.uniform(-0.15, 0.15), rng.uniform(-0.05, 0.10)), z=0.02,
                      quat=(np.cos(rng.uniform(0, np.pi)), 0, 0, np.sin(rng.uniform(0, np.pi))))
    _settle(m, d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=240)
    ap.add_argument("--test-good", type=int, default=60)
    ap.add_argument("--test-bad", type=int, default=15, help="per anomaly type")
    a = ap.parse_args()
    env = gym.make("souschef_env/Thali-v0", disable_env_checker=True).unwrapped
    m, d = env.model, env.data
    manifest = {"train_good": 0, "test_good": 0, "test_bad": {k: 0 for k in ANOMALIES}, "image_size": [240, 320], "camera": "overhead"}

    def save(split: str, cls: str, idx: int) -> None:
        img = env.render_camera("overhead")
        p = DATA / split / cls; p.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(p / f"{idx:04d}.png")

    for i in range(a.train):
        env.reset(seed=100_000 + i, options={"split": "train"})
        nominal_variant(m, d, random.Random(i)); save("train", "good", i); manifest["train_good"] += 1
    for i in range(a.test_good):
        env.reset(seed=200_000 + i, options={"split": "test"})
        nominal_variant(m, d, random.Random(1000 + i)); save("test", "good", i); manifest["test_good"] += 1
    for k in ANOMALIES:
        for i in range(a.test_bad):
            env.reset(seed=300_000 + i + 1000 * ANOMALIES.index(k), options={"split": "test"})
            rng = random.Random(2000 + i)
            nominal_variant(m, d, rng); anomaly_variant(m, d, k, rng); save("test", k, i); manifest["test_bad"][k] += 1
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(manifest)


if __name__ == "__main__":
    main()
