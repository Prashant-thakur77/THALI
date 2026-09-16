"""results/demos.json is self-consistent, and the recorded dataset (if present locally) loads with the right features."""

import json
from pathlib import Path

import pytest

from souschef_env import constants as C

ROOT = Path(__file__).resolve().parent.parent


def test_demos_results_consistent():
    d = json.loads((ROOT / "results" / "demos.json").read_text())
    assert d["total_episodes"] == sum(v["successes"] for v in d["skills"].values())
    assert d["total_frames"] == sum(v["frames"] for v in d["skills"].values())
    assert set(d["skills"]) == {"open_drawer", "pick_place_fork", "pick_place_plate", "pick_place_mug", "handoff_spoon", "hold_mug", "pour"}
    for v in d["skills"].values():
        assert 0 < v["expert_success_rate"] <= 1.0 and len(v["episodes"]) == v["successes"]
    assert d["cameras"] == list(C.CAMERAS)


def test_expert_sweeps_exist():
    for split in ("train", "test"):
        d = json.loads((ROOT / "results" / f"expert_full_task_{split}.json").read_text())
        assert d["seeds"] == 10 and len(d["rows"]) == 10
        assert d["success_rate"] == d["successes"] / 10


@pytest.mark.skipif(not (ROOT / "data" / "lerobot" / "thali_all" / "meta").exists(), reason="dataset not recorded locally")
def test_dataset_loads():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("Prashant-77/thali_all", root=ROOT / "data" / "lerobot" / "thali_all")
    d = json.loads((ROOT / "results" / "demos.json").read_text())
    assert ds.num_episodes == d["total_episodes"]
    x = ds[0]
    assert tuple(x["action"].shape) == (C.N_ACTIONS,) and tuple(x["observation.state"].shape) == (C.N_ACTIONS,)
    for cam in C.CAMERAS:
        assert tuple(x[f"observation.images.{cam}"].shape) == (3, C.IMAGE_HEIGHT, C.IMAGE_WIDTH)
