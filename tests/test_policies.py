"""Phase 3: training configs, executor plumbing (no checkpoint needed), and the trained checkpoints when present."""

import json
from pathlib import Path

import pytest

from runtime.executors import LANG_INSTRUCTION, MAX_STEPS, SKILL_OF, skill_name

ROOT = Path(__file__).resolve().parent.parent
SKILLS = ["open_drawer", "pick_place_fork", "pick_place_plate", "pick_place_mug", "handoff_spoon", "hold_mug", "pour"]


def test_config_files_exist_and_match_dataset_skills():
    demos = json.loads((ROOT / "results" / "demos.json").read_text())
    assert set(demos["skills"]) == set(SKILLS)
    for s in SKILLS:
        y = (ROOT / "policies" / f"act_{s}.yaml").read_text()
        assert "batch_size: 8" in y and "type: act" in y
    assert (ROOT / "policies" / "smolvla_multitask.yaml").exists()
    nb = json.loads((ROOT / "policies" / "kaggle_smolvla.ipynb").read_text())
    src = "".join("".join(c["source"]) for c in nb["cells"])
    assert "lerobot/smolvla_base" in src and "Prashant-77/thali_all" in src and "Prashant-77/thali_smolvla" in src


def test_plan_steps_map_to_dataset_skills():
    from expert.task import FULL_TASK
    for st in FULL_TASK:
        step = {"skill": st.skill, "arm": st.arm, "obj": st.obj, "zone": st.zone, "to_arm": st.arm2}
        name = skill_name(step)
        if st.skill == "place_mug":
            assert name is None  # executed by the expert (a courtesy step, not a trained skill)
        else:
            assert name in SKILLS and name in LANG_INSTRUCTION and name in MAX_STEPS


@pytest.mark.skipif(not (ROOT / "outputs" / "act_open_drawer" / "checkpoints" / "last" / "pretrained_model" / "config.json").exists(),
                    reason="ACT checkpoint not trained yet")
def test_act_checkpoint_loads_and_acts():
    import gymnasium as gym
    import numpy as np
    import souschef_env  # noqa: F401
    from runtime.executors import LoadedPolicy
    env = gym.make("souschef_env/Thali-v0", disable_env_checker=True).unwrapped
    obs, _ = env.reset(seed=0)
    pol = LoadedPolicy(ROOT / "outputs" / "act_open_drawer" / "checkpoints" / "last" / "pretrained_model", device="cpu")
    a = pol.act(obs, LANG_INSTRUCTION["open_drawer"])
    assert a.shape == (12,) and np.all(np.isfinite(a))
    assert 0.0 <= a[5] <= 1.05 and 0.0 <= a[11] <= 1.05  # jaw commands stay in the normalised range
    env.close()
