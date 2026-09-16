"""Scripted expert: individual skills succeed on seed 0 and the oracles agree; instructions cover every skill."""

import random

import gymnasium as gym
import pytest

import souschef_env  # noqa: F401
from souschef_env import constants as C
from souschef_env import oracles
from expert.instructions import PARAPHRASES, canonical, instruction
from expert.primitives import Expert, Workspace


@pytest.fixture(scope="module")
def env():
    e = gym.make("souschef_env/Thali-v0", disable_env_checker=True, obs_type="state").unwrapped
    yield e
    e.close()


def test_open_drawer_seed0(env):
    env.reset(seed=0)
    ex = Expert(env)
    r = ex.open_drawer("a")
    assert r.ok and oracles.drawer_open(env.model, env.data)


def test_pick_place_mug_seed0(env):
    env.reset(seed=0)
    ex = Expert(env)
    r = ex.pick_place("mug", "b", "mug")
    assert r.ok and oracles.object_in_zone(env.model, env.data, "mug", "mug")
    assert r.detail["place_err"] < 0.03


def test_pick_place_plate_seed0(env):
    env.reset(seed=0)
    ex = Expert(env)
    assert ex.pick_place("plate", "a", "plate").ok


def test_handoff_spoon_via_table_seed0(env):
    env.reset(seed=0)
    ex = Expert(env)
    assert ex.open_drawer("a").ok
    r = ex.handoff_place("spoon_1", "a", "b", "spoon")
    assert r.ok and oracles.object_in_zone(env.model, env.data, "spoon_1", "spoon")


def test_hold_mug_and_pour_seed0(env):
    """Standalone hold+pour: water reaches the mug and the mug never leaves B's grasp (the >= 6-sphere
    'poured' threshold is asserted on the full task below, where the pour runs in its real context)."""
    env.reset(seed=0)
    ex = Expert(env)
    assert ex.hold_mug("b").ok
    assert oracles.mug_held(env.model, env.data) == "b"
    r = ex.pour("a")
    assert r.detail["spheres_after"] >= 3
    assert oracles.mug_held(env.model, env.data) == "b"


def test_full_task_seed0_succeeds(env):
    from expert.task import run_full_task
    env.reset(seed=0)
    ex = Expert(env)
    res = run_full_task(ex)
    assert all(r.ok for r in res), [(r.skill, r.detail) for r in res if not r.ok]
    assert oracles.task_success(env.model, env.data)


def test_pour_refused_when_same_arm_holds_mug(env):
    env.reset(seed=0)
    ex = Expert(env)
    assert ex.hold_mug("b").ok
    r = ex.pour("b")
    assert not r.ok and r.detail["reason"] == "pouring arm holds the mug"


def test_workspace_reservation():
    ws = Workspace()
    assert ws.reserve("a")
    assert not ws.reserve("b")
    ws.release("a")
    assert ws.reserve("b")
    assert ws.inside((0.02, 0.01)) and not ws.inside((0.2, 0.0))


def test_instructions_cover_every_skill_with_ten_paraphrases():
    for skill, lst in PARAPHRASES.items():
        assert len(lst) == 10, skill
    rng = random.Random(0)
    texts = {instruction("pick_place", rng, "a", "fork_1", "fork") for _ in range(50)}
    assert len(texts) >= 8
    assert canonical("pour", "a") == "pour water into the mug with arm A"
    assert "arm B" in canonical("handoff", "a", "spoon_1", "spoon", arm2="b")
