import json
from pathlib import Path

import numpy as np

from planner.plan import PlanError, Planner, normalise, parse_plan, repair_json, rule_plan
from planner.state_check import PixelHeuristic, overhead_to_table
from verifier.rules import ALLOW, REORDER, Verifier, World, validate_schema

ROOT = Path(__file__).resolve().parent.parent
FULL = "Open the top drawer, pick up the plate with arm A, place it on the table, pick up the mug with arm B, pour water into the mug with arm A"


def test_schema_file_is_valid_json_with_all_skills():
    s = json.loads((ROOT / "planner" / "schema.json").read_text())
    assert set(s["properties"]["steps"]["items"]["properties"]["skill"]["enum"]) == {"open_drawer", "pick_place", "handoff", "hold_mug", "pour", "place_mug", "put_in_drawer", "close_drawer"}


def test_rule_plan_full_command_validates_and_is_allowed():
    p = rule_plan(FULL)
    assert not validate_schema(p)
    skills = [s["skill"] for s in p["steps"]]
    assert skills[:2] == ["open_drawer", "pick_place"] and "pour" in skills and skills.index("hold_mug") < skills.index("pour")
    v = Verifier(reach_check=False).verify(p, _world())
    assert v.verdict in (ALLOW, REORDER)


def test_rule_plan_gentle_and_little():
    p = rule_plan("pour a little water, gently")
    assert p["mode"] == "gentle" and [s for s in p["steps"] if s["skill"] == "pour"][0]["amount"] == "little"


def test_rule_plan_arm_words():
    p = rule_plan("pour water with arm B")
    pour = [s for s in p["steps"] if s["skill"] == "pour"][0]
    hold = [s for s in p["steps"] if s["skill"] == "hold_mug"][0]
    assert pour["arm"] == "b" and hold["arm"] == "a"


def test_repair_and_normalise_messy_model_output():
    raw = "Sure, here is the plan:\n```json\n{steps: [{skill: 'open_drawer', arm: 'A'}, {skill: 'pick', arm: 'left', object: 'fork', zone: 'left'}, {skill:'hand off', arm:'a', obj:'spoon', to:'B'},]"
    p = normalise(parse_plan(raw))
    assert not validate_schema(p), validate_schema(p)
    assert p["steps"][1] == {"skill": "pick_place", "arm": "a", "obj": "fork_1", "zone": "fork"}
    assert p["steps"][2]["skill"] == "handoff" and p["steps"][2]["to_arm"] == "b" and p["steps"][2]["zone"] == "spoon"


def test_parse_plan_rejects_prose():
    try:
        parse_plan("I cannot help with that.")
    except PlanError:
        return
    raise AssertionError("expected PlanError")


def test_planner_falls_back_to_rules_without_model(tmp_path):
    pl = Planner(backend="auto", model_dir=tmp_path)
    assert pl.backend == "rules"
    r = pl.plan("set the table", scene={}, scene_text="")
    assert r.source == "rules" and len(r.plan["steps"]) == 7


def test_overhead_projection_centre_and_scale():
    x, y = overhead_to_table(160, 120)
    assert abs(x) < 1e-6 and abs(y + 0.05) < 1e-6
    x2, _ = overhead_to_table(260, 120)
    assert 0.3 < x2 < 0.5  # 100 px ~ 0.39 m at the table with fovy 52 from 0.95 m


def test_pixel_heuristic_on_rendered_frame():
    import gymnasium as gym
    import souschef_env  # noqa: F401
    from souschef_env import oracles
    env = gym.make("souschef_env/Thali-v0", disable_env_checker=True).unwrapped
    obs, _ = env.reset(seed=0)
    h = PixelHeuristic()
    img = obs["pixels"]["overhead"]
    assert h.ask(img, "mug_placed") == oracles.subgoals(env.model, env.data)["mug_placed"]
    assert h.ask(img, "drawer_open") is False
    env.close()


def _world() -> World:
    from souschef_env import constants as C
    objs = {"plate": (-0.12, -0.24, 0.0), "mug": (0.15, -0.22, 0.0), "bottle": (0.04, 0.09, 0.0),
            "fork_1": (-0.235, 0.20, 0.008), "fork_2": (-0.185, 0.20, 0.008), "spoon_1": (-0.135, 0.20, 0.008), "spoon_2": (-0.085, 0.20, 0.008)}
    return World(objects=objs, held={k: None for k in objs}, drawer_open=False, in_drawer={k: k in C.CUTLERY for k in objs})
