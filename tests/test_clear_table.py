"""Clear-the-table skills: parser, rule planner, verifier and queue ordering (no simulator)."""

from planner.plan import rule_plan
from runtime.arm_queues import ArmQueues
from verifier.rules import ALLOW, BLOCK, REORDER, Verifier, World
from voice.parser import intents_to_command, parse

SCENE = {"drawer": {"open": True, "travel_cm": 11.5},
         "objects": {"fork_1": {"x_cm": -13, "y_cm": -10, "z_cm": 1, "held_by": None, "in_drawer": False},
                     "spoon_1": {"x_cm": 13, "y_cm": -10, "z_cm": 1, "held_by": None, "in_drawer": False},
                     "fork_2": {"x_cm": -23, "y_cm": 19, "z_cm": 1, "held_by": None, "in_drawer": True},
                     "spoon_2": {"x_cm": -13, "y_cm": 19, "z_cm": 1, "held_by": None, "in_drawer": True},
                     "plate": {"x_cm": 0, "y_cm": -10, "z_cm": 1, "held_by": None, "in_drawer": False},
                     "mug": {"x_cm": 10, "y_cm": 4, "z_cm": 3, "held_by": None, "in_drawer": False},
                     "bottle": {"x_cm": 3, "y_cm": 11, "z_cm": 6, "held_by": None, "in_drawer": False}},
         "arms": {"a": {"jaw": "open"}, "b": {"jaw": "open"}}, "subgoals": {}}


def test_parser_clear_and_close():
    r = parse("please clear the table")
    assert [i.kind for i in r.intents] == ["clear_table"]
    assert intents_to_command(r.intents) == "clear the table"
    r = parse("close the drawer with arm A")
    assert r.intents[0].kind == "close_drawer" and r.intents[0].arm == "a"


def test_rule_plan_clear_orders_spoon_handoff_then_stow_then_close():
    plan = rule_plan("clear the table", SCENE)
    skills = [(s["skill"], s.get("obj")) for s in plan["steps"]]
    assert skills == [("put_in_drawer", "fork_1"), ("handoff", "spoon_1"), ("put_in_drawer", "spoon_1"), ("close_drawer", None)]
    assert plan["steps"][1]["arm"] == "b" and plan["steps"][1]["to_arm"] == "a" and plan["steps"][1]["zone"] is None


def test_rule_plan_close_only_when_drawer_open():
    assert [s["skill"] for s in rule_plan("close the drawer", SCENE)["steps"]] == ["close_drawer"]


def test_verifier_blocks_close_before_stow_and_stow_into_closed_drawer():
    v = Verifier()
    w = World.from_scene_state(SCENE)
    good = {"steps": [{"skill": "put_in_drawer", "arm": "a", "obj": "fork_1"}, {"skill": "close_drawer", "arm": "a"}], "mode": "normal"}
    assert v.verify(good, w).verdict == ALLOW
    wrong_order = {"steps": [{"skill": "close_drawer", "arm": "a"}, {"skill": "put_in_drawer", "arm": "a", "obj": "fork_1"}], "mode": "normal"}
    assert v.verify(wrong_order, World.from_scene_state(SCENE)).verdict in (REORDER, BLOCK)
    closed = dict(SCENE, drawer={"open": False, "travel_cm": 0})
    stow_closed = {"steps": [{"skill": "put_in_drawer", "arm": "a", "obj": "fork_1"}], "mode": "normal"}
    assert v.verify(stow_closed, World.from_scene_state(closed)).verdict == BLOCK


def test_queue_runs_close_drawer_last():
    qs = ArmQueues()
    qs.load(rule_plan("clear the table", SCENE))
    first = qs.next_ready()
    assert first.step["skill"] != "close_drawer"
    close = next(q for q in qs.all if q.step["skill"] == "close_drawer")
    assert len(close.deps) >= 2
