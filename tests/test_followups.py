"""Follow-ups resolved against the previous command's executed steps (no simulator)."""

from runtime.followups import detect, resolve

LAST = [{"skill": "open_drawer", "arm": "a"}, {"skill": "pick_place", "arm": "a", "obj": "fork_1", "zone": "fork"},
        {"skill": "hold_mug", "arm": "b"}, {"skill": "pour", "arm": "a", "amount": "normal"}, {"skill": "place_mug", "arm": "b"}]


def test_detect_kinds():
    assert detect("again") == "again" and detect("do that again") == "again"
    assert detect("a bit more please") == "more" and detect("more water") == "more"
    assert detect("no, the other side") == "other_side"
    assert detect("use the other arm") == "other_arm"
    assert detect("open the top drawer") is None


def test_more_reholds_when_the_mug_was_set_down():
    p = resolve("more", LAST, {"objects": {"mug": {"held_by": None}}})
    assert [s["skill"] for s in p["steps"]] == ["hold_mug", "pour", "place_mug"]
    assert p["steps"][1]["amount"] == "little" and p["steps"][1]["arm"] == "a" and p["steps"][0]["arm"] == "b"


def test_more_skips_hold_when_still_held():
    p = resolve("more", LAST[:4], {"objects": {"mug": {"held_by": "b"}}})
    assert [s["skill"] for s in p["steps"]] == ["pour", "place_mug"]


def test_other_side_moves_the_fork_across_with_a_handoff():
    p = resolve("other_side", LAST)
    assert p["steps"] == [{"skill": "handoff", "arm": "a", "obj": "fork_1", "to_arm": "b", "zone": "spoon"}]


def test_again_and_other_arm():
    assert resolve("again", LAST)["steps"] == [LAST[-1]]
    assert resolve("other_arm", LAST)["steps"][0]["arm"] == "a"
    assert resolve("again", []) is None
