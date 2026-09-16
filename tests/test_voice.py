"""Parser (offline) and the recorded voice-test results; the live Speechmatics calls are exercised by eval/voice_test.py."""

import json
from pathlib import Path

from voice.bargein import BargeIn
from voice.parser import detect_barge_in, intents_to_command, normalize, parse, split_clauses
from planner.plan import rule_plan

ROOT = Path(__file__).resolve().parent.parent
FULL = "Open the top drawer, pick up the plate with arm A, place it on the table, pick up the mug with arm B, pour water into the mug with arm A"


def test_asr_noise_and_homophones():
    r = parse("um okay so open the drawer pick up the plate with arm eight and place it down then pour a little water with arm bee")
    kinds = [(i.kind, i.arm) for i in r.intents]
    assert kinds == [("open_drawer", None), ("pick_place", "a"), ("pour", "b")]
    assert r.intents[2].amount == "little" and not r.diagnostics


def test_full_command_round_trips_to_a_plan():
    r = parse(FULL)
    assert [i.kind for i in r.intents] == ["open_drawer", "pick_place", "pick_place", "pour"]
    plan = rule_plan(intents_to_command(r.intents))
    skills = [s["skill"] for s in plan["steps"]]
    assert skills[:2] == ["open_drawer", "pick_place"] and "pour" in skills and "hold_mug" in skills


def test_hinglish_and_devanagari():
    for t in ("top drawer kholo, plate ko arm A se uthao, aur table pe rakh do", "टॉप ड्रॉअर खोलो, प्लेट को आर्म ए से उठाओ, और टेबल पे रख दो"):
        r = parse(t)
        assert [(i.kind, i.obj, i.arm) for i in r.intents] == [("open_drawer", None, None), ("pick_place", "plate", "a")], (t, r)
        assert not r.diagnostics
    assert parse("रुको").barge_in == "stop"
    assert parse("paani daalo dheere").intents[0].kind == "pour" and parse("paani daalo dheere").intents[0].gentle


def test_barge_in_words():
    assert detect_barge_in("stop") == "stop" and detect_barge_in("no wait") == "stop"
    assert detect_barge_in("use the other arm") == "other_arm"
    assert detect_barge_in("okay continue") == "resume"
    assert detect_barge_in("open the top drawer") is None
    assert parse("stop").intents[0].kind == "stop"


def test_barge_in_speaker_focus():
    b = BargeIn(operator="S1")
    b.on_partial("stop", "S2")
    assert b.pending is None and b.events[-1]["ignored"]
    b.on_partial("stop", "S1")
    assert b.stop_requested() and b.take() == "stop" and b.pending is None


def test_normalize_and_split():
    assert "open the drawer" in normalize("Please, robot: open the drawer!")
    assert split_clauses("open the drawer pick up the plate place it on the table") == ["open the drawer", "pick up the plate", "place it on the table"]
    r = parse("open the drawer pick up the plate place it on the table")
    assert [i.kind for i in r.intents] == ["open_drawer", "pick_place"] and not r.diagnostics


def test_unknown_clause_is_reported_not_dropped():
    r = parse("open the drawer then sing a song")
    assert [i.kind for i in r.intents] == ["open_drawer"]
    assert any("sing a song" in d for d in r.diagnostics)


def test_voice_results_file():
    d = json.loads((ROOT / "results" / "voice_test.json").read_text())
    assert d["samples"] == 4 and d["skill_match_rate"] == 1.0
    files = {r["file"]: r for r in d["rows"]}
    assert files["normal.wav"]["wer"] == 0.0
    assert files["hindi.wav"]["language"] == "hi" and files["hindi.wav"]["normalized_wer"] < files["hindi.wav"]["wer"]
    assert files["noisy.wav"]["ignored_other_speaker"], "speaker focus should have ignored the background speaker"
    for r in d["rows"]:
        assert all(0 < x < 5 for x in r["partial_to_ready_s"])
