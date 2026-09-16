import json
from pathlib import Path

import numpy as np
import pytest

from souschef_env import constants as C
from verifier.audit import GENESIS, AuditLog, verify
from verifier.rules import ALLOW, BLOCK, REORDER, Verifier, World, validate_schema

ROOT = Path(__file__).resolve().parent.parent


def _world() -> World:
    objs = {"plate": (-0.12, -0.24, 0.0), "mug": (0.15, -0.22, 0.0), "bottle": (0.04, 0.09, 0.0),
            "fork_1": (-0.235, 0.20, 0.008), "fork_2": (-0.185, 0.20, 0.008), "spoon_1": (-0.135, 0.20, 0.008), "spoon_2": (-0.085, 0.20, 0.008)}
    return World(objects=objs, held={k: None for k in objs}, drawer_open=False, in_drawer={k: k in C.CUTLERY for k in objs})


@pytest.fixture(scope="module")
def ver():
    return Verifier()


def S(skill, arm, **kw):
    return {"skill": skill, "arm": arm, **kw}


def test_full_task_allowed(ver):
    plan = {"steps": [S("open_drawer", "a"), S("pick_place", "a", obj="fork_1", zone="fork"),
                      S("handoff", "a", obj="spoon_1", to_arm="b", zone="spoon"), S("pick_place", "a", obj="plate", zone="plate"),
                      S("hold_mug", "b"), S("pour", "a"), S("place_mug", "b")]}
    v = ver.verify(plan, _world())
    assert v.verdict == ALLOW and not v.issues


def test_pour_without_hold_blocked(ver):
    v = ver.verify({"steps": [S("pour", "a")]}, _world())
    assert v.verdict == BLOCK and {i.code for i in v.issues} == {"POUR_PRECOND"}


def test_pour_before_hold_is_reordered(ver):
    v = ver.verify({"steps": [S("pour", "a"), S("hold_mug", "b")]}, _world())
    assert v.verdict == REORDER
    assert [s["skill"] for s in v.plan["steps"]] == ["hold_mug", "pour"]


def test_cutlery_before_drawer_is_reordered(ver):
    v = ver.verify({"steps": [S("pick_place", "a", obj="fork_1", zone="fork"), S("open_drawer", "a")]}, _world())
    assert v.verdict == REORDER and v.plan["steps"][0]["skill"] == "open_drawer"


def test_reach_blocks_cross_table_pick(ver):
    v = ver.verify({"steps": [S("pick_place", "a", obj="mug", zone="mug")]}, _world())
    assert v.verdict == BLOCK and any(i.code == "REACH" for i in v.issues)


def test_workspace_reservation(ver):
    v = ver.verify({"steps": [S("hold_mug", "b"), S("handoff", "a", obj="plate", to_arm="b", zone="plate")]}, _world())
    assert v.verdict == BLOCK and any(i.code in ("WORKSPACE", "GRASP_PRECOND") for i in v.issues)


def test_schema_rejects_garbage():
    assert validate_schema({"steps": [{"skill": "fly", "arm": "a"}]})
    assert validate_schema({"steps": []})
    assert validate_schema({"steps": [{"skill": "open_drawer", "arm": "a", "speed": 9}]})
    assert not validate_schema({"steps": [{"skill": "open_drawer", "arm": "a"}]})


def test_velocity_limit_and_gentle_mode():
    q0 = np.zeros(12)
    q1 = q0.copy()
    q1[1] = 0.05  # 0.05 rad in 0.02 s = 2.5 rad/s
    assert Verifier.check_velocity(q0, q1, C.DT) is None
    assert Verifier.check_velocity(q0, q1, C.DT, mode="gentle").code == "VELOCITY"
    q1[5] = 1.0  # jaw entries are ignored
    q1[1] = 0.0
    assert Verifier.check_velocity(q0, q1, C.DT, mode="gentle") is None


def test_audit_chain_detects_tampering(tmp_path):
    p = tmp_path / "audit.jsonl"
    log = AuditLog(p)
    r0 = log.append("verify", {"plan": "x", "verdict": "ALLOW"})
    assert r0["prev_hash"] == GENESIS
    log.append("skill", {"name": "pour", "ok": True})
    log.append("skill", {"name": "place_mug", "ok": True})
    assert verify(p)["ok"]
    lines = p.read_text().splitlines()
    tampered = lines[1].replace('"ok": true', '"ok": false')
    p.write_text("\n".join([lines[0], tampered, lines[2]]) + "\n")
    res = verify(p)
    assert not res["ok"] and res["first_bad_seq"] == 1 and "sha256" in res["reason"]
    p.write_text("\n".join([lines[0], lines[2]]) + "\n")
    assert not verify(p)["ok"]
    # appending to an existing log continues the chain
    p.write_text("\n".join(lines) + "\n")
    AuditLog(p).append("skill", {"name": "done"})
    assert verify(p) == {"ok": True, "records": 4, "first_bad_seq": None, "reason": "chain intact"}


def test_injection_results_are_20_of_20():
    d = json.loads((ROOT / "results" / "verifier_injection.json").read_text())
    assert d["bad_plans"] == 20 and d["caught"] == 20
    assert d["good_passed"] == d["good_plans"]
    assert d["audit_chain"]["ok"]
    assert verify(ROOT / "results" / "verifier_injection_audit.jsonl")["ok"]
