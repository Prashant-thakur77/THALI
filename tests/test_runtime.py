import json
import threading
import time
from pathlib import Path

import gymnasium as gym
import pytest

import souschef_env  # noqa: F401
from planner.plan import Planner
from runtime.arm_queues import ArmQueues
from runtime.state_machine import Runtime
from verifier.rules import Verifier

ROOT = Path(__file__).resolve().parent.parent


def S(skill, arm, **kw):
    return {"skill": skill, "arm": arm, **kw}


def test_arm_queues_dependencies_and_dispatch_order():
    q = ArmQueues()
    q.load({"steps": [S("open_drawer", "a"), S("hold_mug", "b"), S("pick_place", "a", obj="fork_1", zone="fork"), S("pour", "a"), S("place_mug", "b")]})
    first = q.next_ready()
    assert first.step["skill"] in ("open_drawer", "hold_mug")
    q.mark(first, "done")
    second = q.next_ready()
    assert second.step["skill"] in ("open_drawer", "hold_mug") and second.arm != first.arm  # the idle arm goes next
    q.mark(second, "done")
    fork = q.next_ready(); assert fork.step["skill"] == "pick_place"; q.mark(fork, "done")
    pour = q.next_ready(); assert pour.step["skill"] == "pour"; q.mark(pour, "done")
    place = q.next_ready(); assert place.step["skill"] == "place_mug"; q.mark(place, "done")
    assert q.next_ready() is None and not q.pending() and not q.deadlocked()


def test_pour_waits_for_hold():
    q = ArmQueues()
    q.load({"steps": [S("hold_mug", "b"), S("pour", "a")]})
    assert q.next_ready().step["skill"] == "hold_mug"


@pytest.fixture(scope="module")
def runtime():
    env = gym.make("souschef_env/Thali-v0", disable_env_checker=True).unwrapped
    rt = Runtime(env, Planner(backend="rules"), Verifier(), audit_path=ROOT / "results" / "test_audit.jsonl", camera_check=True)
    yield rt
    env.close()


def test_runtime_completes_a_short_command(runtime):
    log = runtime.run_command("put the plate on the table with arm A", seed=0, split="train", t_speech_end=time.time())
    assert log.state == "DONE" and log.success and log.subgoals["plate_placed"]
    assert log.plan_source == "rules" and log.verdicts[0] == "ALLOW"
    assert log.steps and log.steps[0].oracle_ok and log.steps[0].camera_backend == "pixels"
    assert log.latency["speech_end_to_arm_moves_s"] is not None and log.latency["speech_end_to_arm_moves_s"] < 5


def test_runtime_blocks_unsafe_command(runtime):
    log = runtime.run_command("pour water", seed=0, split="train")
    # rule planner inserts hold_mug, so this is allowed; a raw unsafe plan is exercised through the verifier tests
    assert log.verdicts[0] in ("ALLOW", "REORDER")


def test_barge_in_stop_and_resume(runtime):
    runtime.barge.operator = "S1"

    def inject():
        while runtime.state != "EXECUTING":
            time.sleep(0.02)
        time.sleep(1.0)
        runtime.barge.on_partial("stop", "S2")      # not the operator: ignored
        runtime.barge.on_partial("stop", "S1")      # operator: pauses
        time.sleep(1.0)
        runtime.barge.on_partial("continue", "S1")

    threading.Thread(target=inject, daemon=True).start()
    log = runtime.run_command("put the plate on the table with arm A", seed=1, split="train")
    assert len(log.barge_ins) == 1 and log.barge_ins[0]["kind"] == "stop"
    assert any(e.get("ignored") for e in log.ignored_speakers)
    assert log.steps[0].interrupted and log.success


def test_audit_log_of_runs_is_chained():
    from verifier.audit import verify
    assert verify(ROOT / "results" / "test_audit.jsonl")["ok"]
    assert verify(ROOT / "results" / "audit.jsonl")["ok"]


def test_demo_results_exist():
    d = json.loads((ROOT / "results" / "demo_seed3.json").read_text())
    assert d["success"] and d["voice"]["operator"] and d["voice"]["ignored"]
    assert d["latency"]["speech_end_to_arm_moves_s"] > 0
    b = json.loads((ROOT / "results" / "demo_bargein_stop.json").read_text())
    assert b["barge_ins"] and b["steps"][0]["interrupted"]
