"""Step barrier and the safe-pair rule for simultaneous two-arm execution."""

import threading

from expert.primitives import StepBarrier
from runtime.arm_queues import ArmQueues


def test_barrier_steps_once_per_generation_with_two_arms():
    bar = StepBarrier()
    count = {"n": 0}

    def do_step():
        count["n"] += 1

    bar.register("a")
    bar.register("b")
    N = 25

    def worker(arm):
        for _ in range(N):
            bar.step(arm, do_step)
        bar.unregister(arm, do_step)

    ts = [threading.Thread(target=worker, args=(a,)) for a in ("a", "b")]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in ts)
    assert count["n"] == N  # both arms submitted N times -> the physics advanced N times, not 2N


def test_barrier_is_noop_for_a_single_arm():
    bar = StepBarrier()
    n = {"n": 0}
    bar.register("a")
    for _ in range(5):
        bar.step("a", lambda: n.__setitem__("n", n["n"] + 1))
    assert n["n"] == 5


def test_barrier_releases_waiter_when_other_arm_finishes():
    bar = StepBarrier()
    n = {"n": 0}
    bar.register("a")
    bar.register("b")
    done = threading.Event()

    def a_worker():
        for _ in range(3):
            bar.step("a", lambda: n.__setitem__("n", n["n"] + 1))
        done.set()

    t = threading.Thread(target=a_worker)
    t.start()
    bar.step("b", lambda: n.__setitem__("n", n["n"] + 1))   # generation 1
    bar.unregister("b", lambda: n.__setitem__("n", n["n"] + 1))  # arm a was waiting for us: advance and release
    assert done.wait(10)
    t.join(5)


def test_ready_pair_requires_independent_far_apart_single_arm_skills():
    qs = ArmQueues()
    qs.load({"steps": [{"skill": "open_drawer", "arm": "a"}, {"skill": "pick_place", "arm": "b", "obj": "mug", "zone": "mug"}]})
    pair = qs.ready_pair({"mug": (0.10, 0.04)})
    assert pair and {p.step["skill"] for p in pair} == {"open_drawer", "pick_place"}
    # handoff needs the shared zone: never paired
    qs.load({"steps": [{"skill": "open_drawer", "arm": "a"}, {"skill": "hold_mug", "arm": "b"}]})
    assert qs.ready_pair({"mug": (0.10, 0.04)}) is None
    # objects too close together: never paired
    qs.load({"steps": [{"skill": "pick_place", "arm": "a", "obj": "plate", "zone": "plate"}, {"skill": "pick_place", "arm": "b", "obj": "mug", "zone": "mug"}]})
    assert qs.ready_pair({"plate": (0.0, -0.1), "mug": (0.02, -0.08)}) is None
