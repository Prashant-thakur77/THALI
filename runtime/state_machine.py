"""Thali runtime (plan Phase 7): voice -> planner -> verifier -> per-arm queues -> skill -> camera check + oracle
-> next / replan, with barge-in (stop / other arm / resume) and a hash-chained audit log.

States: IDLE -> PLANNING -> VERIFYING -> EXECUTING -> CHECKING -> (EXECUTING | REPLANNING | DONE | BLOCKED),
plus PAUSED from EXECUTING on a barge-in.  The executor is pluggable: the scripted expert (this phase) or a
learned policy with retry/fallback (Phase 3, ``runtime/executors.py``).  Every transition, plan, verdict and
skill result is appended to the audit log; the latency chain speech-end -> plan-ready -> arm-moves is timed
per command and written to the run's results file.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from souschef_env import constants as C
from souschef_env import oracles
from souschef_env.env import ThaliEnv
from souschef_env.scene_description import describe, scene_state
from expert.primitives import Expert, Interrupted, SkillResult
from planner.plan import Planner
from planner.state_check import make_state_check
from runtime.arm_queues import ArmQueues, QueuedStep
from verifier.audit import AuditLog
from verifier.rules import ALLOW, BLOCK, REORDER, Verifier, World
from voice.bargein import BargeIn

ROOT = Path(__file__).resolve().parent.parent
SUBGOAL_OF = {"open_drawer": "drawer_open", "pour": "poured", "place_mug": "mug_placed", "close_drawer": "drawer_closed"}
ZONE_SUBGOAL = {"plate": "plate_placed", "fork": "fork_placed", "spoon": "spoon_placed", "mug": "mug_placed"}
MAX_REPLANS = 2
MAX_FINAL_CHECKS = 1  # after the queue drains, re-verify every goal and redo what moved (once)


@dataclass
class StepRecord:
    idx: int
    step: dict
    arm: str
    result: dict
    oracle_ok: bool
    camera_ok: bool | None
    camera_backend: str | None
    t_start: float
    t_end: float
    executor: str
    interrupted: bool = False
    replan_round: int = 0
    concurrent: bool = False           # executed simultaneously with the other arm's step
    table_ok: bool | None = None       # PatchCore table-state check after the step (None = check not enabled)
    table_score: float | None = None


@dataclass
class RunLog:
    seed: int
    split: str
    command: str
    plan_source: str = ""
    verdicts: list[str] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)
    replans: int = 0
    final_checks: list[dict] = field(default_factory=list)
    barge_ins: list[dict] = field(default_factory=list)
    latency: dict = field(default_factory=dict)
    subgoals: dict = field(default_factory=dict)
    success: bool = False
    state: str = "IDLE"
    ignored_speakers: list[dict] = field(default_factory=list)
    sim_steps: int = 0
    wall_s: float = 0.0

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["steps"] = [dict(s.__dict__) for s in self.steps]
        return d


class ExpertExecutor:
    """Runs a plan step with the scripted expert. Phase 3 adds PolicyExecutor with the same ``run`` signature."""
    name = "expert"

    def __init__(self, ex: Expert):
        self.ex = ex

    def run(self, step: dict) -> SkillResult:
        from expert.task import Step
        return Step(step["skill"], step["arm"], step.get("obj"), step.get("zone"), step.get("to_arm"), amount=step.get("amount")).run(self.ex)


class Runtime:
    def __init__(self, env: ThaliEnv, planner: Planner | None = None, verifier: Verifier | None = None,
                 executor_factory: Callable[[Expert], Any] | None = None, state_check=None,
                 audit_path: Path | None = None, say: Callable[[str], None] | None = None, camera_check: bool = True,
                 anomaly_check: Any | None = None, concurrent: bool = False):
        self.env = env
        self.planner = planner or Planner(backend="auto")
        self.verifier = verifier or Verifier()
        self.executor_factory = executor_factory or ExpertExecutor
        self.state_check = state_check or make_state_check(getattr(self.planner, "vlm", None))
        self.audit = AuditLog(audit_path or ROOT / "results" / "audit.jsonl")
        self.say = say or (lambda text: None)
        self.camera_check = camera_check
        self.anomaly_check = anomaly_check  # optional anomaly/check.TableAnomalyCheck (PatchCore IR on the overhead camera)
        self.concurrent = concurrent        # both arms at once for independent steps (expert executor; see ArmQueues.ready_pair)
        self.barge = BargeIn()
        self.state = "IDLE"
        self.ex: Expert | None = None
        self.queues = ArmQueues()
        self.on_step: Callable[[], None] | None = None  # e.g. a video recorder; called every control step
        self.after_check: Callable[[int, dict], None] | None = None  # eval hook: called after each step's check (perturbations)
        self.on_event: Callable[[str, dict, Any], None] | None = None  # UI hook: every audit record (kind, payload, record)
        self.last_steps: list[dict] = []   # executed steps of the previous command, for follow-ups ("again", "a bit more", "other side")

    # ------------------------------------------------------------ helpers
    def _log(self, kind: str, payload: dict) -> None:
        rec = self.audit.append(kind, payload)
        if self.on_event is not None:
            try:
                self.on_event(kind, payload, rec)
            except Exception:  # a UI listener must never break the control loop
                pass

    def _transition(self, new: str, log: RunLog, **why) -> None:
        self._log("state", {"from": self.state, "to": new, **why})
        self.state = new
        log.state = new

    def _frame(self) -> np.ndarray:
        return self.env.render_camera("overhead")

    def _subgoal_for(self, step: dict) -> str | None:
        if step["skill"] in SUBGOAL_OF:
            return SUBGOAL_OF[step["skill"]]
        if step["skill"] in ("pick_place", "handoff") and step.get("zone"):
            return ZONE_SUBGOAL.get(step["zone"])
        if step["skill"] == "put_in_drawer" and step.get("obj"):
            return {"fork": "fork_stowed", "spoon": "spoon_stowed"}.get(step["obj"].split("_")[0])
        return None

    # ------------------------------------------------------------ main loop
    def run_command(self, command: str, seed: int, split: str = "test", t_speech_end: float | None = None,
                    max_sim_steps: int = 20000, reset: bool = True) -> RunLog:
        """``reset=False`` keeps the current scene (a follow-up command on a table that is already set)."""
        t_wall0 = time.time()
        log = RunLog(seed=seed, split=split, command=command)
        if reset:
            self.env.reset(seed=seed, options={"split": split})
        if hasattr(self.state_check, "set_reference"):
            self.state_check.set_reference(self._frame())
        if self.anomaly_check is not None and hasattr(self.anomaly_check, "set_reference"):
            self.anomaly_check.set_reference(self._frame())
        self.ex = Expert(self.env, on_step=(lambda a, o: self.on_step()) if self.on_step else None)
        self.ex.interrupt = self.barge.stop_requested
        executor = self.executor_factory(self.ex)
        self._log("command", {"text": command, "seed": seed, "split": split})

        # ---- plan + verify
        self._transition("PLANNING", log)
        st, txt = scene_state(self.env.model, self.env.data), describe(self.env.model, self.env.data)
        world = World.from_scene_state(st)
        t_plan0 = time.time()
        from runtime.followups import detect as detect_followup, resolve as resolve_followup
        fk = detect_followup(command) if (not reset and self.last_steps) else None
        fplan = resolve_followup(fk, self.last_steps, st) if fk else None
        if fplan is not None:
            from planner.plan import PlanResult
            v = self.verifier.verify(fplan, World.from_scene_state(st))
            pr = PlanResult(plan=v.plan if v.verdict != BLOCK else fplan, source=f"followup:{fk}", verdict=v.verdict, attempts=[], latency_s=0.0)
            self._log("followup", {"kind": fk, "resolved_from": self.last_steps[-1], "steps": fplan["steps"]})
        else:
            pr = self.planner.plan(command, st, txt, self._frame(), verifier=self.verifier, world=world)
        t_plan_ready = time.time()
        log.plan_source = pr.source
        log.verdicts.append(pr.verdict or "n/a")
        self._log("plan", {"source": pr.source, "verdict": pr.verdict, "steps": pr.plan["steps"], "mode": pr.plan.get("mode"),
                           "attempts": [{k: v for k, v in a.items() if k != "raw"} for a in pr.attempts]})
        self._transition("VERIFYING", log)
        if pr.verdict == BLOCK or pr.verdict is None:
            v = self.verifier.verify(pr.plan, world)
            if v.verdict == BLOCK:
                self._transition("BLOCKED", log, reasons=[i.message for i in v.issues])
                self.say(f"I can't do that: {v.issues[0].message}")
                log.subgoals = oracles.subgoals(self.env.model, self.env.data)
                log.wall_s = time.time() - t_wall0
                self._log("result", log.as_dict() | {"steps": len(log.steps)})
                return log
            pr.plan = v.plan
        plan = pr.plan
        mode = plan.get("mode", "normal")
        if mode == "gentle":
            self.say("Gentle mode on. I will move slowly.")
        self.say("Okay. " + ", then ".join(f"{s['skill'].replace('_', ' ')} with arm {s['arm'].upper()}" for s in plan["steps"]))
        self.queues.load(plan)

        # ---- execute
        replan_round = 0
        final_checks = 0
        t_first_move: float | None = None
        while True:
            pair = None
            if self.concurrent and isinstance(executor, ExpertExecutor):
                st_now = scene_state(self.env.model, self.env.data)
                pair = self.queues.ready_pair({n: (o["x_cm"] / 100.0, o["y_cm"] / 100.0) for n, o in st_now["objects"].items()})
            batch = list(pair) if pair else ([q] if (q := self.queues.next_ready()) is not None else [])
            if not batch:
                if self.queues.pending():
                    self._transition("BLOCKED", log, reasons=["queue deadlock"])
                    break
                # ---- final-state verification: the world may have changed since a step was checked (knocked item,
                # a later skill disturbing an earlier placement).  Redo exactly the steps whose goal no longer holds.
                sg_end = oracles.subgoals(self.env.model, self.env.data)
                missing = [g for g in self._goals_for(plan) if not sg_end[g]]
                if missing and final_checks < MAX_FINAL_CHECKS:
                    final_checks += 1
                    log.replans += 1
                    redo = [s for s in plan["steps"] if self._subgoal_for(s) in missing]
                    self._transition("REPLANNING", log, final_check=missing)
                    log.final_checks.append({"missing": missing, "redo": redo})
                    self._log("final_check", {"missing": missing, "redo": redo})
                    self.say("Something moved. Redoing " + ", ".join(g.replace("_", " ") for g in missing) + ".")
                    self.queues.load({"steps": redo, "mode": mode})
                    continue
                break
            concurrent = len(batch) > 1
            self._transition("EXECUTING", log, steps=[b.idx for b in batch], skills=[b.step["skill"] for b in batch],
                             arms=[b.arm for b in batch], concurrent=concurrent)
            for b in batch:
                self.queues.mark(b, "running")
                if b.step["skill"] == "pour":
                    self.say("Pouring now. Say stop anytime.")
            if concurrent:
                self.say("Both arms at once: " + " and ".join(f"{b.step['skill'].replace('_', ' ')} with arm {b.arm.upper()}" for b in batch))
            t0 = time.time()
            if t_first_move is None:
                t_first_move = t0
            interrupted = False
            results: dict[int, SkillResult] = {}
            while True:
                try:
                    if concurrent:
                        results = self._run_concurrent(executor, batch)
                    else:
                        results = {batch[0].idx: executor.run(batch[0].step)}
                    break
                except Interrupted:
                    interrupted = True
                    kind = self.barge.take()
                    self._transition("PAUSED", log, barge_in=kind, steps=[b.idx for b in batch])
                    log.barge_ins.append({"t": time.time(), "kind": kind, "step": batch[0].idx, "skill": batch[0].step["skill"]})
                    self.say("Stopped." if kind == "stop" else "Switching to the other arm.")
                    resume = self._wait_for_resume(kind)
                    if resume == "other_arm" and not concurrent:
                        batch[0].step = {**batch[0].step, "arm": "b" if batch[0].step["arm"] == "a" else "a"}
                        self.say("Continuing with the other arm.")
                    else:
                        self.say("Continuing.")
                    self._transition("EXECUTING", log, steps=[b.idx for b in batch], resumed=True)
                    continue
            t1 = time.time()
            render_was = self.env.render_enabled
            self.env.render_enabled = True
            # ---- checks: sim oracle (ground truth) and camera-based judgement, per step
            self._transition("CHECKING", log, steps=[b.idx for b in batch])
            failed: QueuedStep | None = None
            for b in batch:
                result = results[b.idx]
                sg = oracles.subgoals(self.env.model, self.env.data)
                key = self._subgoal_for(b.step)
                oracle_ok = bool(result.ok) if key is None else bool(sg[key]) or bool(result.ok)
                cam_ok, cam_backend = None, None
                if self.camera_check and key is not None:
                    try:
                        cam_ok = bool(self.state_check.ask(self._frame(), key))
                        cam_backend = self.state_check.name
                    except Exception as e:  # pragma: no cover
                        cam_backend = f"error:{type(e).__name__}"
                table_ok, table_score = None, None
                if self.anomaly_check is not None:
                    try:
                        table_score = round(float(self.anomaly_check.score(self._frame())), 4)
                        table_ok = bool(table_score < self.anomaly_check.threshold)
                        if not table_ok:
                            self.say("The table looks disturbed. I will check everything at the end.")
                    except Exception:  # pragma: no cover
                        table_ok = None
                rec = StepRecord(b.idx, b.step, b.step["arm"], {"ok": bool(result.ok), **{k: _jsonable(v) for k, v in result.detail.items()}},
                                 oracle_ok, cam_ok, cam_backend, t0, t1, getattr(executor, "name", "expert"), interrupted, replan_round)
                rec.table_ok, rec.table_score, rec.concurrent = table_ok, table_score, concurrent
                log.steps.append(rec)
                self._log("skill", {"step": b.idx, "skill": b.step["skill"], "arm": b.step["arm"], "ok": bool(result.ok), "oracle": oracle_ok,
                                    "camera": cam_ok, "camera_backend": cam_backend, "table_ok": table_ok, "table_score": table_score,
                                    "concurrent": concurrent, "seconds": round(t1 - t0, 2), "detail": rec.result})
                if self.after_check:
                    self.after_check(b.idx, b.step)
                if oracle_ok:
                    self.queues.mark(b, "done")
                else:
                    self.queues.mark(b, "failed")
                    failed = failed or b
            self.env.render_enabled = render_was   # per-step 3-camera rendering is only for policies; the expert runs blind
            if failed is not None:
                result = results[failed.idx]
                if replan_round >= MAX_REPLANS:
                    self._transition("BLOCKED", log, reasons=[f"step {failed.idx} failed after {MAX_REPLANS} replans"])
                    break
                replan_round += 1
                log.replans += 1
                self._transition("REPLANNING", log, failed=failed.idx, reason=str(result.detail.get("stage", result.detail.get("reason", "oracle false"))))
                st, txt = scene_state(self.env.model, self.env.data), describe(self.env.model, self.env.data)
                remaining = {"steps": [failed.step] + [p.step for p in self.queues.pending()], "mode": mode}
                rp = self.planner.replan(command, st, txt, remaining, 0, str(result.detail), self._frame())
                v = self.verifier.verify(rp.plan, World.from_scene_state(st))
                log.verdicts.append(v.verdict)
                self._log("replan", {"source": rp.source, "verdict": v.verdict, "steps": rp.plan["steps"], "issues": [i.message for i in v.issues]})
                if v.verdict == BLOCK:
                    self._transition("BLOCKED", log, reasons=[i.message for i in v.issues])
                    break
                self.queues.load(v.plan)
            if self.ex.steps > max_sim_steps:
                self._transition("BLOCKED", log, reasons=["sim step budget exhausted"])
                break
        # ---- done
        log.subgoals = oracles.subgoals(self.env.model, self.env.data)
        log.success = all(log.subgoals[g] for g in self._goals_for(plan))
        if self.state != "BLOCKED":
            self._transition("DONE", log, success=log.success)
            self.say("The table is set. Enjoy your meal." if log.success else "I finished but something is not in place.")
        log.latency = {
            "speech_end_to_plan_ready_s": round(t_plan_ready - t_speech_end, 3) if t_speech_end else None,
            "plan_s": round(t_plan_ready - t_plan0, 3),
            "plan_ready_to_arm_moves_s": round(t_first_move - t_plan_ready, 3) if t_first_move else None,
            "speech_end_to_arm_moves_s": round(t_first_move - t_speech_end, 3) if (t_speech_end and t_first_move) else None,
        }
        log.ignored_speakers = list(self.barge.events)
        self.last_steps = [dict(s.step) for s in log.steps if s.oracle_ok]
        log.sim_steps = self.ex.steps
        log.wall_s = round(time.time() - t_wall0, 1)
        self._log("result", {"success": log.success, "subgoals": log.subgoals, "latency": log.latency, "replans": log.replans, "sim_steps": log.sim_steps})
        return log

    def _run_concurrent(self, executor: Any, batch: list[QueuedStep]) -> dict[int, SkillResult]:
        """Drive both arms' steps in two threads through the expert's step barrier; re-raise Interrupted once."""
        import threading
        results: dict[int, SkillResult] = {}
        errors: list[BaseException] = []

        def worker(b: QueuedStep) -> None:
            self.ex.begin_concurrent(b.arm)
            try:
                results[b.idx] = executor.run(b.step)
            except BaseException as e:
                errors.append(e)
            finally:
                self.ex.end_concurrent(b.arm)

        threads = [threading.Thread(target=worker, args=(b,), daemon=True) for b in batch]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.ex.sync_from_env()
        if any(isinstance(e, Interrupted) for e in errors):
            raise Interrupted()
        if errors:
            raise errors[0]
        for b in batch:  # a thread that raised something else than Interrupted leaves no result: count it as failed
            results.setdefault(b.idx, SkillResult(b.step["skill"], False, 0, {"reason": "thread error"}))
        return results

    def _goals_for(self, plan: dict) -> list[str]:
        goals = []
        for s in plan["steps"]:
            k = self._subgoal_for(s)
            if k and k not in goals:
                goals.append(k)
        return goals or list(oracles.FULL_TASK)

    def _wait_for_resume(self, kind: str | None, timeout_s: float = 30.0) -> str:
        """Hold position until the operator says continue / other arm (or the timeout, which resumes)."""
        assert self.ex is not None
        if kind == "other_arm":
            return "other_arm"
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            k = self.barge.take()
            if k in ("resume", "other_arm"):
                return k
            self.ex.hold_still(5)
            if self.resume_hook is not None and self.resume_hook():
                return "resume"
        return "resume"

    resume_hook: Callable[[], bool] | None = None


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v
