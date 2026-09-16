"""Local VLM planner on OpenVINO (plan Phase 4.2/4.3).

Input: overhead frame + structured scene description + transcript.  Output: a plan that validates against
planner/schema.json, after ManipulaX-style JSON repair, one retry with the validation errors fed back, and a
deterministic rule-planner fallback.  Every call is timed; the last latency and tokens/s are exposed for the
bench.  ``replan`` re-enters with the failure reason (RePlanTable's pattern).

Backends: "vlm" (Qwen2-VL-2B INT4 via openvino_genai.VLMPipeline on GPU/CPU), "rules" (no model), "auto".
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from souschef_env import constants as C
from verifier.rules import validate_schema

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "planner" / "qwen2vl_int4"
PROMPTS = ROOT / "planner" / "prompts"
SYSTEM = (PROMPTS / "system.txt").read_text()
USER = (PROMPTS / "user.txt").read_text()
REPLAN = (PROMPTS / "replan.txt").read_text()


class PlanError(ValueError):
    pass


# ---------------------------------------------------------------------------
# JSON repair (the shape is right far more often than the punctuation)
# ---------------------------------------------------------------------------
def repair_json(text: str) -> str:
    js = text.strip()
    js = re.sub(r"```(?:json)?", "", js)
    js = js.replace("“", '"').replace("”", '"').replace("'", '"')
    js = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*):', r'\1"\2"\3:', js)
    js = re.sub(r",\s*([}\]])", r"\1", js)
    js += "]" * max(0, js.count("[") - js.count("]"))
    js += "}" * max(0, js.count("{") - js.count("}"))
    return js


def parse_plan(text: str) -> dict:
    """First JSON object in ``text`` -> plan dict (repairing), else PlanError."""
    m = re.search(r"\{.*", text, re.S)
    if not m:
        raise PlanError("no JSON object in model output")
    raw = m.group(0)
    last: Exception | None = None
    for cand in (raw, repair_json(raw)):
        try:
            out = json.loads(cand)
            if isinstance(out, dict) and "steps" in out:
                return out
            if isinstance(out, list):
                return {"steps": out}
        except json.JSONDecodeError as e:
            last = e
    steps = []
    for obj in re.finditer(r"\{[^{}]*\}", repair_json(raw)):
        try:
            d = json.loads(obj.group(0))
            if "skill" in d:
                steps.append(d)
        except json.JSONDecodeError:
            pass
    if steps:
        return {"steps": steps}
    raise PlanError(f"plan is not valid JSON: {last}")


def normalise(plan: dict) -> dict:
    """Coerce the small things a 2B model gets wrong: arm case, 'A'/'left', missing nulls, 'to' vs 'to_arm'."""
    arm_map = {"a": "a", "b": "b", "left": "a", "right": "b", "arm a": "a", "arm b": "b", "arm_a": "a", "arm_b": "b"}
    out = {"steps": [], "mode": plan.get("mode", "normal") if plan.get("mode") in ("normal", "gentle") else "normal"}
    for s in plan.get("steps", []):
        if not isinstance(s, dict):
            continue
        t = {}
        t["skill"] = str(s.get("skill", s.get("action", ""))).lower().replace("-", "_").replace(" ", "_")
        if t["skill"] in ("pick", "place", "pickplace", "pick_and_place", "move"):
            t["skill"] = "pick_place"
        if t["skill"] in ("hand_off", "hand_over", "transfer", "pass"):
            t["skill"] = "handoff"
        if t["skill"] in ("hold", "hold_the_mug", "grab_mug"):
            t["skill"] = "hold_mug"
        if t["skill"] in ("open_the_drawer", "drawer", "open"):
            t["skill"] = "open_drawer"
        t["arm"] = arm_map.get(str(s.get("arm", "a")).lower().strip(), "a")
        obj = s.get("obj", s.get("object"))
        if obj is not None:
            obj = str(obj).lower().strip().replace(" ", "_")
            obj = {"fork": "fork_1", "spoon": "spoon_1", "cup": "mug", "water": "bottle"}.get(obj, obj)
        t["obj"] = obj
        zone = s.get("zone", s.get("target"))
        if zone is not None:
            zone = str(zone).lower().strip()
            zone = {"plate_spot": "plate", "left": "fork", "right": "spoon", "mug_spot": "mug", "cup": "mug"}.get(zone, zone)
        t["zone"] = zone
        to = s.get("to_arm", s.get("to"))
        t["to_arm"] = arm_map.get(str(to).lower().strip()) if to is not None else None
        if t["skill"] == "pour":
            t["amount"] = s.get("amount") if s.get("amount") in ("little", "normal") else "normal"
        # fill the obvious defaults so the schema does not reject a terse but correct answer
        if t["skill"] == "hold_mug":
            t["obj"] = None
        if t["skill"] == "pick_place" and t["zone"] is None and t["obj"] is not None:
            t["zone"] = {"plate": "plate", "mug": "mug"}.get(t["obj"], "fork" if "fork" in t["obj"] else "spoon" if "spoon" in t["obj"] else None)
        if t["skill"] == "handoff" and t["zone"] is None and t["obj"] is not None:
            t["zone"] = {"plate": "plate", "mug": "mug"}.get(t["obj"], "fork" if "fork" in t["obj"] else "spoon")
        if t["skill"] == "handoff" and t["to_arm"] is None:
            t["to_arm"] = "b" if t["arm"] == "a" else "a"
        for k in ("obj", "zone", "to_arm"):
            if t.get(k) is None:
                t.pop(k, None)
        # degenerate repetition: drop a step identical to any earlier one (no skill is ever legitimately repeated
        # on the same object/arm), and stop at the schema's maximum
        if t in out["steps"] or len(out["steps"]) >= 12:
            continue
        out["steps"].append(t)
    return out


# ---------------------------------------------------------------------------
# Rule planner: the deterministic fallback, and the reference for the swap tests
# ---------------------------------------------------------------------------
_ARM_WORDS = {"arm a": "a", "left arm": "a", "arm b": "b", "right arm": "b", "arm ay": "a", "arm bee": "b"}


def _arm_in(text: str, default: str) -> str:
    for w, a in _ARM_WORDS.items():
        if w in text:
            return a
    return default


def rule_plan(command: str, scene: dict | None = None) -> dict:
    """Keyword planner: maps the command's nouns/verbs to skills with reach-aware arm assignment."""
    t = command.lower()
    gentle = "gentl" in t or "slow" in t or "careful" in t
    little = "little" in t or "bit" in t or "half" in t
    done = (scene or {}).get("subgoals", {})
    drawer_open = bool((scene or {}).get("drawer", {}).get("open", False))
    steps: list[dict] = []
    wants_all = any(w in t for w in ("set the table", "set table", "lay the table", "everything", "whole", "full", "dinner"))
    want_fork = wants_all or "fork" in t or "cutlery" in t
    want_spoon = wants_all or "spoon" in t or "cutlery" in t
    want_plate = wants_all or "plate" in t or "dish" in t
    want_mug = wants_all or ("mug" in t and "pour" not in t and "water" not in t) or "cup" in t
    want_pour = wants_all or "pour" in t or "water" in t or "drink" in t or "fill" in t
    if "drawer" in t or ((want_fork or want_spoon) and not drawer_open):
        if not drawer_open and not done.get("drawer_open"):
            # the handle is on arm A's side; honour an arm word only if it is said in the drawer clause
            seg = t.split("drawer")[0][-25:] + " " + t.split("drawer")[1][:25] if "drawer" in t else ""
            steps.append({"skill": "open_drawer", "arm": _arm_in(seg, "a")})
    if want_fork and not done.get("fork_placed"):
        steps.append({"skill": "pick_place", "arm": "a", "obj": "fork_1", "zone": "fork"})
    if want_spoon and not done.get("spoon_placed"):
        steps.append({"skill": "handoff", "arm": "a", "obj": "spoon_1", "to_arm": "b", "zone": "spoon"})
    if want_plate and not done.get("plate_placed"):
        steps.append({"skill": "pick_place", "arm": "a", "obj": "plate", "zone": "plate"})  # plate zone: arm A's side
    if want_pour and not done.get("poured"):
        pour_arm = "a"
        if "pour" in t:
            seg = t[t.index("pour"):]
            pour_arm = _arm_in(seg, "a")
        hold_arm = "b" if pour_arm == "a" else "a"
        steps.append({"skill": "hold_mug", "arm": hold_arm})
        steps.append({"skill": "pour", "arm": pour_arm, "amount": "little" if little else "normal"})
        steps.append({"skill": "place_mug", "arm": hold_arm})
    elif want_mug and not done.get("mug_placed"):
        steps.append({"skill": "pick_place", "arm": "b", "obj": "mug", "zone": "mug"})  # mug and its zone: arm B's side
    if "hold" in t and "mug" in t and not want_pour:
        steps.append({"skill": "hold_mug", "arm": _arm_in(t, "b")})
    if not steps:
        raise PlanError(f"rule planner found nothing to do in {command!r}")
    # "first the plate, then the mug": order object steps by where their object is first mentioned
    def mention(s: dict) -> int:
        obj = s.get("obj") or ("mug" if s["skill"] in ("hold_mug", "pour", "place_mug") else "drawer")
        word = obj.split("_")[0]
        i = t.find(word)
        return i if i >= 0 else -1
    if "then" in t or "first" in t or "after" in t:
        head = [s for s in steps if s["skill"] == "open_drawer"]
        rest = [s for s in steps if s["skill"] != "open_drawer"]
        rest.sort(key=mention)
        steps = head + rest
    return {"steps": steps, "mode": "gentle" if gentle else "normal"}


# ---------------------------------------------------------------------------
# VLM backend
# ---------------------------------------------------------------------------
class OpenVinoVLM:
    def __init__(self, model_dir: Path = MODEL_DIR, device: str = "CPU", max_new_tokens: int = 260):
        import openvino_genai as og
        t0 = time.perf_counter()
        self.pipe = og.VLMPipeline(str(model_dir), device)
        self.load_s = time.perf_counter() - t0
        self.device = device
        self.cfg = og.GenerationConfig()
        self.cfg.max_new_tokens = max_new_tokens
        self.cfg.do_sample = False
        self.cfg.repetition_penalty = 1.15  # a 2B model loops on hold_mug/place_mug without it
        self.last: dict = {}

    def __call__(self, image: np.ndarray, prompt: str) -> str:
        import openvino as ov
        img = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
        tensor = ov.Tensor(img[None])  # NHWC uint8
        # stateless: the system prompt travels inside the user turn. start_chat/finish_chat between calls
        # trips the GPU plugin (CL_INVALID_VALUE on the second generate) in openvino-genai 2026.3.
        full = SYSTEM + "\n\n" + prompt
        t0 = time.perf_counter()
        res = self.pipe.generate(full, image=tensor, generation_config=self.cfg)
        dt = time.perf_counter() - t0
        text = str(res)
        n_tok = None
        try:
            n_tok = int(res.perf_metrics.get_num_generated_tokens())
            ttft = float(res.perf_metrics.get_ttft().mean)
            tps = float(res.perf_metrics.get_throughput().mean)
        except Exception:  # pragma: no cover - perf metrics are optional
            ttft, tps = None, (n_tok / dt if n_tok else None)
        self.last = {"latency_s": dt, "tokens": n_tok, "tokens_per_s": tps, "ttft_ms": ttft, "device": self.device}
        return text


@dataclass
class PlanResult:
    plan: dict
    source: str                     # vlm | vlm_retry | rules
    attempts: list[dict] = field(default_factory=list)
    latency_s: float = 0.0
    vlm_metrics: dict = field(default_factory=dict)
    verdict: str | None = None


class Planner:
    def __init__(self, backend: str = "auto", device: str = "CPU", model_dir: Path = MODEL_DIR):
        # CPU by default: on this i7-13650HX the iGPU plugin crashes on the second generate (docs/BLOCKERS.md)
        # and its prompt processing is 10x slower than the CPU's anyway.
        self.vlm: OpenVinoVLM | None = None
        if backend in ("vlm", "auto") and Path(model_dir, "openvino_language_model.xml").exists():
            try:
                self.vlm = OpenVinoVLM(model_dir, device)
            except Exception as e:  # pragma: no cover
                if backend == "vlm":
                    raise
                self.vlm = None
                self.load_error = repr(e)
        elif backend == "vlm":
            raise FileNotFoundError(f"no exported model at {model_dir}; run planner/export.sh")
        self.backend = "vlm" if self.vlm else "rules"

    def _ask(self, image: np.ndarray | None, prompt: str) -> tuple[dict, str, dict]:
        assert self.vlm is not None
        if image is None:
            image = np.zeros((C.IMAGE_HEIGHT, C.IMAGE_WIDTH, 3), np.uint8)
        text = self.vlm(image, prompt)
        return normalise(parse_plan(text)), text, dict(self.vlm.last)

    def plan(self, command: str, scene: dict, scene_text: str, image: np.ndarray | None = None, done: str = "nothing",
             verifier=None, world=None) -> PlanResult:
        """VLM -> schema check -> (optional) verifier; one retry with the rejection reasons fed back; then rules.

        With ``verifier`` and ``world`` given, the returned plan is the verifier-approved one (ALLOW or the
        REORDERed plan) and ``PlanResult.verdict`` records the last verdict.
        """
        t0 = time.perf_counter()
        attempts: list[dict] = []
        if self.vlm is not None:
            prompt = USER.format(scene=scene_text, done=done, command=command)
            for k in range(2):
                try:
                    plan, raw, metrics = self._ask(image, prompt)
                    issues = [i.message for i in validate_schema(plan)]
                    verdict = None
                    if not issues and verifier is not None and world is not None:
                        v = verifier.verify(plan, world)
                        verdict = v.verdict
                        if v.verdict == "BLOCK":
                            issues = [i.message for i in v.issues]
                        else:
                            plan = v.plan
                    attempts.append({"try": k, "raw": raw[:600], "issues": issues, "verdict": verdict, "metrics": metrics})
                    if not issues:
                        return PlanResult(plan, "vlm" if k == 0 else "vlm_retry", attempts, time.perf_counter() - t0, metrics, verdict)
                    prompt = prompt + f"\n\nYour previous answer was rejected: {'; '.join(issues)}. Reply with only the corrected JSON."
                except PlanError as e:
                    attempts.append({"try": k, "error": str(e), "metrics": dict(self.vlm.last)})
                    prompt = prompt + "\n\nYour previous answer was not valid JSON. Reply with only the JSON object."
        plan = rule_plan(command, scene)
        verdict = None
        if verifier is not None and world is not None:
            v = verifier.verify(plan, world)
            verdict = v.verdict
            if v.verdict != "BLOCK":
                plan = v.plan
        return PlanResult(plan, "rules", attempts, time.perf_counter() - t0, attempts[-1]["metrics"] if attempts else {}, verdict)

    def replan(self, command: str, scene: dict, scene_text: str, plan_so_far: dict, failed_step: int, reason: str,
               image: np.ndarray | None = None) -> PlanResult:
        remaining = plan_so_far["steps"][failed_step:]
        t0 = time.perf_counter()
        attempts: list[dict] = []
        if self.vlm is not None:
            prompt = REPLAN.format(scene=scene_text, plan=json.dumps(plan_so_far["steps"]), failed_step=failed_step,
                                   reason=reason, remaining=json.dumps(remaining), command=command)
            try:
                p, raw, metrics = self._ask(image, prompt)
                issues = validate_schema(p)
                attempts.append({"try": 0, "raw": raw[:600], "issues": [i.message for i in issues], "metrics": metrics})
                if not issues:
                    return PlanResult(p, "vlm", attempts, time.perf_counter() - t0, metrics)
            except PlanError as e:
                attempts.append({"try": 0, "error": str(e)})
        # rule fallback: the remaining steps, with the failed one retried by the other arm where that is legal
        steps = [dict(s) for s in remaining]
        if steps:
            f = steps[0]
            if f["skill"] == "pick_place" and f.get("obj") in ("plate",):
                f["arm"] = "b" if f["arm"] == "a" else "a"
        return PlanResult({"steps": steps, "mode": plan_so_far.get("mode", "normal")}, "rules", attempts, time.perf_counter() - t0)
