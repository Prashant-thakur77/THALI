"""Skill executors for the runtime (plan Phase 3.3): learned policy first, then retry, then the scripted expert.

``PolicyExecutor.run(step)`` picks the checkpoint for the step's skill (per-skill ACT, or the multi-task SmolVLA
when ``--policy smolvla`` and the checkpoint exists), rolls it out closed-loop in ThaliEnv for up to
``max_steps`` control steps, and stops early when the skill's oracle is satisfied.  Modes:
  policy_only   one rollout, report the oracle
  policy_retry  a second rollout from wherever the first one left the scene
  policy_fallback  after the retry, hand the step to the scripted expert
Every run records which stage succeeded, so eval/run_seeds.py can report the three rates side by side.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from souschef_env import constants as C
from souschef_env import oracles
from expert.primitives import Expert, SkillResult
from expert.task import Step
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors, make_policy
from lerobot.configs.policies import PreTrainedConfig

ROOT = Path(__file__).resolve().parent.parent

# plan step -> (dataset skill name, oracle check kwargs)
SKILL_OF = {
    ("open_drawer", None): "open_drawer",
    ("pick_place", "fork_1"): "pick_place_fork", ("pick_place", "fork_2"): "pick_place_fork",
    ("pick_place", "plate"): "pick_place_plate", ("pick_place", "mug"): "pick_place_mug",
    ("handoff", "spoon_1"): "handoff_spoon", ("handoff", "spoon_2"): "handoff_spoon",
    ("hold_mug", None): "hold_mug", ("pour", None): "pour",
}
MAX_STEPS = {"open_drawer": 700, "pick_place_fork": 800, "pick_place_plate": 700, "pick_place_mug": 800,
             "handoff_spoon": 1300, "hold_mug": 600, "pour": 1600}
LANG_INSTRUCTION = {  # canonical instruction per skill for the language-conditioned policy
    "open_drawer": "open the top drawer with arm A",
    "pick_place_fork": "pick up a fork with arm A and place it on the left of the plate",
    "pick_place_plate": "pick up the plate with arm A and place it on the plate spot",
    "pick_place_mug": "pick up the mug with arm B and place it on the mug spot",
    "handoff_spoon": "hand a spoon from arm A to arm B",
    "hold_mug": "hold the mug steady with arm B",
    "pour": "pour water into the mug with arm A",
}


def skill_name(step: dict) -> str | None:
    return SKILL_OF.get((step["skill"], step.get("obj")))


def skill_done(ex: Expert, step: dict) -> bool:
    m, d = ex.m, ex.d
    sk = step["skill"]
    if sk == "open_drawer":
        return oracles.drawer_open(m, d)
    if sk == "pick_place":
        return oracles.object_in_zone(m, d, step["obj"], step["zone"])
    if sk == "handoff":
        return oracles.object_in_zone(m, d, step["obj"], step["zone"]) if step.get("zone") else oracles.held_by(m, d, step["obj"]) == step["to_arm"]
    if sk == "hold_mug":
        return oracles.mug_held(m, d) == step["arm"] and abs(oracles.object_pos(m, d, "mug")[2] - C.POUR_POSE[2]) < 0.03
    if sk == "pour":
        return oracles.poured(m, d)
    if sk == "place_mug":
        return oracles.object_in_zone(m, d, "mug", "mug")
    return False


class LoadedPolicy:
    def __init__(self, path: Path, device: str = "cuda"):
        cfg = PreTrainedConfig.from_pretrained(str(path))
        cfg.pretrained_path = path
        cfg.device = device if torch.cuda.is_available() else "cpu"
        self.policy = make_policy(cfg, ds_meta=None, env_cfg=None) if False else self._load(cfg, path)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(path),
                                                       preprocessor_overrides={"device_processor": {"device": str(cfg.device)}})
        self.device = cfg.device
        self.type = cfg.type

    @staticmethod
    def _load(cfg, path):
        from lerobot.policies.factory import get_policy_class
        cls = get_policy_class(cfg.type)
        return cls.from_pretrained(str(path), config=cfg)

    def reset(self) -> None:
        self.policy.reset()

    @torch.no_grad()
    def act(self, obs: dict, task: str) -> np.ndarray:
        o = preprocess_observation({"pixels": obs["pixels"], "agent_pos": obs["agent_pos"]})
        o["task"] = [task]
        o = self.pre(o)
        a = self.policy.select_action(o)
        a = self.post(a)
        return a[0].detach().cpu().numpy()


class PolicyExecutor:
    """Learned policy -> retry -> scripted fallback, per plan step."""

    def __init__(self, ex: Expert, kind: str = "act", mode: str = "policy_fallback", device: str = "cuda",
                 act_root: Path = ROOT / "outputs", smolvla_path: Path | None = None):
        self.ex = ex
        self.kind = kind
        self.mode = mode
        self.device = device
        self.act_root = act_root
        self.smolvla_path = smolvla_path
        self._cache: dict[str, LoadedPolicy] = {}
        self.name = f"{kind}:{mode}"
        self.last: dict[str, Any] = {}

    def _policy_for(self, skill: str) -> LoadedPolicy | None:
        if self.kind == "smolvla":
            key = "smolvla"
            path = self.smolvla_path
        else:
            key = skill
            path = self.act_root / f"act_{skill}" / "checkpoints" / "last" / "pretrained_model"
        if path is None or not (Path(path) / "config.json").exists():
            return None
        if key not in self._cache:
            self._cache[key] = LoadedPolicy(Path(path), self.device)
        return self._cache[key]

    def _rollout(self, pol: LoadedPolicy, step: dict, skill: str) -> tuple[bool, int, float]:
        env = self.ex.env
        env.render_enabled = True
        pol.reset()
        obs = env._obs()
        t0 = time.perf_counter()
        n = 0
        task = LANG_INSTRUCTION[skill]
        for n in range(1, MAX_STEPS[skill] + 1):
            if self.ex.interrupt is not None and self.ex.interrupt():
                from expert.primitives import Interrupted
                raise Interrupted()
            a = pol.act(obs, task)
            obs, _, _, _, _ = env.step(a)
            self.ex.steps += 1
            if self.ex.on_step:
                self.ex.on_step(a, obs)
            if n % 10 == 0 and skill_done(self.ex, step):
                break
        self.ex.sync_from_env()
        return bool(skill_done(self.ex, step)), n, time.perf_counter() - t0

    def run(self, step: dict) -> SkillResult:
        skill = skill_name(step)
        pol = self._policy_for(skill) if skill else None
        stages: list[dict] = []
        if pol is None:
            r = Step(step["skill"], step["arm"], step.get("obj"), step.get("zone"), step.get("to_arm")).run(self.ex)
            r.detail = {**r.detail, "stage": "expert_only", "reason": "no policy for this step", "stages": []}
            return r
        ok, n, dt = self._rollout(pol, step, skill)
        stages.append({"stage": "policy", "ok": ok, "steps": n, "seconds": round(dt, 2)})
        if not ok and self.mode in ("policy_retry", "policy_fallback"):
            self.ex.open_jaw(step["arm"])
            self.ex.park(step["arm"])
            ok, n, dt = self._rollout(pol, step, skill)
            stages.append({"stage": "retry", "ok": ok, "steps": n, "seconds": round(dt, 2)})
        if not ok and self.mode == "policy_fallback":
            self.ex.open_jaw(step["arm"])
            self.ex.park(step["arm"])
            r = Step(step["skill"], step["arm"], step.get("obj"), step.get("zone"), step.get("to_arm")).run(self.ex)
            ok = bool(r.ok)
            stages.append({"stage": "fallback", "ok": ok, "steps": int(r.steps)})
        else:
            self.ex.open_jaw(step["arm"]) if step["skill"] not in ("hold_mug",) else None
        won = next((s["stage"] for s in stages if s["ok"]), None)
        self.last = {"skill": skill, "won_by": won, "stages": stages}
        return SkillResult(step["skill"], ok, sum(s.get("steps", 0) for s in stages), {"policy": self.kind, "won_by": won, "stages": stages, "mode": self.mode})
