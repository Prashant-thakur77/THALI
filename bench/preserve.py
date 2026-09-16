"""Does optimisation preserve success? 10-seed full-task success at each precision (plan Phase 8.4).

The ACT policies are run in the env through OpenVINO (fp32 / fp16 / int8 IR on the chosen device) instead of
PyTorch, with the same LeRobot pre/post-processors, in ``policy_only`` mode (no retry, no fallback: any drop is
the optimisation's).  Writes results/preserve.json with the delta table against the PyTorch run.

    python -m bench.preserve --device CPU --precisions fp32 fp16 int8 --seeds 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import openvino as ov
import torch

from bench.export_ir import IR_ROOT
from eval.run_seeds import run
from runtime import executors
from runtime.executors import LoadedPolicy

ROOT = Path(__file__).resolve().parent.parent


class OVActPolicy(LoadedPolicy):
    """Same interface as LoadedPolicy.act, but the network is the OpenVINO IR."""

    def __init__(self, path: Path, skill: str, precision: str, device: str):
        super().__init__(path, device="cpu")
        core = ov.Core()
        self.compiled = core.compile_model(core.read_model(IR_ROOT / f"act_{skill}" / precision / "model.xml"), device,
                                           {"PERFORMANCE_HINT": "LATENCY"})
        self.req = self.compiled.create_infer_request()
        self.n_action_steps = self.policy.config.n_action_steps
        self.queue: list[np.ndarray] = []

    def reset(self) -> None:
        self.queue = []

    @torch.no_grad()
    def act(self, obs: dict, task: str) -> np.ndarray:
        if not self.queue:
            from lerobot.envs.utils import preprocess_observation
            o = preprocess_observation({"pixels": obs["pixels"], "agent_pos": obs["agent_pos"]})
            o["task"] = [task]
            o = self.pre(o)
            feed = {"state": o["observation.state"].float().numpy(), "overhead": o["observation.images.overhead"].float().numpy(),
                    "wrist_a": o["observation.images.wrist_a"].float().numpy(), "wrist_b": o["observation.images.wrist_b"].float().numpy()}
            chunk = self.req.infer(feed)[self.compiled.output(0)][0]  # (chunk, 12) normalised
            acts = self.post(torch.from_numpy(chunk[: self.n_action_steps]).float())
            self.queue = [a.numpy() for a in acts]
        return self.queue.pop(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--precisions", nargs="*", default=["fp32", "fp16", "int8"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    results = {}
    torch_run = ROOT / "results" / f"seeds_act_policy_only_{args.split}.json"
    if torch_run.exists():
        d = json.loads(torch_run.read_text())
        results["torch"] = {"successes": d["successes"], "seeds": d["seeds"], "per_subgoal_rate": d["per_subgoal_rate"]}
    orig = executors.PolicyExecutor._policy_for
    for prec in args.precisions:
        def _policy_for(self, skill, prec=prec):
            key = f"{skill}:{prec}"
            path = self.act_root / f"act_{skill}" / "checkpoints" / "last" / "pretrained_model"
            if not (path / "config.json").exists() or not (IR_ROOT / f"act_{skill}" / prec / "model.xml").exists():
                return None
            if key not in self._cache:
                self._cache[key] = OVActPolicy(path, skill, prec, args.device)
            return self._cache[key]
        executors.PolicyExecutor._policy_for = _policy_for
        out = run("act", "policy_only", args.seeds, args.split, __import__("souschef_env.randomize", fromlist=["AXES"]).AXES, "cpu",
                  tag=f"preserve_{prec}_{args.device}")
        results[prec] = {"successes": out["successes"], "seeds": out["seeds"], "per_subgoal_rate": out["per_subgoal_rate"], "device": args.device}
    executors.PolicyExecutor._policy_for = orig
    base = results.get("torch") or results.get("fp32")
    table = {k: {"successes": v["successes"], "delta_vs_torch": (v["successes"] - base["successes"]) if base else None} for k, v in results.items()}
    out = {"caption": "measured on i7-13650HX CPU + UHD iGPU; same IR runs on Core Ultra NPU with -d NPU and static shapes, not measured here.",
           "device": args.device, "split": args.split, "seeds": args.seeds, "mode": "policy_only (no retry, no fallback)", "results": results, "table": table}
    (ROOT / "results" / "preserve.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
