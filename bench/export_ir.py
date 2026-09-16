"""Export the per-skill ACT policies to OpenVINO IR at FP32 and FP16 (plan Phase 8.1).

The wrapped module takes exactly what the LeRobot pre-processor hands the policy (normalised state and the
three camera tensors, batch 1) and returns the action chunk (1, chunk, 12); normalisation stays in the
LeRobot processors so the IR is a drop-in for ``policy.model``.  Static shapes: batch 1, images 3x240x320 --
which is also what an NPU would need (not present on this box; see README).  Writes
bench/ir/act_<skill>/{fp32,fp16}/model.xml and results/ir_export.json (sizes, max |torch - IR| on real frames).

    python -m bench.export_ir [--skills open_drawer ...]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import openvino as ov
import torch

from souschef_env import constants as C
from runtime.executors import LoadedPolicy

ROOT = Path(__file__).resolve().parent.parent
IR_ROOT = ROOT / "bench" / "ir"
SKILLS = ["open_drawer", "pick_place_fork", "pick_place_plate", "pick_place_mug", "handoff_spoon", "hold_mug", "pour"]
INPUT_NAMES = ["observation.state", *(f"observation.images.{c}" for c in C.CAMERAS)]


class ACTWrapper(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.act = policy.model
        self.keys = list(policy.config.image_features)

    def forward(self, state: torch.Tensor, overhead: torch.Tensor, wrist_a: torch.Tensor, wrist_b: torch.Tensor) -> torch.Tensor:
        imgs = {f"observation.images.overhead": overhead, "observation.images.wrist_a": wrist_a, "observation.images.wrist_b": wrist_b}
        batch = {"observation.state": state, "observation.images": [imgs[k] for k in self.keys]}
        return self.act(batch)[0]


def checkpoint(skill: str) -> Path:
    return ROOT / "outputs" / f"act_{skill}" / "checkpoints" / "last" / "pretrained_model"


def example_inputs(pol: LoadedPolicy, device: str = "cpu") -> dict[str, torch.Tensor]:
    """A real normalised observation from the dataset's first frame of this skill, batch 1."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("Prashant-77/thali_all", root=ROOT / "data" / "lerobot" / "thali_all")
    idx = int(ds.meta.episodes["dataset_from_index"][pol.first_episode]) if hasattr(pol, "first_episode") else 0
    item = ds[idx]
    obs = {"observation.state": item["observation.state"][None], **{k: item[k][None] for k in INPUT_NAMES[1:]}}
    obs = pol.pre(obs)
    return {k: obs[k].to(device).float() for k in INPUT_NAMES}


def export_skill(skill: str, device_check: str = "CPU") -> dict:
    pol = LoadedPolicy(checkpoint(skill), device="cpu")
    pol.policy.eval()
    wrapper = ACTWrapper(pol.policy).eval()
    ex = example_inputs(pol)
    args = tuple(ex[k] for k in INPUT_NAMES)
    with torch.no_grad():
        ref = wrapper(*args).numpy()
    t0 = time.time()
    ov_model = ov.convert_model(wrapper, example_input=args, input=[tuple(a.shape) for a in args])
    for inp, name in zip(ov_model.inputs, ["state", "overhead", "wrist_a", "wrist_b"]):
        inp.get_tensor().set_names({name})
    ov_model.outputs[0].get_tensor().set_names({"actions"})
    out: dict = {"skill": skill, "chunk": int(ref.shape[1]), "action_dim": int(ref.shape[2]), "convert_s": round(time.time() - t0, 1), "precisions": {}}
    core = ov.Core()
    for prec, fp16 in (("fp32", False), ("fp16", True)):
        d = IR_ROOT / f"act_{skill}" / prec
        d.mkdir(parents=True, exist_ok=True)
        ov.save_model(ov_model, d / "model.xml", compress_to_fp16=fp16)
        size_mb = round(sum(f.stat().st_size for f in d.iterdir()) / 1e6, 1)
        compiled = core.compile_model(core.read_model(d / "model.xml"), device_check)
        pred = compiled({k: a.numpy() for k, a in zip(["state", "overhead", "wrist_a", "wrist_b"], args)})[compiled.output(0)]
        out["precisions"][prec] = {"path": str(d / "model.xml"), "size_mb": size_mb, "max_abs_diff_vs_torch": float(np.abs(pred - ref).max()),
                                   "mean_abs_diff_vs_torch": float(np.abs(pred - ref).mean())}
        print(f"  {skill} {prec}: {size_mb} MB, max|Δ| {out['precisions'][prec]['max_abs_diff_vs_torch']:.2e}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills", nargs="*", default=SKILLS)
    args = ap.parse_args()
    res = {"model": "ACT (resnet18 backbone, chunk 50), per skill", "inputs": "batch 1: state (1,12) + 3 x (1,3,240,320) float, LeRobot-normalised",
           "static_shapes": True, "note": "static batch-1 shapes as an NPU would need; no NPU on this box, not measured", "skills": {}}
    prev = ROOT / "results" / "ir_export.json"
    if prev.exists():
        res["skills"] = json.loads(prev.read_text()).get("skills", {})
    for s in args.skills:
        if not (checkpoint(s) / "config.json").exists():
            print(f"  {s}: no checkpoint yet, skipped")
            continue
        res["skills"][s] = export_skill(s)
    (ROOT / "results" / "ir_export.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
