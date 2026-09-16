"""NNCF post-training INT8 quantisation of the ACT IRs with ~300 calibration frames (plan Phase 8.1).

Pattern: nncf examples/post_training_quantization/openvino/mobilenet_v2 -- ``nncf.quantize(ov_model,
nncf.Dataset(loader, transform_fn))``.  Calibration frames are real dataset frames of the same skill run through
the LeRobot pre-processor, so the quantiser sees exactly the input distribution the policy sees at run time.
Writes bench/ir/act_<skill>/int8/model.xml and updates results/ir_export.json.

    python -m bench.quantize [--skills ...] [--frames 300]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nncf
import numpy as np
import openvino as ov
import torch

from souschef_env import constants as C
from bench.export_ir import IR_ROOT, INPUT_NAMES, SKILLS, checkpoint
from runtime.executors import LoadedPolicy

ROOT = Path(__file__).resolve().parent.parent


def calibration_items(skill: str, n: int) -> list[dict[str, np.ndarray]]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    demos = json.loads((ROOT / "results" / "demos.json").read_text())
    ds = LeRobotDataset("Prashant-77/thali_all", root=ROOT / "data" / "lerobot" / "thali_all")
    pol = LoadedPolicy(checkpoint(skill), device="cpu")
    eps = demos["skills"][skill]["episodes"]
    starts = [int(ds.meta.episodes["dataset_from_index"][e]) for e in eps]
    lengths = [int(ds.meta.episodes["dataset_to_index"][e]) - s for e, s in zip(eps, starts)]
    rng = np.random.default_rng(0)
    items = []
    for _ in range(n):
        k = int(rng.integers(len(eps)))
        idx = starts[k] + int(rng.integers(lengths[k]))
        it = ds[idx]
        obs = {"observation.state": it["observation.state"][None], **{key: it[key][None] for key in INPUT_NAMES[1:]}}
        obs = pol.pre(obs)
        items.append({name: obs[k].float().numpy() for k, name in zip(INPUT_NAMES, ["state", "overhead", "wrist_a", "wrist_b"])})
    return items


def quantize_skill(skill: str, frames: int) -> dict:
    core = ov.Core()
    model = core.read_model(IR_ROOT / f"act_{skill}" / "fp32" / "model.xml")
    items = calibration_items(skill, frames)
    calib = nncf.Dataset(items, lambda x: x)
    q = nncf.quantize(model, calib, preset=nncf.QuantizationPreset.MIXED, subset_size=frames,
                      model_type=nncf.ModelType.TRANSFORMER)
    d = IR_ROOT / f"act_{skill}" / "int8"
    d.mkdir(parents=True, exist_ok=True)
    ov.save_model(q, d / "model.xml")
    size_mb = round(sum(f.stat().st_size for f in d.iterdir()) / 1e6, 1)
    # accuracy on the calibration frames: INT8 vs FP32 action chunk
    c32 = core.compile_model(model, "CPU")
    c8 = core.compile_model(core.read_model(d / "model.xml"), "CPU")
    diffs = []
    for it in items[:50]:
        a = c32(it)[c32.output(0)]
        b = c8(it)[c8.output(0)]
        diffs.append(np.abs(a - b).mean())
    out = {"path": str(d / "model.xml"), "size_mb": size_mb, "calibration_frames": frames, "preset": "MIXED, model_type=TRANSFORMER",
           "mean_abs_diff_vs_fp32": float(np.mean(diffs)), "max_mean_abs_diff_vs_fp32": float(np.max(diffs))}
    print(f"  {skill} int8: {size_mb} MB, mean|Δ| vs fp32 {out['mean_abs_diff_vs_fp32']:.4f} (normalised action units)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills", nargs="*", default=SKILLS)
    ap.add_argument("--frames", type=int, default=300)
    args = ap.parse_args()
    res_path = ROOT / "results" / "ir_export.json"
    res = json.loads(res_path.read_text())
    for s in args.skills:
        if s not in res["skills"]:
            print(f"  {s}: no IR yet, skipped")
            continue
        res["skills"][s]["precisions"]["int8"] = quantize_skill(s, args.frames)
        res_path.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
