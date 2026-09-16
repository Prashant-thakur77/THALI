"""Export the SmolVLA vision encoder (+ connector) to OpenVINO IR and benchmark it (plan Phase 8.1/8.3).

The encoder is shared by the base model and any fine-tune (it is frozen during SmolVLA training), so it can be
exported and measured from ``lerobot/smolvla_base`` before the Kaggle checkpoint exists; the action expert IR is
exported from the fine-tuned checkpoint when ``Prashant-77/thali_smolvla`` is available (``--checkpoint``).
Static shape: one 512x512 image (SmolVLA's resize-with-padding size), which is what an NPU needs.  Writes
bench/ir/smolvla_vision/{fp32,fp16}/model.xml and results/smolvla_ir.json.

    python -m bench.export_smolvla [--checkpoint <dir>] [--devices CPU GPU]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import openvino as ov
import torch

ROOT = Path(__file__).resolve().parent.parent
IR = ROOT / "bench" / "ir" / "smolvla_vision"


class VisionWrapper(torch.nn.Module):
    def __init__(self, vlm):
        super().__init__()
        self.vision = vlm.vision_model
        self.connector = vlm.connector

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        h = self.vision(pixel_values=pixel_values, patch_attention_mask=None).last_hidden_state
        return self.connector(h)


def bench(xml: Path, device: str, seconds: float = 6.0) -> dict:
    core = ov.Core()
    compiled = core.compile_model(core.read_model(xml), device, {"PERFORMANCE_HINT": "LATENCY"})
    req = compiled.create_infer_request()
    x = np.random.default_rng(0).standard_normal((1, 3, 512, 512)).astype(np.float32)
    for _ in range(3):
        req.infer({0: x})
    lat, t_end = [], time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        req.infer({0: x})
        lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat)
    return {"n": len(lat), "p50_ms": round(float(np.percentile(lat, 50)), 1), "p95_ms": round(float(np.percentile(lat, 95)), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="lerobot/smolvla_base")
    ap.add_argument("--devices", nargs="*", default=["CPU", "GPU"])
    args = ap.parse_args()
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    t0 = time.time()
    pol = SmolVLAPolicy.from_pretrained(args.checkpoint).eval().float()
    vlm = pol.model.vlm_with_expert.get_vlm_model()
    wrapper = VisionWrapper(vlm).eval()
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        ref = wrapper(x).numpy()
    m = ov.convert_model(wrapper, example_input=(x,), input=[(1, 3, 512, 512)])
    out = {"source": args.checkpoint, "component": "SigLIP vision encoder + connector (frozen in SmolVLA fine-tuning)",
           "input": "1x3x512x512", "output_shape": list(ref.shape), "load_convert_s": round(time.time() - t0, 1), "precisions": {}, "bench": [],
           "action_expert": "exported from the fine-tuned checkpoint when Prashant-77/thali_smolvla exists (pending)" if "smolvla_base" in args.checkpoint else "n/a",
           "caption": "measured on i7-13650HX CPU + UHD iGPU; same IR runs on Core Ultra NPU with -d NPU and static shapes, not measured here."}
    core = ov.Core()
    for prec, fp16 in (("fp32", False), ("fp16", True)):
        d = IR / prec
        d.mkdir(parents=True, exist_ok=True)
        ov.save_model(m, d / "model.xml", compress_to_fp16=fp16)
        size = round(sum(f.stat().st_size for f in d.iterdir()) / 1e6, 1)
        pred = core.compile_model(core.read_model(d / "model.xml"), "CPU")({0: x.numpy()})[0]
        out["precisions"][prec] = {"size_mb": size, "max_abs_diff_vs_torch": float(np.abs(pred - ref).max())}
        for dev in args.devices:
            if dev not in core.available_devices:
                continue
            try:
                r = bench(d / "model.xml", dev)
                out["bench"].append({"precision": prec, "device": dev, **r})
                print(f"smolvla vision {prec} {dev}: p50 {r['p50_ms']} ms p95 {r['p95_ms']} ms ({size} MB)")
            except Exception as e:
                out["bench"].append({"precision": prec, "device": dev, "error": repr(e)[:160]})
    (ROOT / "results" / "smolvla_ir.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
