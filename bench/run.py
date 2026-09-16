"""OpenVINO latency/throughput benchmark of the ACT IRs (plan Phase 8.3) -> results/bench.json + markdown.

For every skill IR x precision (fp32/fp16/int8) x device (CPU, GPU): p50/p95 latency of one policy call (batch 1,
3 cameras), throughput in the latency hint, plus the VLM planner's tokens/s and TTFT (from
results/planner_eval.json, measured on CPU).  Every table is captioned with the hardware note the README uses.

    python -m bench.run --devices CPU GPU --seconds 10
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import openvino as ov

from bench.export_ir import IR_ROOT, SKILLS

ROOT = Path(__file__).resolve().parent.parent
CAPTION = ("measured on i7-13650HX CPU + UHD iGPU; same IR runs on Core Ultra NPU with -d NPU and static shapes, "
           "not measured here.")


def hardware() -> dict:
    core = ov.Core()
    cpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    model = next((l.split(":", 1)[1].strip() for l in cpu.splitlines() if l.startswith("Model name")), platform.processor())
    devs = {d: core.get_property(d, "FULL_DEVICE_NAME") for d in core.available_devices}
    return {"cpu": model, "openvino": ov.__version__, "devices": devs, "npu": "NPU" in devs,
            "note": "Dell G15 5530, Intel Core i7-13650HX (Raptor Lake), UHD iGPU, no NPU, not Core Ultra"}


def bench_one(xml: Path, device: str, seconds: float) -> dict:
    core = ov.Core()
    model = core.read_model(xml)
    compiled = core.compile_model(model, device, {"PERFORMANCE_HINT": "LATENCY"})
    req = compiled.create_infer_request()
    inputs = {}
    for inp in compiled.inputs:
        shape = list(inp.get_partial_shape().get_min_shape())
        inputs[inp.get_any_name()] = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    for _ in range(5):
        req.infer(inputs)
    lat = []
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        req.infer(inputs)
        lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat)
    return {"n": int(len(lat)), "p50_ms": round(float(np.percentile(lat, 50)), 2), "p95_ms": round(float(np.percentile(lat, 95)), 2),
            "mean_ms": round(float(lat.mean()), 2), "throughput_per_s": round(1000 / float(lat.mean()), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="*", default=["CPU", "GPU"])
    ap.add_argument("--precisions", nargs="*", default=["fp32", "fp16", "int8"])
    ap.add_argument("--skills", nargs="*", default=SKILLS)
    ap.add_argument("--seconds", type=float, default=8.0)
    args = ap.parse_args()
    hw = hardware()
    avail = [d for d in args.devices if d in hw["devices"]]
    skipped = [d for d in args.devices if d not in hw["devices"]]
    rows = []
    for skill in args.skills:
        for prec in args.precisions:
            xml = IR_ROOT / f"act_{skill}" / prec / "model.xml"
            if not xml.exists():
                continue
            size_mb = round(sum(f.stat().st_size for f in xml.parent.iterdir()) / 1e6, 1)
            for dev in avail:
                try:
                    r = bench_one(xml, dev, args.seconds)
                    rows.append({"skill": skill, "precision": prec, "device": dev, "size_mb": size_mb, **r})
                    print(f"{skill:17s} {prec:5s} {dev:4s} p50 {r['p50_ms']:7.2f} ms  p95 {r['p95_ms']:7.2f} ms  {r['throughput_per_s']:6.1f}/s", flush=True)
                except Exception as e:
                    rows.append({"skill": skill, "precision": prec, "device": dev, "size_mb": size_mb, "error": repr(e)[:200]})
                    print(f"{skill:17s} {prec:5s} {dev:4s} ERROR {e!r}"[:120])
    vlm = {}
    pe = ROOT / "results" / "planner_eval.json"
    if pe.exists():
        d = json.loads(pe.read_text())
        vlm = {"model": d["model"], "device": d["device"], "tokens_per_s": d["vlm_tokens_per_s"], "ttft_ms": d["vlm_ttft_ms"],
               "mean_plan_latency_s": d["mean_latency_s"], "igpu": "loads and answers once, then crashes in the GPU plugin (docs/BLOCKERS.md)"}
    # summary per (precision, device): mean over skills
    summary = {}
    for r in rows:
        if "error" in r:
            continue
        k = f"{r['precision']}/{r['device']}"
        summary.setdefault(k, []).append(r["p50_ms"])
    summary = {k: {"mean_p50_ms": round(float(np.mean(v)), 2), "skills": len(v)} for k, v in summary.items()}
    out = {"caption": CAPTION, "hardware": hw, "devices_benchmarked": avail, "devices_skipped": skipped + (["NPU"] if not hw["npu"] else []),
           "seconds_per_case": args.seconds, "rows": rows, "summary": summary, "vlm_planner": vlm}
    (ROOT / "results" / "bench.json").write_text(json.dumps(out, indent=2))
    md = [f"_{CAPTION}_", "", "| skill | precision | device | size MB | p50 ms | p95 ms | infer/s |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['skill']} | {r['precision']} | {r['device']} | {r['size_mb']} | " + (f"{r['p50_ms']} | {r['p95_ms']} | {r['throughput_per_s']} |" if "error" not in r else f"error | | |"))
    if vlm:
        md += ["", f"VLM planner ({vlm['model']}) on {vlm['device']}: **{vlm['tokens_per_s']} tok/s**, TTFT {vlm['ttft_ms']} ms, mean plan {vlm['mean_plan_latency_s']} s. iGPU: {vlm['igpu']}."]
    (ROOT / "results" / "bench.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[-8:]))


if __name__ == "__main__":
    main()
