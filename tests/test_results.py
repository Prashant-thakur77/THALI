"""Every results file the README renders from exists, is internally consistent, and carries the hardware caption."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"


def load(n):
    return json.loads((R / n).read_text())


def test_seed_tables_and_aggregate():
    agg = load("seeds.json")["rows"]
    have = {(r["policy"], r["mode"], r["split"]) for r in agg if "successes" in r}
    assert {("expert", "expert", "test"), ("act", "policy_only", "test"), ("act", "policy_retry", "test"), ("act", "policy_fallback", "test")} <= have
    for r in agg:
        if "successes" in r:
            d = load(r["file"])
            assert d["seeds"] == 10 and d["successes"] == sum(x["success"] for x in d["rows"])
        else:
            assert r["policy"] == "smolvla" and "pending" in r["status"]


def test_heatmap_expert_consistent():
    h = load("heatmap_expert.json")
    assert h["seeds"] == 10 and set(h["columns"]) == {"placement", "mass", "friction", "shape", "lighting", "background", "all"}
    for c in h["columns"]:
        assert abs(sum(h["matrix"][c]) / 10 - h["per_axis_success_rate"][c]) < 1e-9
    assert (R / "heatmap_expert.png").exists()


def test_bench_and_preserve_captioned_and_no_npu():
    b = load("bench.json")
    assert "not measured here" in b["caption"] and "NPU" in b["devices_skipped"] and not b["hardware"]["npu"]
    assert {"CPU", "GPU"} == set(b["devices_benchmarked"])
    assert b["summary"]["int8/CPU"]["mean_p50_ms"] < b["summary"]["fp32/CPU"]["mean_p50_ms"]
    p = load("preserve.json")
    assert "not measured here" in p["caption"] and set(p["table"]) >= {"fp32", "fp16", "int8"}
    ir = load("ir_export.json")
    assert len(ir["skills"]) == 7 and all("int8" in v["precisions"] for v in ir["skills"].values())
    assert all(v["precisions"]["fp32"]["max_abs_diff_vs_torch"] < 1e-4 for v in ir["skills"].values())
    s = load("smolvla_ir.json")
    assert s["precisions"]["fp32"]["max_abs_diff_vs_torch"] < 1e-2 and any(r.get("device") == "CPU" and "p50_ms" in r for r in s["bench"])


def test_camera_swap_planner_voice_verifier():
    c = load("camera_vs_oracle.json")
    assert c["seeds"] == 10 and c["backends"]["pixels"]["agreement"] > c["backends"]["vlm"]["agreement"]
    s = load("instruction_swap.json")
    assert s["total"] == 10 and 0 <= s["correct"] <= 10
    p = load("planner_eval.json")
    assert p["verifier_approved_rate"] == 1.0
    v = load("voice_test.json")
    assert v["samples"] == 4
    inj = load("verifier_injection.json")
    assert inj["caught"] == 20


def test_readme_renders_without_errors_and_states_hardware():
    txt = (ROOT / "README.md").read_text()
    assert "⚠️" not in txt and "No NPU" in txt and "i7-13650HX" in txt
    assert "pending" in txt  # SmolVLA rows are honestly pending until the Kaggle run
