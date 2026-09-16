import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_planner_eval_results():
    d = json.loads((ROOT / "results" / "planner_eval.json").read_text())
    assert d["commands"] == 8 and d["accepted_from_vlm"] + d["accepted_from_rules"] == 8
    assert d["verifier_approved_rate"] == 1.0 and d["covers_requested_rate"] == 1.0
    assert d["vlm_tokens_per_s"] > 5 and d["device"] == "CPU"
    assert "no NPU" in d["hardware"]
