"""Download the pre-converted OpenVINO INT4 Qwen3-VL-4B (PegBit's planner model) for the planner comparison.

    python -m planner.get_qwen3vl
    THALI_PLANNER_MODEL=planner/qwen3vl_int4 python -m eval.planner_eval --device CPU
"""

from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "planner" / "qwen3vl_int4"

if __name__ == "__main__":
    snapshot_download("OpenVINO/Qwen3-VL-4B-Instruct-int4-ov", local_dir=OUT)
    print("downloaded to", OUT, sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) // 1_000_000, "MB")
