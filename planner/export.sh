#!/usr/bin/env bash
# Export Qwen2-VL-2B-Instruct to OpenVINO IR with INT4 weights (plan Phase 4.1).
# Source: openvino_notebooks/notebooks/qwen2-vl export cell. Output dir is gitignored (~1.5 GB).
set -euo pipefail
# optimum stages the IR in $TMPDIR before moving it; /tmp is a small tmpfs on this box, so use a disk-backed dir
export TMPDIR="${TMPDIR:-$HOME/tmp}"; mkdir -p "$TMPDIR"
cd "$(dirname "$0")/.."
OUT="$(pwd)/planner/qwen2vl_int4"
mkdir -p "$OUT"
MODEL=${MODEL:-Qwen/Qwen2-VL-2B-Instruct}
if [ -f "$OUT/openvino_language_model.xml" ]; then
  echo "already exported: $OUT"; exit 0
fi
.venv/bin/optimum-cli export openvino --model "$MODEL" "$OUT" \
  --task image-text-to-text --weight-format int4 --group-size 128 --ratio 1.0 --trust-remote-code
ls -la "$OUT"
