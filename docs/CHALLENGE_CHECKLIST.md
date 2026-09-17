# Challenge checklist — rubric row → what exists → where

| Rubric row (pts) | Deliverable | Evidence |
|---|---|---|
| Task completion + bimanual (30) | 7-skill dinner task: drawer → fork → spoon **handed A→B via the table** → plate → **B holds mug while A pours** → mug set down | expert 5/10 test / 7/10 train; ACT policy-only 0/10, +retry 0/10, +fallback 0/10 → `results/seeds.json` |
| VLA / multi-modal (20) | per-skill ACT (trained), multi-task SmolVLA (pending Kaggle run), local Qwen2-VL-2B INT4 planner on OpenVINO, camera yes/no state check, instruction-swap matrix, replan | `results/planner_eval.json` (4/8 VLM, 100% approved), `results/instruction_swap.json` (7/10), `results/camera_vs_oracle.json` |
| Robustness (15) | 6-axis randomiser, held-out test ranges (+2 textures, +1 mug shape), seeds × axis heatmap, policy-only / retry / fallback | `results/heatmap_expert.json` (hardest: shape), `results/heatmap_act.json`, `results/seeds_*` |
| OpenVINO / Core Ultra (20) | ACT → IR fp32/fp16 + NNCF int8, CPU + iGPU latency, 10-seed preservation per precision, VLM tok/s | `results/bench.json` (fp32/CPU 49.13 ms, fp32/GPU 283.24 ms, fp16/CPU 49.51 ms, fp16/GPU 282.82 ms, int8/CPU 17.06 ms, int8/GPU 201.85 ms), `results/preserve.json`, `results/ir_export.json`. **No NPU here**; static shapes ready for `-d NPU`. |
| Reproducibility (10) | `Makefile` (demos/train/eval/bench/demo/verify-log/test), `Dockerfile`, `environment.yml`+`requirements.txt` pins, CI, seeded env, dataset+checkpoints (HF push pending token) | `tests/` (see CI), `docs/KAGGLE_TODO.md` |
| Innovation (5) | verifier with reach/precondition/workspace/order rules + hash-chained audit log; speaker-focused barge-in | `results/verifier_injection.json` (20/20), `results/audit.jsonl`, `results/demo_bargein_stop.json` |
| Speechmatics bonus | realtime, partials, end-of-turn 0.6 s, speaker diarization → speaker focus, custom vocab, `hi` session, TTS loop, barge-in, latency | `results/voice_test.json`, `results/demo_seed3.json` (10.447 s speech-end → arm-moves) |
