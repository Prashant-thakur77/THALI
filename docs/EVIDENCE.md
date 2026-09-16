# Evidence index

Every number below is rendered from the named `results/*.json` file by `python -m docs.render_readme`. Regenerate after any run.

| claim | number | produced by | file |
|---|---|---|---|
| Expert full task, test split (10 seeds) | 5/10 | `eval/run_seeds.py --policy expert --split test` | results/seeds_expert_expert_test.json |
| Expert full task, train split | 7/10 | `eval/run_seeds.py --policy expert --split train` | results/seeds_expert_expert_train.json |
| ACT policy-only, test | 0/10 | `eval/run_seeds.py --policy act --mode policy_only` | results/seeds_act_policy_only_test.json |
| ACT + retry, test | 0/10 | `eval/run_seeds.py --policy act --mode policy_retry` | results/seeds_act_policy_retry_test.json |
| ACT + retry + expert fallback, test | 0/10 | `eval/run_seeds.py --policy act --mode policy_fallback` | results/seeds_act_policy_fallback_test.json |
| SmolVLA rows | pending SmolVLA run (docs/KAGGLE_TODO.md) | `eval/run_seeds.py --policy smolvla` | results/seeds.json |
| Heatmap seeds × axis, expert | hardest axis shape | `eval/heatmap.py --policy expert` | results/heatmap_expert.json / .png |
| Heatmap seeds × axis, ACT | pending | `eval/heatmap.py --policy act` | results/heatmap_act.json / .png |
| Instruction swap | 7/10 | `eval/instruction_swap.py` | results/instruction_swap.json |
| Camera vs oracle | pixels 84%, vlm 38% | `eval/camera_vs_oracle.py` | results/camera_vs_oracle.json |
| Planner: VLM-accepted / approved / tok/s | 4/8 · 100% · 44.6 | `eval/planner_eval.py` | results/planner_eval.json |
| Verifier injection | 20/20 caught, 6/6 sane passed | `verifier/inject_bad_plans.py` | results/verifier_injection.json + audit jsonl |
| Audit chain | chain intact | `make verify-log` | results/audit.jsonl |
| Voice test (4 samples) | mean WER 0.438, normalised 0.295, skills 100% | `eval/voice_test.py` | results/voice_test.json |
| Voice → arm latency | 10.447 s (seed 3, noisy.wav) | `runtime/demo.py --voice` | results/demo_seed3.json |
| Barge-in | 1 stop/resume, step completed: True | `runtime/demo.py --barge-in stop@3` | results/demo_bargein_stop.json |
| OpenVINO latency | fp32/CPU 49.13 ms, fp32/GPU 283.24 ms, fp16/CPU 49.51 ms, fp16/GPU 282.82 ms, int8/CPU 17.06 ms, int8/GPU 201.85 ms | `bench/run.py` | results/bench.json / bench.md |
| Precision preservation | fp32 0, fp16 0, int8 0 | `bench/preserve.py` | results/preserve.json |
| IR sizes / max Δ | open_drawer: fp32 132.7MB/fp16 67.1MB/int8 34.9MB, pick_place_fork: fp32 132.7MB/fp16 67.1MB/int8 34.9MB, pick_place_plate: fp32 132.7MB/fp16 67.1MB/int8 34.9MB, pick_place_mug: fp32 132.7MB/fp16 67.1MB/int8 34.9MB, handoff_spoon: fp32 132.7MB/fp16 67.1MB/int8 34.9MB, hold_mug: fp32 132.7MB/fp16 67.1MB/int8 34.9MB, pour: fp32 132.7MB/fp16 67.1MB/int8 34.9MB | `bench/export_ir.py`, `bench/quantize.py` | results/ir_export.json |
| Dataset | 420 episodes, 299233 frames | `expert/make_demos.py` | results/demos.json |
| Reach / handoff geometry | handoff [-0.010000000000000009, 0.0, 0.03] (via_table), 330 doubly-reachable cells | `souschef_env/reach.py` | results/reach_envelope.json |
| Jaw pads | 0.23° tilt, 15.8 mm gap | `souschef_env/build_scene.py` | results/jaw_pads.json |

Hardware for all of the above: Intel Core i7-13650HX + UHD iGPU (no NPU, not Core Ultra); RTX 3050 6 GB for ACT training only.
