# Evidence index

Every number below is rendered from the named `results/*.json` file by `python -m docs.render_readme`. Regenerate after any run.

| claim | number | produced by | file |
|---|---|---|---|
| Expert full task, test split (10 seeds) | {{ frac(load("seeds_expert_expert_test.json")) }} | `eval/run_seeds.py --policy expert --split test` | results/seeds_expert_expert_test.json |
| Expert full task, train split | {{ frac(load("seeds_expert_expert_train.json")) }} | `eval/run_seeds.py --policy expert --split train` | results/seeds_expert_expert_train.json |
| ACT policy-only, test | {{ frac(load("seeds_act_policy_only_test.json")) }} | `eval/run_seeds.py --policy act --mode policy_only` | results/seeds_act_policy_only_test.json |
| ACT + retry, test | {{ frac(load("seeds_act_policy_retry_test.json")) }} | `eval/run_seeds.py --policy act --mode policy_retry` | results/seeds_act_policy_retry_test.json |
| ACT + retry + expert fallback, test | {{ frac(load("seeds_act_policy_fallback_test.json")) }} | `eval/run_seeds.py --policy act --mode policy_fallback` | results/seeds_act_policy_fallback_test.json |
| SmolVLA rows | {{ "present" if load("seeds_smolvla_policy_fallback_test.json") else "pending SmolVLA run (docs/KAGGLE_TODO.md)" }} | `eval/run_seeds.py --policy smolvla` | results/seeds.json |
| Heatmap seeds × axis, expert | hardest axis {{ load("heatmap_expert.json")["hardest_axis"] if load("heatmap_expert.json") else "pending" }} | `eval/heatmap.py --policy expert` | results/heatmap_expert.json / .png |
| Heatmap seeds × axis, ACT | {{ ("hardest axis " + load("heatmap_act.json")["hardest_axis"]) if load("heatmap_act.json") else "pending" }} | `eval/heatmap.py --policy act` | results/heatmap_act.json / .png |
| Per-skill ACT policy-only (60 ep) | {{ skill_row("act_60ep") }} | `python -m eval.skill_eval --act-root outputs_60ep --tag act_60ep` | `results/skill_eval_act_60ep.json` |
| Per-skill ACT policy-only (1050 ep) | {{ skill_row("act_1050ep") }} | `python -m eval.skill_eval --tag act_1050ep` | `results/skill_eval_act_1050ep.json` |
| Table-state anomaly check | {{ anomaly_row() }} | `python -m anomaly.make_data; .venv-anomalib/bin/python -m anomaly.train_patchcore; python -m anomaly.check --score` | `results/anomaly.json` |
| Mid-task perturbation recovery | {{ load("recovery.json")["recovered"] }}/{{ load("recovery.json")["total"] }} recovered ({{ load("recovery.json")["detected"] }} detected) | `python -m eval.recovery` | `results/recovery.json` |
| Instruction swap | {{ load("instruction_swap.json")["correct"] }}/{{ load("instruction_swap.json")["total"] }} | `eval/instruction_swap.py` | results/instruction_swap.json |
| Camera vs oracle | {{ ", ".join(f"{k} {pct(v['agreement'])}" for k, v in load("camera_vs_oracle.json")["backends"].items()) }} | `eval/camera_vs_oracle.py` | results/camera_vs_oracle.json |
| Planner: VLM-accepted / approved / tok/s | {{ load("planner_eval.json")["accepted_from_vlm"] }}/8 · {{ pct(load("planner_eval.json")["verifier_approved_rate"]) }} · {{ load("planner_eval.json")["vlm_tokens_per_s"] }} | `eval/planner_eval.py` | results/planner_eval.json |
| Verifier injection | {{ load("verifier_injection.json")["caught"] }}/{{ load("verifier_injection.json")["bad_plans"] }} caught, {{ load("verifier_injection.json")["good_passed"] }}/{{ load("verifier_injection.json")["good_plans"] }} sane passed | `verifier/inject_bad_plans.py` | results/verifier_injection.json + audit jsonl |
| Audit chain | {{ load("verifier_injection.json")["audit_chain"]["reason"] }} | `make verify-log` | results/audit.jsonl |
| Voice test (4 samples) | mean WER {{ load("voice_test.json")["mean_wer"] }}, normalised {{ load("voice_test.json")["mean_normalized_wer"] }}, skills {{ pct(load("voice_test.json")["skill_match_rate"]) }} | `eval/voice_test.py` | results/voice_test.json |
| Voice → arm latency | {{ load("demo_seed3.json")["latency"]["speech_end_to_arm_moves_s"] }} s (seed 3, noisy.wav) | `runtime/demo.py --voice` | results/demo_seed3.json |
| Barge-in | {{ len(load("demo_bargein_stop.json")["barge_ins"]) }} stop/resume, step completed: {{ load("demo_bargein_stop.json")["steps"][0]["oracle_ok"] }} | `runtime/demo.py --barge-in stop@3` | results/demo_bargein_stop.json |
| OpenVINO latency | {{ ", ".join(f"{k} {v['mean_p50_ms']} ms" for k, v in load("bench.json")["summary"].items()) if load("bench.json") else "pending" }} | `bench/run.py` | results/bench.json / bench.md |
| Precision preservation | {{ ", ".join(f"{k} {v['successes']}" for k, v in load("preserve.json")["table"].items()) if load("preserve.json") else "pending" }} | `bench/preserve.py` | results/preserve.json |
| IR sizes / max Δ | {{ ", ".join(f"{s}: " + "/".join(f"{p} {v['size_mb']}MB" for p, v in d["precisions"].items()) for s, d in load("ir_export.json")["skills"].items()) if load("ir_export.json") else "pending" }} | `bench/export_ir.py`, `bench/quantize.py` | results/ir_export.json |
| Dataset | {{ load("demos.json")["total_episodes"] }} episodes, {{ load("demos.json")["total_frames"] }} frames | `expert/make_demos.py` | results/demos.json |
| Reach / handoff geometry | handoff {{ load("reach_envelope.json")["handoff_pose_m"] }} ({{ load("reach_envelope.json")["handover_mode"] }}), {{ load("reach_envelope.json")["overlap"]["0.03"]["ik_feasible_cells"] }} doubly-reachable cells | `souschef_env/reach.py` | results/reach_envelope.json |
| Jaw pads | {{ round(load("jaw_pads.json")["arm_a_gripper_pad"]["tilt_deg"], 2) }}° tilt, {{ round(load("jaw_pads.json")["arm_a_moving_jaw_so101_v1_pad"]["face_m"] * 1000, 1) }} mm gap | `souschef_env/build_scene.py` | results/jaw_pads.json |

Hardware for all of the above: Intel Core i7-13650HX + UHD iGPU (no NPU, not Core Ultra); RTX 3050 6 GB for ACT training only.
