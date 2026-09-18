# Model card — Thali policies and planner

## Per-skill ACT policies (`outputs_50k/act_<skill>`, 12k-step baselines in `outputs/`)
- Architecture: LeRobot ACT, ResNet-18 backbone, 3 cameras 240×320 + 12-D state, chunk 50, n_action_steps 50, VAE on.
- Data: `Prashant-77/thali_all` — 150 scripted-expert episodes per skill ({{ load("demos.json")["total_episodes"] }} total,
  {{ load("demos.json")["total_frames"] }} frames, 50 Hz), train-split randomisation, deliberate-miss recovery episodes for the pick skills.
- Training: batch 8, AMP, seed 1000, RTX 3050 6 GB; 12 000 steps (`policies/train_act.sh`) and 50 000 steps (`STEPS=50000`, ~3.5 h/skill).
- Per-skill evaluation, policy only, 20 held-out seeds from task-consistent start states (`eval/skill_eval.py`): {{ skill_rows_50k() }}.
  Data vs steps: {{ act_ablation_row() }}.
- Full task through the runtime (`results/seeds_act_*_test.json`): policy-only {{ frac(load("seeds_act_policy_only_test.json")) }}, +retry {{ frac(load("seeds_act_policy_retry_test.json")) }}, +expert fallback {{ frac(load("seeds_act_policy_fallback_test.json")) }} — the handoff and pour policies are the open problems.
- OpenVINO: fp32 / fp16 / NNCF int8 IRs (`results/ir_export.json`), latency `results/bench.json`, preservation `results/preserve.json`.
- Intended use: skill execution inside the Thali runtime, with the verifier gating every plan and the expert as fallback. Not for real hardware without re-training on real data.

## Multi-task SmolVLA (`Prashant-77/thali_smolvla`)
- Base `lerobot/smolvla_base`, fine-tuned on all {{ load("demos.json")["total_episodes"] }} episodes with language conditioning (10 paraphrases per skill), batch 16, 20 000 steps; camera keys renamed overhead/wrist_a/wrist_b → camera1/2/3 at train and inference time.
- Per skill, policy only, 20 held-out seeds: {{ smolvla_row() }}.
- Full task through the runtime (`results/seeds_smolvla_*_test.json`): policy-only {{ frac(load("seeds_smolvla_policy_only_test.json")) }}, +retry {{ frac(load("seeds_smolvla_policy_retry_test.json")) }}, +expert fallback {{ frac(load("seeds_smolvla_policy_fallback_test.json")) }}.
- Notebook: `policies/kaggle_smolvla.ipynb`; the best checkpoint is kept under `step_14000/` in the Hub repo.

## Planner: Qwen2-VL-2B-Instruct, OpenVINO INT4 (`planner/qwen2vl_int4`, not in git)
- Export: `optimum-cli export openvino --weight-format int4 --group-size 128` (`planner/export.sh`), 1.8 GB.
- Runs on CPU (44 tok/s); iGPU unstable on this driver. {{ load("planner_eval.json")["accepted_from_vlm"] }}/8 plans accepted from the model, rest by rule fallback.
- Prompts in `planner/prompts/`; output validated against `planner/schema.json` and the verifier before anything moves.

## Data and safety notes
- All data is simulated (MuJoCo); no personal data. Voice samples in `voice/test_samples/` were provided with the task.
- The verifier blocks plans that pour without the mug held, reach across the table, or violate joint-velocity limits; gentle mode halves velocity limits when a person is at the table.
