# Model card — Thali policies and planner

## Per-skill ACT policies (`outputs_50k/act_<skill>`, 12k-step baselines in `outputs/`)
- Architecture: LeRobot ACT, ResNet-18 backbone, 3 cameras 240×320 + 12-D state, chunk 50, n_action_steps 50, VAE on.
- Data: `Prashant-77/thali_all` — 150 scripted-expert episodes per skill (1050 total,
  742837 frames, 50 Hz), train-split randomisation, deliberate-miss recovery episodes for the pick skills.
- Training: batch 8, AMP, seed 1000, RTX 3050 6 GB; 12 000 steps (`policies/train_act.sh`) and 50 000 steps (`STEPS=50000`, ~3.5 h/skill).
- Per-skill evaluation, policy only, 20 held-out seeds from task-consistent start states (`eval/skill_eval.py`): pick_place_plate **15/20** (median 1.85 cm from zone) · pick_place_fork **14/20** (median 1.32 cm from zone) · pick_place_mug **10/20** (median 3.92 cm from zone) · hold_mug **18/20** · handoff_spoon **1/20** (median 38.91 cm from zone) · pour **2/20**.
  Data vs steps: 12k steps, 60 episodes/skill: drawer 20/20 · plate 8/20 · fork 2/20 · hold 10/20 — 12k steps, 150 episodes/skill: drawer 20/20 · plate 8/20 · fork 6/20 · hold 10/20 — 2.5× the data barely moved it; 4× the steps did (row above).
- Full task through the runtime (`results/seeds_act_*_test.json`): policy-only 0/10, +retry 0/10, +expert fallback 0/10 — the handoff and pour policies are the open problems.
- OpenVINO: fp32 / fp16 / NNCF int8 IRs (`results/ir_export.json`), latency `results/bench.json`, preservation `results/preserve.json`.
- Intended use: skill execution inside the Thali runtime, with the verifier gating every plan and the expert as fallback. Not for real hardware without re-training on real data.

## Multi-task SmolVLA (`Prashant-77/thali_smolvla`)
- Base `lerobot/smolvla_base`, fine-tuned on all 1050 episodes with language conditioning (10 paraphrases per skill), batch 16, 20 000 steps; camera keys renamed overhead/wrist_a/wrist_b → camera1/2/3 at train and inference time.
- Per skill, policy only, 20 held-out seeds: best, step 14000: open_drawer **14/20** · pick_place_fork **0/20** (median 31.07 cm from zone) · pick_place_plate **2/20** (median 18.38 cm from zone) · pick_place_mug **0/20** (median 25.91 cm from zone) · handoff_spoon **0/20** (median 39.63 cm from zone) · hold_mug **8/20** · pour **0/20** — latest, step 20000: open_drawer **11/20** · pick_place_fork **0/20** (median 30.66 cm from zone) · pick_place_plate **0/20** (median 20.43 cm from zone) · pick_place_mug **0/20** (median 25.41 cm from zone) · handoff_spoon **0/20** (median 39.58 cm from zone) · hold_mug **3/20** · pour **0/20** (the last 6 000 steps ran at batch 4 with a fresh optimizer on a smaller GPU and lost ground; the step-14 000 checkpoint is kept on the Hub under `step_14000/`).
- Full task through the runtime (`results/seeds_smolvla_*_test.json`): policy-only 0/10, +retry pending, +expert fallback pending.
- Notebook: `policies/kaggle_smolvla.ipynb`; the best checkpoint is kept under `step_14000/` in the Hub repo.

## Planner: Qwen2-VL-2B-Instruct, OpenVINO INT4 (`planner/qwen2vl_int4`, not in git)
- Export: `optimum-cli export openvino --weight-format int4 --group-size 128` (`planner/export.sh`), 1.8 GB.
- Runs on CPU (44 tok/s); iGPU unstable on this driver. 4/8 plans accepted from the model, rest by rule fallback.
- Prompts in `planner/prompts/`; output validated against `planner/schema.json` and the verifier before anything moves.

## Data and safety notes
- All data is simulated (MuJoCo); no personal data. Voice samples in `voice/test_samples/` were provided with the task.
- The verifier blocks plans that pour without the mug held, reach across the table, or violate joint-velocity limits; gentle mode halves velocity limits when a person is at the table.
