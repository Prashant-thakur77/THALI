# Model card — Thali policies and planner

## Per-skill ACT baselines (`outputs/act_<skill>`)
- Architecture: LeRobot ACT, ResNet-18 backbone, 3 cameras 240×320 + 12-D state, chunk 50, n_action_steps 50, VAE on.
- Data: `Prashant-77/thali_all` — 60 scripted-expert episodes per skill (1050 total,
  742837 frames, 50 Hz), train-split randomisation, 10 % deliberate-miss recovery episodes for pick skills.
- Training: batch 8, AMP, 8 000 steps, seed 1000, RTX 3050 6 GB (~29 min/skill) — `policies/train_act.sh`, `policies/act_<skill>.yaml`.
- Evaluation: `results/seeds_act_*_test.json` (policy-only 0/10, +retry 0/10, +fallback 0/10 on the held-out split).
- OpenVINO: fp32 / fp16 / NNCF int8 IRs (`results/ir_export.json`), preservation in `results/preserve.json`.
- Intended use: skill execution inside the Thali runtime, with the verifier gating every plan and the expert as fallback. Not for real hardware without re-training on real data.

## Multi-task SmolVLA (`Prashant-77/thali_smolvla`)
- Base `lerobot/smolvla_base`, fine-tuned on all 420 episodes with language conditioning (203 distinct instructions), batch 16, 20 000 steps, Kaggle T4.
- Status: **pending** — notebook `policies/kaggle_smolvla.ipynb`, steps in `docs/KAGGLE_TODO.md`.

## Planner: Qwen2-VL-2B-Instruct, OpenVINO INT4 (`planner/qwen2vl_int4`, not in git)
- Export: `optimum-cli export openvino --weight-format int4 --group-size 128` (`planner/export.sh`), 1.8 GB.
- Runs on CPU (44 tok/s); iGPU unstable on this driver. 4/8 plans accepted from the model, rest by rule fallback.
- Prompts in `planner/prompts/`; output validated against `planner/schema.json` and the verifier before anything moves.

## Data and safety notes
- All data is simulated (MuJoCo); no personal data. Voice samples in `voice/test_samples/` were provided with the task.
- The verifier blocks plans that pour without the mug held, reach across the table, or violate joint-velocity limits; gentle mode halves velocity limits when a person is at the table.
