# Thali — CLAUDE.md

## Project (5 lines)
1. Thali: voice-controlled two-arm robot (2× simulated SO-101 in MuJoCo) that sets a dinner table and pours a drink for people who can speak but can't reach.
2. Submission to the Intel Bimanual VLA online track (lablab.ai AI Infra Summit) + the Speechmatics bonus. Full plan: [build-plan-souschef.md](build-plan-souschef.md); rubric in its section 4 is the acceptance criteria.
3. Pipeline: Speechmatics realtime STT (partials, end-of-turn, diarization/speaker focus, custom vocab, barge-in) → local Qwen2-VL-2B INT4 planner on OpenVINO iGPU → deterministic verifier + hash-chained audit log → one multi-task SmolVLA (ACT per-skill baseline, scripted mink-IK expert fallback) → camera-state check + sim oracle → replan.
4. Evaluated over 10 seeds × 6 perturbation axes with a held-out split; OpenVINO CPU/iGPU × FP32/FP16/INT8 bench that re-runs the seeds at each precision.
5. Hardware here: Dell G15 5530, Intel Core i7-13650HX (Raptor Lake, UHD iGPU), OpenVINO devices = `['CPU','GPU']`. **No NPU, not Core Ultra — say so plainly in README and every bench table; skip all NPU targets.** ACT trains locally on RTX 3050 6 GB (batch ≤ 8); SmolVLA fine-tunes on Kaggle via `policies/kaggle_smolvla.ipynb`, checkpoints synced through HF Hub (`HF_USER=Prashant-77`).

## Repo layout (plan §1)
```
souschef_env/     env.py · tasks.py · constants.py (from gym-aloha) · assets/{so_arm100.xml, scene.xml, meshes/, dinner_table.xml, textures/} · randomize.py · oracles.py · scene_description.py
expert/           mink IK primitives · make_demos.py
policies/         smolvla_multitask.yaml · act_<skill>.yaml · kaggle_smolvla.ipynb
planner/          export.sh · plan.py · prompts/ · schema.json
verifier/         rules.py · audit.py · inject_bad_plans.py
voice/            listen.py (speechmatics-rt) · speak.py (speechmatics-tts)
runtime/          state_machine.py · arm_queues.py · demo.py
bench/            export_ir.py · quantize.py · run.py · preserve.py
eval/             run_seeds.py · heatmap.py · instruction_swap.py · camera_vs_oracle.py
tests/            pytest (verifier, oracles, planner schema, parser)
docs/             EVIDENCE.md · CHALLENGE_CHECKLIST.md · MODEL_CARD.md
results/          *.json — the only source of truth for numbers
Makefile · Dockerfile · environment.yml · requirements.txt · README.md · TODO.md
```

## Rubric → directory (plan §4)
| Row | Pts | Satisfied by |
|---|---|---|
| Task completion + bimanual | 30 | `souschef_env/` + `expert/` + `runtime/` (drawer→cutlery→plate→mug→pour, via-table handoff, hold/pour) → `results/seeds.json` |
| VLA / multi-modal | 20 | `policies/` (multi-task SmolVLA) + `planner/` (local Qwen2-VL) + `eval/instruction_swap.py` + `eval/camera_vs_oracle.py` + replan in `runtime/` |
| Robustness | 15 | `souschef_env/randomize.py` (6 axes, train/test ranges) + `eval/heatmap.py` + policy-only/retry/fallback in `eval/run_seeds.py` |
| OpenVINO / Core Ultra | 20 | `bench/run.py` table + `bench/preserve.py` delta + VLM tok/s on iGPU (`planner/`); CPU+iGPU only, stated |
| Reproducibility | 10 | `Makefile`, `Dockerfile`, `environment.yml`/`requirements.txt`, HF Hub artefacts, `verify_stack.py` output in README, CI + `tests/` |
| Innovation | 5 | `verifier/` (20/20 caught + hash-chained audit, `make verify-log`) + speaker-focused barge-in in `voice/` |
| Speechmatics bonus | — | `voice/listen.py` (realtime, end-of-turn, diarization, custom vocab, barge-in, Hindi) + `voice/speak.py` (TTS) + latency number in `results/` |

## Conventions
- Python 3.11, type hints on every function signature, `pytest` in `tests/`.
- Env: `environment.yml` (conda) → `requirements.txt` holds the single set of pins. Locally: `uv venv .venv --python 3.11 && uv pip install -r requirements.txt`.
- Secrets in `.env` (gitignored), loaded with `python-dotenv`; `.env.example` lists the keys. Never hard-code keys or print them.
- Makefile targets: `demos train eval bench demo verify-log`.
- Seed everything (MuJoCo/numpy/torch) and commit the seeds. Eval on `test_ranges`, seeds 0–9.
- Commit after every phase, phase name in the message (e.g. `Phase 1 — Environment: ...`).
- Third-party code copied in (gym-aloha, mujoco_menagerie, mink examples, hackathon repos) keeps its licence file and is credited in README Attribution.

## The rule
**Every claim in README must be backed by a file in `results/`.** No number, percentage, latency or "N/N caught" appears in README, docs or the video that isn't produced by a script in this repo and written to `results/*.json`.
