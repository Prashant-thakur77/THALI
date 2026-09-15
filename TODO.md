# TODO — Thali build phases (from build-plan-souschef.md §3)

Hardware note for every phase: i7-13650HX, OpenVINO `['CPU','GPU']`, **no NPU** — all NPU sub-steps are marked skipped, not done.

## Phase 0 — Setup
- [x] CLAUDE.md, repo skeleton, gym-aloha env files, SO-ARM100 assets, environment.yml, Makefile stubs, TODO.md
- [x] Verify `import mujoco, gymnasium, mink, openvino` and `so_arm100.xml` loads

## Phase 1 — Environment
- [ ] 1.1 Fork gym-aloha → `souschef_env`; `<include>` `so_arm100.xml` twice at `pos="-0.35 0 0"` / `pos="0.35 0 0"`, prefix joints/actuators `arm_a_`/`arm_b_` (optionally true SO-101 geometry from ashish-doing / inzuppato)
- [ ] 1.2 `dinner_table.xml`: table, drawer (prismatic joint + handle site, from SentinelEdge `simulation/scene.xml`), 2 spoons + 2 forks in drawer, plate, mug, bottle with ~20 free-joint spheres; cameras `overhead`, `wrist_a`, `wrist_b`
- [ ] 1.3 Gripper first: collision hull vs finger geometry, tool-site offset, force/torque-limited gripper; measure both reach envelopes; compute handoff overlap region; hard-code handoff pose; via-table handover default
- [ ] 1.4 `randomize.py` seeded from `reset(seed=)`: placement xy+yaw · mass ×U(0.7,1.5) · friction ×U(0.6,1.4) · shape (3 variants + scale ×U(0.85,1.15)) · lighting · background (8 textures + 3 skyboxes); `train_ranges` / `test_ranges` (±1.5×, +2 unseen textures, +1 unseen mug)
- [ ] 1.5 `oracles.py`: drawer qpos > 0.08; object-in-zone; mug-held (jaw-pad/mug contact); poured (sphere count); full task = all sub-goals
- [ ] 1.6 `scene_description.py`: object names/poses/held-state as text for the planner
- [ ] 1.7 Register env in LeRobot (`configs.py` + `factory.py`) so `lerobot-eval --env.type=souschef` works

## Phase 2 — Scripted expert + demos
- [ ] 2.1 mink primitives: `open_drawer`, `pick_place`, `handoff`, `hold_mug`, `pour` (position weight high, orientation low; start from inzuppato `primitives/`, VectorForge `control/primitives.py`)
- [ ] 2.2 Shared-workspace rule in the expert: handoff zone is a reservation
- [ ] 2.3 `expert/make_demos.py` → LeRobotDataset v3.0 (3 cams + state + action + instruction); 150–300 eps/skill, 10 paraphrases × arm swaps, 10% missed-grasp-with-retry demos; push to HF Hub

## Phase 3 — Policies
- [ ] 3.1 One multi-task SmolVLA on all skills (Kaggle via `policies/kaggle_smolvla.ipynb`, checkpoints synced through HF Hub)
- [ ] 3.2 Per-skill ACT baselines locally on RTX 3050 (batch ≤ 8), language-conditioned via MiniLM (VoiceSort `policy/text_embed.py`)
- [ ] 3.3 Runtime order SmolVLA → oracle → retry with expert; report policy-only / policy+retry / policy+fallback

## Phase 4 — Local VLM planner
- [ ] 4.1 `planner/export.sh`: `optimum-cli export openvino --model Qwen/Qwen2-VL-2B-Instruct planner/qwen2vl_int4 --weight-format int4` (alt: Qwen3-VL-4B if iGPU memory allows)
- [ ] 4.2 `planner/plan.py`: `VLMPipeline` on `GPU`; input = overhead frame + scene description + transcript; strict JSON per `schema.json`; validate/repair, one retry, rule-planner fallback
- [ ] 4.3 `replan(frame, remaining_plan, failure_reason)` entry point

## Phase 5 — Verifier + audit
- [ ] 5.1 `verifier/rules.py`: reachability, grasp precondition, workspace reservation, drawer-open-before-cutlery, pour-only-if-mug-held-under-spout, joint velocity limits → `ALLOW | REORDER | BLOCK` + reason
- [ ] 5.2 `verifier/audit.py`: JSON lines with `prev_hash`, `sha256`; `make verify-log` recomputes
- [ ] 5.3 `verifier/inject_bad_plans.py`: 20 unsafe/impossible plans → "20/20 caught" table in `results/`

## Phase 6 — Voice
- [ ] 6.1 `voice/listen.py` (speechmatics-rt): 16 kHz PCM, `enable_partials`, `diarization="speaker"`, `additional_vocab`, `end_of_utterance_silence_trigger=0.6`, dispatch on `END_OF_UTTERANCE`; ASR-tolerant parser ported from duet `src/lib/language`
- [ ] 6.2 Barge-in: partials matched against `stop | wait | other arm | no` → pause skill, replan (interrupt/resume from jawad-glitch `arm_control.py`)
- [ ] 6.3 Speaker focus: first speaker to say "SousChef, listen" becomes operator; others logged + ignored, shown on HUD
- [ ] 6.4 One Hindi/Hinglish command in the demo (`language="hi"` session or batch clip)
- [ ] 6.5 `voice/speak.py` (speechmatics-tts) confirms each step
- [ ] 6.6 Log speech-end → plan-ready → arm-moves latency; write to `results/`

## Phase 7 — Runtime
- [ ] 7.1 `runtime/state_machine.py`: voice → planner → verifier → per-arm queues (`arm_queues.py`) → skill → camera check + oracle → next/replan; `python -m souschef.runtime.demo --seed 3 --voice`

## Phase 8 — OpenVINO bench
- [ ] 8.1 `bench/export_ir.py`: SmolVLA vision encoder + action expert, ACT → IR FP32/FP16; `bench/quantize.py` NNCF INT8 with 300 calib frames
- [ ] ~~8.2 NPU static shapes (`model.reshape`)~~ — **skipped: no NPU on this box**; state it in the table
- [ ] 8.3 `bench/run.py --device CPU|GPU --precision fp32|fp16|int8` → p50/p95 latency, throughput → markdown; include VLM tokens/s on iGPU
- [ ] 8.4 `bench/preserve.py`: 10-seed success at each precision → delta table
- [ ] 8.5 Run on Intel CPU+iGPU (i7-13650HX), stated plainly; print `verify_stack.py` + `lscpu` in README and video

## Phase 9 — Eval suite
- [ ] 9.1 `make eval SEEDS=10`: full command on seeds 0–9, `test_ranges`; success %, per-skill %, seeds × axis heatmap (`eval/heatmap.py`)
- [ ] 9.2 `eval/instruction_swap.py`: swap arms/objects/order → confusion matrix
- [ ] 9.3 `eval/camera_vs_oracle.py`: VLM image judgments vs sim oracle agreement across 10 seeds
- [ ] ~~9.4 `make bench DEVICE=NPU`~~ — **skipped: no NPU**; run `DEVICE=CPU` and `DEVICE=GPU` instead
- [ ] 9.5 All outputs to `results/*.json` + `docs/EVIDENCE.md`; seed MuJoCo/numpy/torch; commit seeds

## Phase 10 — Packaging + presentation
- [ ] 10.1 Dockerfile, `environment.yml`, Makefile, `docs/CHALLENGE_CHECKLIST.md`, `docs/MODEL_CARD.md`, pytest for verifier/oracles/parser/schema, CI
- [ ] 10.2 Hosted demo: Gradio Space or Streamlit with pre-recorded seeds + live planner
- [ ] 10.3 Datasets + checkpoints + IR on HF Hub (`Prashant-77`)
- [ ] 10.4 Submit a draft early; update until deadline
- [ ] 10.5 Business slide: hospitality/kitchen automation, assistive dining, why local inference
