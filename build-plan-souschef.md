# Build plan v2 — "Thali" (formerly SousChef)

Target: Intel Bimanual VLA online track (1st, $3,000) + Best Use of Speechmatics ($500 + 1,000 credits). Time is not a constraint. All repos below were fetched and verified on Sep 15 2026.

---

## The pitch

**Thali — Voice-controlled two-arm robot that sets the table and pours drinks for people who can't use their hands.**

Same shape as the winners: *Ladybug — arm reads books aloud for kids with learning disabilities.* *Flip & Ship — two arms flip parcels and sort them.* One sentence, one clear person, one clear job.

Short description (lablab "short description" field):
> Thali listens to you and sets your table: two SO-101 arms open the drawer, lay out cutlery and plates, hold your mug and pour — for anyone with a stroke, Parkinson's, arthritis or age who can still speak but can't reach. Runs entirely on an Intel laptop with OpenVINO; voice by Speechmatics.

Long description, opening paragraph:
> Millions of people can talk perfectly well but can't set a table or pour a glass of water without help — stroke survivors, people with tremors, older adults living alone. Thali gives that back. Say what you want and two robot arms do it: one holds the mug steady while the other pours, one passes the plate to the side you can't reach. Say "stop" and it stops. It runs on the laptop already in the house, so nothing leaves the home.

**How each part of the build serves that person:** two arms = the steady hands they lost (hold + pour, handoff across the table) · realtime Speechmatics = "stop" works mid-motion, only the user's voice commands the robot, slurred and accented speech still works · local OpenVINO = private and offline · verifier = never pours without the mug held, never swings through the shared workspace · 10 seeds = real tables are messy.

**Scale slide:** ~1.3 billion people live with significant disability (WHO); ~1 in 4 adults will have a stroke, upper-limb weakness is one of the most common lasting effects. Existing feeding-assist robots are single-arm, pre-programmed, thousands of dollars; two SO-101s are ~$200 plus a laptop. Same build → elder care at home, hospital isolation wards.

**Build additions:** gentle mode (lower velocity limits when a person is seated), "pour a little", TTS "pouring now, say stop anytime", a short voice test on slurred/accented/Hindi samples with accuracy in the README, and the video opening on a person at a table before any code.

---

## 0. The project

Two simulated SO-101 arms in MuJoCo set a dinner table from spoken commands.

Speechmatics realtime STT (partials, end-of-turn, speaker diarization → operator-only "speaker focus", custom vocab, barge-in) → a **local** VLM on OpenVINO (Qwen2-VL-2B INT4 on iGPU) receives the overhead frame **plus a structured scene description** and emits a JSON skill plan with arm assignment → a **deterministic verifier** checks every step (shared-workspace reservation, grasp/pour/drawer preconditions, reachability, joint/velocity limits) and writes a hash-chained audit log → skills are executed by **one multi-task, language-conditioned SmolVLA** fine-tuned via LeRobot from scripted-IK demos (per-skill ACT as baseline, scripted expert as fallback) → after each skill a **camera-based state check** (VLM yes/no) and a sim oracle both confirm success or trigger replan → evaluated over 10 seeds × 6 perturbation axes with a held-out generalisation split → OpenVINO CPU/iGPU/NPU × FP32/FP16/INT8 benchmark that re-runs the 10 seeds at each precision to show optimisation preserved success.

**Bar to clear (from reading the field):** ManipulaX already has a local SmolVLM planner + real Core Ultra 7 155H CPU/iGPU/NPU numbers + hold-while-pour; PegBit's two-arm-table-setter has Qwen3-VL-4B on OpenVINO with a 30/30 claim (ACT only for the mug skill). Nobody has: a fine-tuned VLA doing the whole sequence with a 10-seed rate under all 6 axes, a precision-vs-success preservation table, a verifier with a caught-count, or realtime Speechmatics with diarization/speaker focus/barge-in. Those four are the win.

---

## 1. Base repo

**Fork `huggingface/gym-aloha`** → https://github.com/huggingface/gym-aloha (Apache-2.0). Bimanual MuJoCo gymnasium env, named-camera rendering, seedable reset, success plumbing, already registered in `lerobot-eval --env.type=aloha`. Swap the ViperX arms for SO-arm models, replace the cube task with the dinner scene.

Second-best base if you prefer starting from an SO-101 env: **VoiceSort's `env/bimanual_env.py`** (https://github.com/mukeshkbj/voicesort-so101, branch `master`) — already dual SO-101 in MuJoCo with 3 objects; you'd add drawer/plate/mug/bottle and LeRobot env registration.

Layout:

```
souschef/
├── souschef_env/     env.py · assets/{so_arm100.xml, dinner_table.xml, textures/} · randomize.py · oracles.py · scene_description.py
├── expert/           mink IK primitives · make_demos.py
├── policies/         smolvla_multitask.yaml · act_<skill>.yaml
├── planner/          export.sh · plan.py · prompts/ · schema.json
├── verifier/         rules.py · audit.py · inject_bad_plans.py
├── voice/            listen.py (speechmatics-rt) · speak.py (speechmatics-tts)
├── runtime/          state_machine.py · arm_queues.py · demo.py
├── bench/            export_ir.py · quantize.py · run.py · preserve.py
├── eval/             run_seeds.py · heatmap.py · instruction_swap.py · camera_vs_oracle.py
├── tests/            pytest (verifier, oracles, planner schema, parser)
├── docs/             EVIDENCE.md · CHALLENGE_CHECKLIST.md · MODEL_CARD.md
├── Makefile          demos / train / eval / bench / demo / verify-log
├── Dockerfile · environment.yml · results/*.json
└── README.md
```

---

## 2. Source map — what to take from where

### Open-source infra

| Need | From | Exactly what | License |
|---|---|---|---|
| Env skeleton | https://github.com/huggingface/gym-aloha | `gym_aloha/env.py`, `gym_aloha/tasks/sim.py`, `gym_aloha/constants.py` | Apache-2.0 |
| SO-arm MJCF | https://github.com/google-deepmind/mujoco_menagerie/tree/main/trs_so_arm100 | `so_arm100.xml`, `scene.xml`, meshes; instantiate twice with `arm_a_`/`arm_b_` prefixes. True SO-101 geometry: URDF+STL from https://github.com/TheRobotStudio/SO-ARM100 (`Simulation/SO101/`), or the vendored SO-101 in `so101_scene/` of https://github.com/ashish-doing/bimanual-vla-manipulation, or `assets/so101/` in https://github.com/Ahmadbey678/intel-bimanual-vla-hackathon | Apache-2.0 / check |
| IK | https://github.com/kevinzakka/mink | `examples/arm_aloha.py` (two `FrameTask`s) — set orientation weight low, the arm is 5-DoF | Apache-2.0 |
| Training / eval / dataset | https://github.com/huggingface/lerobot | `lerobot-train`, `lerobot-eval --seed=N --eval.n_episodes=10`, `LeRobotDataset` v3.0, `src/lerobot/policies/{smolvla,act}/`, env registration in `src/lerobot/envs/configs.py` + `factory.py` | Apache-2.0 |
| VLA checkpoint | https://huggingface.co/lerobot/smolvla_base | `--policy.path=lerobot/smolvla_base` | Apache-2.0 |
| Local VLM | https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/qwen2-vl | export cell (`optimum-cli export openvino --model Qwen/Qwen2-VL-2B-Instruct out/ --weight-format int4`) + `ov_genai.VLMPipeline(dir, "GPU")`. Fallback `notebooks/phi-3-vision/` | Apache-2.0 |
| IR export / INT8 | https://github.com/huggingface/optimum-intel · https://github.com/openvinotoolkit/nncf | `ov.convert_model(torch_module, example_input=...)`; `nncf.quantize(ov_model, nncf.Dataset(loader, fn))` with ~300 calib frames (`examples/post_training_quantization/openvino/mobilenet_v2/main.py` pattern) | Apache-2.0 |
| Benchmark | https://github.com/openvinotoolkit/openvino/tree/master/tools/benchmark_tool | `benchmark_app -m x.xml -d CPU|GPU|NPU -hint latency -t 30` | Apache-2.0 |
| Speechmatics | https://github.com/speechmatics/speechmatics-python-sdk (`speechmatics-rt`, `speechmatics-voice[smart]`, `speechmatics-tts`) | `sdk/rt/examples/` mic streaming; `TranscriptionConfig(enable_partials=True, diarization="speaker", additional_vocab=[...])`, `conversation_config.end_of_utterance_silence_trigger`, `END_OF_UTTERANCE`; docs: https://docs.speechmatics.com/speech-to-text/realtime/end-of-turn · …/realtime-diarization · …/features/custom-dictionary | MIT |
| Intel stack | https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/resources/hackathon_resources.html | `1_install_drivers.sh`, `2_install_software.sh`, `verify_stack.py` (OpenVINO 2026.3) | — |

### Repos from this exact challenge (take the component, rewrite it in your structure)

| Project | GitHub | Demo | Take this |
|---|---|---|---|
| VoiceSort | https://github.com/mukeshkbj/voicesort-so101 (`master`) · release: https://github.com/mukeshkbj/voicesort-so101/releases/tag/v1.0 | — | `env/bimanual_env.py` (dual SO-101 MuJoCo), `policy/text_embed.py` (MiniLM→ACT), `scripts/` IR export + CPU/iGPU bench, instruction-swap + paraphrase tests. Their FP16 drop (13→8/24) is why `bench/preserve.py` exists. |
| VectorForge | https://github.com/sadishihab/bimanual-vla | https://youtu.be/nHI5p-FtSo0 | `control/{ik,gripper,primitives,language}.py`: gripper collision-hull ≠ finger, tool-site offset, force control to hold grasp, reach/torque envelope, via-table handover. Read the README before Phase 2. |
| handoff | https://github.com/peacestate/aiinfra-vla (`master`) | https://webapp-chi-vert.vercel.app | SmolVLA-450M → IR → NNCF INT8 (Phase 8 as-is), 20-seed latency+success matrix, swap-test script, "Traps hit" list, Speechmatics + TTS layer. |
| SentinelEdge | https://github.com/shambhushekharsinha-engg/sentineledge-core | https://www.youtube.com/watch?v=ugT-6m7i8ls | Dockerfile, Makefile, conda env, `MODEL_CARD.md`, `CHALLENGE_CHECKLIST.md`, 39 pytest + CI, `scripts/evaluate.py` 10-seed HUD video, `simulation/scene.xml` drawer + FSM reward, Gradio `app.py`. |
| ManipulaX | https://github.com/etisamhaq/manipulax-ai | https://manipulax.vercel.app | SmolVLM-500M INT8 planner with schema validate/repair + rule fallback, hand-off FSM, hold-while-pour, Core Ultra 7 155H CPU/iGPU/NPU bench. The planner bar to clear. |
| PegBit two-arm-table-setter | https://github.com/PegBitStudio/two-arm-table-setter | https://pegbitstudio.github.io/two-arm-table-setter/ | `brain/{get_model,perception,planner}.py` (Qwen3-VL-4B INT4 on iGPU, depth-map perception, hand-off/set-down swap, look-again recovery), `train/` ACT→OpenVINO, `docs/core_ultra_guide.md`. |
| Talos | https://github.com/jannissio/talos-ai-infra (MIT) | https://huggingface.co/spaces/jannis-sms/talos-dinner-robotics | Six-skill dinner set incl. drawer/fork/spoon, table-supported bottle relay, `hosting/` Gradio Space, `docs/EVIDENCE.md`. |
| inzuppato | https://github.com/Ahmadbey678/intel-bimanual-vla-hackathon | Drive video | Cleanest layout: `sim/{build_scene,env,randomization,ik}.py`, `primitives/{pick,place,handoff}.py`, `perception/clip_ground.py` (CLIP→OpenVINO cross-check), `eval/run_seeds.py`, `scripts/record_video.py`. |
| RePlanTable | https://github.com/mubashir0x1-creator/replantable-ai- (trailing hyphen) | https://replantable-ai.streamlit.app | `planner/replanner.py`, `simulation/recovery_executor.py`, `state/world_state.py`, `vision/verifier.py`, `voice/speechmatics.py`. |
| TaskForge | https://github.com/builtbyrehan/taskforge-vla | — | `taskforge/planner` precondition validation + arm assignment + recovery demo. |
| VoiceVLA | https://github.com/advGKJha/VoiceVLA | — | 10-language Speechmatics layer, FastAPI+WS control API, "anti-hijack" operator-only idea. |
| Interruptible arm | https://github.com/jawad-glitch/ROBOT-ARM | Colab | `arm_control.py` interrupt → run new task → resume old. |
| duet | https://github.com/kmt9967/duet (MIT, TS) | https://duet-alpha-ebon.vercel.app | `src/lib/language` ASR-tolerant parser, `src/lib/planner` dependency graph, `src/lib/eval` 4×10 seeds. Port to Python; ignore the 2D sim. |
| ashish-doing | https://github.com/ashish-doing/bimanual-vla-manipulation | — | `so101_scene/build_dinner_scene.py`, `primitives_so101.py`, CONFIRM/DISPUTE verify loop, FastAPI dashboard. Assets/scene builder only. |
| Data-Laur | https://github.com/Data-Laur/ai-infra-summit-hack | — | `stage1_voice … stage7_eval` + `common/types.py` pydantic contracts. Structure only. |
| Winner patterns (other hackathons) | https://github.com/othnielObasi/sovereign-robotics-ops · https://github.com/NRdrgz/AMD_Robotics_Hackathon_2025_InverseKinematricks | Verifier allow/slow/stop/replan + audit schema; per-arm inference/execution queues. | check |
| Alt SO-100/101 envs | https://github.com/xuaner233/gym-so100 · https://github.com/lachlanhurst/so100-mujoco-sim · https://github.com/johnsutor/so101-nexus | Single-arm starting points. | — |

---

## 3. Build phases

### Phase 1 — Environment
1. Fork gym-aloha → `souschef_env`. `<include>` `so_arm100.xml` twice at `pos="-0.35 0 0"` / `pos="0.35 0 0"`, prefix joints/actuators `arm_a_`/`arm_b_`. If using true SO-101 geometry, take the vendored assets from ashish-doing or inzuppato.
2. `dinner_table.xml`: table, drawer (prismatic joint + handle site; copy the drawer body from SentinelEdge `simulation/scene.xml`), 2 spoons + 2 forks in drawer, plate, mug, bottle with **~20 free-joint spheres inside** (poured = ≥N spheres in mug volume). Cameras: `overhead`, `wrist_a`, `wrist_b`.
3. **Gripper first** (VectorForge's lesson): fix collision hull vs finger geometry, set tool-site offset, use force/torque-limited gripper so grasps hold. Measure reach envelope of both arms; **compute the handoff overlap region** by sampling reachable poses and intersecting; hard-code the handoff pose inside it. Default to via-table handover; in-air handover only if the overlap region allows.
4. `randomize.py` — seeded from `reset(seed=)`: placement xy+yaw · mass ×U(0.7,1.5) · friction ×U(0.6,1.4) · shape (3 variants per object + scale ×U(0.85,1.15)) · lighting pos/diffuse · background (8 table textures + 3 skyboxes). Keep `train_ranges` and `test_ranges` (test = ±1.5× train, +2 unseen textures, +1 unseen mug shape).
5. `oracles.py` — drawer qpos > 0.08; object-in-zone; mug-held = jaw-pad/mug contact; poured = sphere count. Full task = all sub-goals.
6. `scene_description.py` — dumps object names/poses/held-state as text for the planner.
7. Register env in LeRobot (`configs.py` + `factory.py`) so `lerobot-eval --env.type=souschef` works.

### Phase 2 — Scripted expert + demos
1. mink primitives: `open_drawer(arm)`, `pick_place(obj, arm, target)`, `handoff(obj, from, to)`, `hold_mug(arm)`, `pour(arm)`. Position task weight high, orientation low. Start from inzuppato `primitives/` and VectorForge `control/primitives.py`.
2. Shared-workspace rule in the expert: handoff zone is a reservation.
3. `expert/make_demos.py` → `LeRobotDataset` v3.0: 3 cams + joint state + action + instruction string. 150–300 episodes per skill, **10 paraphrases × arm swaps** per skill, **10% deliberate missed grasps with retry** (recovery demos). Push to HF Hub.

### Phase 3 — Policies
1. **One multi-task SmolVLA** on all skills (instruction selects behaviour):
   `lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=$HF_USER/souschef_all --batch_size=64 --steps=40000 --output_dir=outputs/smolvla_mt --policy.device=cuda`
2. Per-skill ACT baselines (`--policy.type=act`), language-conditioned via MiniLM as in VoiceSort `policy/text_embed.py`.
3. Runtime order: SmolVLA → oracle → on failure retry with expert. Report policy-only, policy+retry, policy+fallback.

### Phase 4 — Local VLM planner
1. `planner/export.sh`: `optimum-cli export openvino --model Qwen/Qwen2-VL-2B-Instruct planner/qwen2vl_int4 --weight-format int4`. Alternative: PegBit's Qwen3-VL-4B path if iGPU memory allows.
2. `planner/plan.py`: `VLMPipeline` on `GPU`; input = overhead frame + `scene_description` text + transcript; output strict JSON per `schema.json`; validate/repair (ManipulaX pattern), one retry, then rule-planner fallback.
3. `replan(frame, remaining_plan, failure_reason)` entry point.

### Phase 5 — Verifier + audit
1. `verifier/rules.py`: reachability (IK solvable in limits), grasp precondition, workspace reservation, drawer-open-before-cutlery, pour-only-if-mug-held-by-other-arm-under-spout, joint velocity limits → `ALLOW | REORDER | BLOCK` + reason. Borrow precondition list from TaskForge, loop shape from RePlanTable/Sovereign.
2. `verifier/audit.py`: JSON lines with `prev_hash`, `sha256`; `make verify-log` recomputes.
3. `verifier/inject_bad_plans.py`: 20 unsafe/impossible plans → "20/20 caught" table.

### Phase 6 — Voice
1. `voice/listen.py` (`speechmatics-rt`): 16 kHz PCM, `enable_partials=True`, `diarization="speaker"`, `additional_vocab` for arm A/B, mug, fork, spoon, plate, drawer; `end_of_utterance_silence_trigger=0.6`; dispatch on `END_OF_UTTERANCE`. Parser: port duet `src/lib/language` (ASR-tolerant).
2. Barge-in: partials matched against `stop | wait | other arm | no` → pause skill, replan (interrupt/resume from jawad-glitch `arm_control.py`).
3. Speaker focus: first speaker to say "SousChef, listen" becomes operator; other speaker labels logged + ignored, shown on HUD.
4. One Hindi/Hinglish command in the demo (`language="hi"` session or Melia batch clip).
5. `voice/speak.py` (`speechmatics-tts`) confirms each step.
6. Log speech-end → plan-ready → arm-moves latency; print it.

### Phase 7 — Runtime
`runtime/state_machine.py`: voice → planner → verifier → per-arm queues (`arm_queues.py`, Flip & Ship pattern) → skill → camera check + oracle → next/replan. `python -m souschef.runtime.demo --seed 3 --voice`.

### Phase 8 — OpenVINO bench
1. `bench/export_ir.py`: SmolVLA vision encoder + action expert, ACT → IR FP32/FP16; `bench/quantize.py` NNCF INT8 with 300 calib frames (handoff's `aiinfra-vla` pipeline).
2. **NPU needs static shapes**: `model.reshape(...)` the vision encoder + ACT for NPU; action expert stays on iGPU; state it in the table.
3. `bench/run.py --device CPU|GPU|NPU --precision fp32|fp16|int8` → p50/p95 latency, throughput, precision, device → markdown. Include VLM tokens/s on iGPU.
4. `bench/preserve.py`: 10-seed success at each precision → delta table.
5. Run on Core Ultra Series 2/3 if available; else Intel CPU+iGPU, stated. Print `verify_stack.py` + `lscpu` in README and video.

### Phase 9 — Eval suite
1. `make eval SEEDS=10`: full command ("Open the top drawer, pick up the plate with arm A, place it on the table, pick up the mug with arm B, pour water into the mug with arm A") on seeds 0–9, `test_ranges`; success %, per-skill %, seeds × axis heatmap (`eval/heatmap.py`).
2. `eval/instruction_swap.py`: same seed, swap arms/objects/order → confusion matrix (handoff + VoiceSort scripts as reference).
3. `eval/camera_vs_oracle.py`: VLM-from-image state judgments vs sim oracle agreement across the 10 seeds.
4. `make bench DEVICE=NPU`.
5. All outputs to `results/*.json` + `docs/EVIDENCE.md`. Seed MuJoCo/numpy/torch; commit seeds.

### Phase 10 — Packaging + presentation
1. Dockerfile, `environment.yml`, Makefile, `docs/CHALLENGE_CHECKLIST.md` (rubric → file/table), `docs/MODEL_CARD.md`, pytest for verifier/oracles/parser/schema, CI (SentinelEdge pattern).
2. Hosted demo: Gradio Space (Talos `hosting/`) or Streamlit with pre-recorded seeds + live planner.
3. Datasets + checkpoints + IR on HF Hub (VoiceSort-style release).
4. **Submit a draft early**; /live ranks by community vote. Update until deadline.
5. Business slide: hospitality/kitchen automation, assistive dining, why local inference.

---

## 4. Rubric → deliverable

| Row | Pts | Where it lives |
|---|---|---|
| Task completion + bimanual | 30 | Full drawer→cutlery→plate→mug→pour; via-table handoff + hold/pour; `results/seeds.json` |
| VLA / multi-modal | 20 | Multi-task SmolVLA + local Qwen2-VL planner + camera-state check + instruction-swap matrix + replan |
| Robustness | 15 | 6-axis randomiser, held-out test ranges, heatmap, policy-only vs retry vs fallback |
| OpenVINO / Core Ultra | 20 | `bench/run.py` table + `bench/preserve.py` delta + VLM on iGPU + NPU static-shape note |
| Reproducibility | 10 | Makefile targets, Docker, pinned env, HF Hub artefacts, `verify_stack.py` output, CI |
| Innovation | 5 | Verifier 20/20 + hash-chained audit; speaker-focused barge-in |
| Speechmatics bonus | — | realtime, end-of-turn, diarization/speaker focus, custom vocab, barge-in, Hindi, TTS loop, latency number |

## 5. Video (≤3 min)
0:00 pitch + architecture · 0:10 live command → partials → JSON plan → verifier ALLOW rows · 0:40 execution with HUD (skill, oracle ✓, camera ✓) · 1:30 barge-in "stop, use the other arm" + second speaker ignored + Hindi command · 2:00 10-seed 2×5 montage with counter → % · 2:30 bench table on Intel box, preservation table, 20/20 caught · 2:50 repo + HF links.

## 6. README
Pitch · Architecture · Models (SmolVLA multi-task, ACT baselines, Qwen2-VL-2B) · Bimanual coordination (queues + verifier + handoff geometry) · Training (demos, paraphrases, recovery demos) · Robustness (axes, held-out, heatmap) · Results (10-seed, swap matrix, camera-vs-oracle, policy/retry/fallback) · OpenVINO (export, NNCF, bench, preservation, hardware) · Speechmatics (features, latency) · Reproduce (make targets) · Limitations · Attribution.
