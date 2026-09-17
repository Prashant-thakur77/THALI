# Phase log

One entry per phase: what was built, what was measured (and the `results/` file), what in the plan
was wrong or infeasible, and which rubric rows (plan §4) are satisfied with evidence vs still open.

---

## Phase 1 — Environment

**Built** (`souschef_env/`):
- `build_scene.py` composes two vendored SO-101s (TheRobotStudio MJCF, Apache-2.0, `assets/so101/`) via
  `MjSpec.attach` with prefixes `arm_a_`/`arm_b_`, plus table, cabinet with a sliding drawer (SentinelEdge's
  slide-joint pattern, tray open at the top so a top-down gripper reaches inside), 2 forks + 2 spoons on
  risers (VectorForge's 65 mm cutlery), a rimmed plate, a hollow mug, a hollow bottle with 20 free "water"
  spheres, cameras `overhead`/`wrist_a`/`wrist_b`, 10 table materials + 3 backdrops. Writes `assets/dinner_table.xml`
  and `assets/arms_only.xml` (IK model).
- `gripper.py`: VectorForge's three gripper findings, re-implemented: jaw pads laid on the measured flat inner
  faces of both jaws, jaw actuator turned into a torque source (OPEN +1.0 / GRIP −0.8 / HOLD −0.2 N·m), and
  prop-vs-jaw-mesh contacts excluded so grasps act on the pads, not the convex hull.
- `env.py` (`souschef_env/Thali-v0`): gymnasium env, 12-D action (5 joints + normalised jaw per arm),
  3-camera + `agent_pos` observation, jaw state machine (open → closing → holding, switched on the *moving*
  pad's contact force), seeded reset through `randomize.py`.
- `randomize.py`: 6 axes (placement, mass, friction, shape, lighting, background), `train_ranges`/`test_ranges`
  (test = 1.5× wider, +2 unseen table textures, +1 unseen mug shape), byte-identical for a given seed.
- `oracles.py`, `scene_description.py`, `ik.py` (mink, arms-only model), `reach.py`, `lerobot_plugin.py`.

**Measured** (all by scripts):
- `results/jaw_pads.json`: pad faces 15.8 mm apart at the reference angle, 0.23°/0.24° tilt, 0.3 mm residual
  (matches VectorForge's independent measurement of the same CAD).
- `results/reach_envelope.json`: FK sweep 143 325 configs/arm, max radius 0.479 m; position-reach overlap 1349
  cells (1 cm grid) at 3 cm grasp height; top-down-IK-feasible for *both* arms at 3 yaws: 330/522 tested cells
  (0.033 m²), bbox x ∈ [−0.10, 0.08], y ∈ [−0.12, 0.12]; handoff pose (−0.01, 0.00, 0.03), via-table.
- Render cost: 20 → 6 ms per 240×320 frame with shadows/reflections off (physics 5 ms per 20 ms control step).

**Plan deviations:**
- Arms at ±0.24 m, not ±0.35 m (plan 1.1): SO-101 top-down reach is ~0.25–0.30 m; ashish-doing hit the same wall.
- True SO-101 geometry used (plan offered it as an option); menagerie `so_arm100` kept in `assets/` for reference.
- "3 skyboxes" → 3 backdrop-wall materials (MuJoCo has one skybox texture per model).
- LeRobot registration via its plugin hook instead of editing `configs.py`/`factory.py` in site-packages.
- EGL unavailable → glfw (see BLOCKERS.md).
- The IK's lateral "tilt" is unreachable on a 5-DoF wrist; reach map uses pure top-down with ROT_TOL 20°.

**Tests:** 20 passing (`tests/test_scene.py`, `test_randomize.py`, `test_oracles.py`, `test_env.py`,
`test_lerobot_plugin.py`).

**Rubric status after Phase 1:**
| Row | Status |
|---|---|
| Task completion + bimanual | open — env ready; expert/skills come in Phase 2 |
| VLA / multi-modal | open |
| Robustness | partial — 6-axis randomiser + held-out split exist and are tested; no success numbers yet |
| OpenVINO | open |
| Reproducibility | partial — pinned env, tests, seeded reset; Makefile targets still stubs |
| Innovation | open |
| Speechmatics | open |

---

## Phase 2 — Scripted expert + demos

**Built** (`expert/`):
- `primitives.py`: mink-IK expert driving `ThaliEnv` through its 12-D action space (every recorded action is what a
  policy has to reproduce). Skills: `open_drawer`, `pick`/`place`/`pick_place`, via-table `handoff` (+`handoff_place`),
  `hold_mug` (side grasp), `pour` (side grasp, wrist roll in chunks with the measured spout-lip re-centred over the
  rim between chunks). Top-down grasps follow VectorForge: jaw aligned with the feature's narrow axis, site offset by
  half-width + 4 mm because the fixed jaw sits on the site, partial jaw opening sized to the object so the moving jaw
  does not sweep neighbours, sweep-blocked check against other objects and the drawer walls. `Workspace` implements
  the plan's shared-zone reservation. Carried frames are re-yawed to the arm's bearing (a 5-DoF wrist can't hold an
  off-bearing orientation, so commanding it produced real tilts). Skill-level retries (pick again from where the
  piece landed; other grasp frame for side grasps).
- `souschef_env/env.py` additions: continuous jaw command (0 close / 1 open / else hold angle), **grasp-assist weld**
  activated only after both pads physically load on an object (so a miss stays a miss) and released on open;
  `render_enabled` toggle. `oracles.held_by` counts the weld.
- `task.py`: the full 7-step program (drawer → fork (A) → spoon (A hands to B) → plate (A) → B holds mug → A pours → B
  sets mug down). `sweep.py`: expert success over 10 seeds per split. `instructions.py`: 10 paraphrases × 5 skills
  with arm/object/zone slots. `make_demos.py` + `merge_demos.py`: parallel shard recording into LeRobotDataset v3.0.

**Measured:**
- `results/expert_full_task_train.json`: **7/10** full-task successes on train seeds 0–9; per skill 1.0 except
  hold_mug 0.9, pour 0.8, place_mug 0.9.
- `results/expert_full_task_test.json`: **4/10** on the held-out split (pour 0.6 is the bottleneck; a fork place
  and a plate pick each fail once).
- `results/demos.json`: 420 episodes / 299 233 frames at 50 Hz, 3 cameras 240×320, 203 distinct instructions;
  expert rates while recording: 1.0 / 0.95 / 1.0 / 0.95 / 1.0 / 0.95 / **0.42 (pour, standalone start)**; 22
  recovery episodes (deliberate 2 cm miss + retry) among the pick_place skills.

**Plan deviations / infeasible:**
- 60 episodes per skill instead of 150–300: recording runs at ~20 frames/s per process on this laptop (glfw
  rendering, no EGL), so 420 episodes already took ~2 h on 4 parallel shards; 300/skill would be ~15 h.
- The dataset is not on the Hub yet (no `HF_TOKEN` on this machine) — `docs/KAGGLE_TODO.md` item 1.
- Pour is the weak skill (spheres need ≥ 92° tilt from a 30°-pitched roll axis; the exit lip swings ~5 cm during the
  roll). Left at 0.6–0.8 rather than sinking more time; the runtime's retry/fallback covers it.
- The mug is set down at the end by a separate `place_mug` step (not in the plan's 5-skill list) so the "mug in zone"
  sub-goal is reachable after the pour.

**Tests:** 9 new (`tests/test_expert.py`, `tests/test_demos.py`); full suite green.

**Rubric status after Phase 2:**
| Row | Status |
|---|---|
| Task completion + bimanual | evidence: expert 7/10 train, 4/10 test with via-table handoff + hold/pour (`results/expert_full_task_*.json`); policy numbers still open |
| VLA / multi-modal | open (dataset ready) |
| Robustness | partial — held-out split numbers exist for the expert |
| OpenVINO | open |
| Reproducibility | partial — `make demos` real; HF push pending token |
| Innovation | open |
| Speechmatics | open |

---

## Phase 5 — Verifier + audit

**Built** (`verifier/`, `planner/schema.json`):
- `planner/schema.json`: the plan contract shared by planner, verifier and runtime — `steps[{skill, arm, obj, zone, to_arm, amount}]`, `mode: normal|gentle`.
- `rules.py`: `Verifier.verify(plan, World) -> ALLOW | REORDER | BLOCK` with coded issues (SCHEMA, UNKNOWN_OBJECT, REACH, GRASP_PRECOND, WORKSPACE, POUR_PRECOND, ORDER, VELOCITY, SELF_HANDOFF, ZONE_MISMATCH). The plan is simulated step by step on a symbolic world built from the scene description, so preconditions are checked in the state each step will actually see. Reach uses the same mink IK bar as the expert. `ORDER` issues (cutlery before open_drawer, pour before hold_mug) are repaired by `_reorder` and returned as REORDER with the fixed plan; everything else is BLOCK. `check_velocity` is the per-step joint-velocity hook for the runtime (3 rad/s, halved in gentle mode).
- `audit.py`: JSON-lines log, each record hashed (sha256 over canonical JSON) and chained through `prev_hash` from a genesis of 64 zeros; `verify()` recomputes and names the first bad line. `make verify-log LOG=...`.
- `inject_bad_plans.py`: 20 unsafe/impossible plans + 6 sane plans (2 of them only valid after reorder) against seed 0's real scene state; every decision is appended to a hash-chained log.

**Measured:** `results/verifier_injection.json` — **20/20 unsafe plans caught, 6/6 sane plans passed**; the run's own audit log `results/verifier_injection_audit.jsonl` verifies intact.

**Plan deviations:** the verifier operates on the plan/world level; joint-velocity limits are enforced at runtime per command (`check_velocity`) rather than at plan time, because a plan carries no joint trajectory. Two of the 20 injected cases (unknown object/zone) are caught by the schema enum before the world check, which is the intended defence in depth.

**Tests:** 10 new in `tests/test_verifier.py` (rules, reorder, velocity/gentle, audit tamper detection, results consistency).

**Rubric status after Phase 5:** Innovation row now has evidence (20/20 + hash chain); the rest unchanged from Phase 2.

---

## Phase 4 — Local VLM planner

**Built** (`planner/`): `export.sh` (Qwen2-VL-2B-Instruct → OpenVINO IR, INT4 group-128, 1.8 GB, gitignored);
`plan.py` — `OpenVinoVLM` (openvino_genai `VLMPipeline`, stateless prompt, repetition penalty), ManipulaX-style
JSON repair + `normalise` (arm/object synonyms, defaults, de-duplication of looped steps), `Planner.plan`: VLM →
schema → verifier, one retry with the rejection reasons fed back, then the deterministic `rule_plan`;
`Planner.replan(frame, remaining_plan, failure_reason)`; `prompts/{system,user,replan}.txt` with a worked
example; `state_check.py` — per-sub-goal yes/no camera check (VLM, or a pixel heuristic when no model is loaded)
used by the runtime and by `eval/camera_vs_oracle.py`.

**Measured:** `results/planner_eval.json` (8 commands, seed-0 scene, CPU): **4/8 plans accepted straight from the
VLM, 4/8 from the rule fallback; 8/8 verifier-approved and covering every requested skill; 44.6 tok/s, 1.1 s
TTFT, mean 5.4 s per plan** including verification. On the iGPU: 9 tok/s, 24–55 s TTFT, crash on the second call.

**Plan deviations:** planner runs on **CPU**, not the iGPU (see BLOCKERS.md); the plan's "VLM on iGPU" claim is
replaced by "VLM on Intel CPU, iGPU unstable on this Raptor Lake box" in every table. Qwen3-VL-4B was not tried
(3.1 GB INT4; the 2B model already needs the rule fallback half the time, and latency doubles).

**Tests:** 10 (`tests/test_planner.py`, `tests/test_planner_results.py`): schema, repair/normalise, rule planner
coverage, no-model fallback, overhead projection, pixel heuristic vs oracle on a rendered frame, results file.

---

## Phase 6 — Voice

**Built** (`voice/`): `parser.py` — duet's ASR-tolerant parser ported to Python and extended with Hinglish and
Devanagari normalisation (Speechmatics' `hi` session returns Devanagari), homophone arm names ("arm eight/hey/
bee"), filler stripping, verb-onset clause splitting that tolerates verb-final Hindi, barge-in word detection,
diagnostics for uninterpretable clauses; `listen.py` — Speechmatics realtime (`enable_partials`, `diarization=
"speaker"`, `additional_vocab`, `end_of_utterance_silence_trigger=0.6`, dispatch on `EndOfUtterance`, wake-phrase
or first-speaker **speaker focus**, per-utterance latency, quota retry); `bargein.py` — partials → stop / other
arm / resume with speaker focus; `speak.py` — Speechmatics TTS with a phrase table ("Pouring now. Say stop
anytime.") and an on-disk cache; `eval/voice_test.py` — the four provided samples.

**Measured:** `results/voice_test.json` — normal.wav WER 0.00; tired.wav 0.44 (drops "with arm A"); hindi.wav raw
WER 1.00 vs the romanised reference but **0.43 after normalisation**, and "arm A se" was heard as "आराम से"
("gently") — a real ASR confusion, recorded as such; noisy.wav 0.31 with the background podcast diarised as
speaker S2 and **ignored by speaker focus**. Skill sequence recovered on **4/4** samples; first-partial → parsed
command 0.45–1.26 s. TTS: 2.5 s first synthesis, cached afterwards.

**Plan deviations:** the "SousChef, listen" wake phrase is `Thali, listen` (also accepts sous chef); when no wake
phrase is present the first speaker becomes the operator so the recorded samples exercise the focus rule.
Barge-in and the plan-ready → arm-moves latency are wired in Phase 7 (they need the runtime loop).

**Tests:** 8 (`tests/test_voice.py`) — offline parser cases (English, ASR noise, Hinglish, Devanagari, barge-in,
speaker focus) and the results file.

---

## Phase 7 — Runtime

**Built** (`runtime/`): `state_machine.py` — `Runtime.run_command`: IDLE → PLANNING (VLM/rules) → VERIFYING →
EXECUTING → CHECKING (sim oracle + camera-based yes/no) → next / REPLANNING (`Planner.replan` on the remaining
steps, re-verified) → DONE | BLOCKED, with PAUSED on a barge-in; every transition, plan, verdict and skill result
appended to the hash-chained audit log; latency chain speech-end → plan-ready → arm-moves recorded per command.
`arm_queues.py` — per-arm queues with dependencies (drawer before cutlery, hold before pour, receiver free
before a handoff), dispatching the idle arm's runnable step first. `demo.py` — `python -m runtime.demo --seed 3
--voice <wav>` (Speechmatics + speaker focus), `--command`, `--barge-in "stop@4"` / `"other arm@4"`, `--tts`.
`expert/primitives.py` gained an `interrupt` predicate polled every control step (`Interrupted`) and
`hold_still` for the paused state; a resumed step re-runs from the current world state.

**Measured:**
- `results/demo_seed3.json`: noisy.wav → operator S1, podcast speaker S2 ignored → parsed "open the top drawer,
  put the plate on the table with arm A" → VLM plan (CPU) ALLOWed → both skills succeed (oracle and camera agree)
  → **speech-end → plan-ready 7.5 s (VLM 6.0 s), plan-ready → arm-moves 3.0 s (first TTS synthesis), total 10.4 s**.
- `results/demo_bargein_stop.json`: injected "stop" partial 3 s into step 0 pauses the arm within one control
  step, "continue" resumes and the step completes; a failing pour triggers two replans before BLOCKED with the
  reason in the log.
- `results/demo_seed0.json`: "set the table" → 7-step plan → DONE, 6/6 sub-goals, 0.76 s speech-end → arm-moves
  with the rule planner.
- `results/audit.jsonl` verifies intact across all runs (`make verify-log LOG=results/audit.jsonl`).

**Plan deviations:** the two arms execute one skill at a time (the queues decide *which* arm goes next; they do
not move simultaneously). The pixel-heuristic camera check is used when no VLM is loaded; its agreement with the
oracle is measured in Phase 9. Latency numbers include TTS synthesis when `--tts` is on (cached afterwards).

**Tests:** 7 (`tests/test_runtime.py`): queue dependencies/dispatch, a full short command, barge-in with speaker
focus (S2's "stop" ignored, S1's honoured, resume), audit chain of the runs, demo result files.

---

## Phase 3 — Policies

**Built** (`policies/`, `runtime/executors.py`):
- Seven per-skill **ACT** baselines trained locally (`policies/train_act.sh`, `policies/act_<skill>.yaml`): ResNet-18,
  3 cameras 240×320 + 12-D state, chunk 50, batch 8, AMP, 8 000 steps, seed 1000 on the RTX 3050 6 GB — ~29 min per
  skill, final L1 losses 0.09–0.12 (`results/act_training.json`).
- **Multi-task SmolVLA**: `policies/kaggle_smolvla.ipynb` (streams `Prashant-77/thali_all`, fine-tunes
  `lerobot/smolvla_base`, batch 16, 20 k steps, pushes `Prashant-77/thali_smolvla`), `policies/smolvla_multitask.yaml`,
  step-by-step in `docs/KAGGLE_TODO.md`. **Not run here** (no HF token to push the dataset, no Kaggle access from this
  session); every SmolVLA row in the results is "pending SmolVLA run" and `eval/run_seeds.py --policy smolvla`
  fills them in the moment the Hub checkpoint exists.
- `runtime/executors.py`: `PolicyExecutor` — plan step → dataset skill → checkpoint (per-skill ACT, or the multi-task
  SmolVLA with the canonical instruction), closed-loop rollout with an early exit on the skill's oracle, then
  `policy_only` / `policy_retry` (second rollout from wherever the first left the scene) / `policy_fallback`
  (scripted expert). Which stage won each skill is recorded per seed.

**Measured** (test split, seeds 0–9, `results/seeds_act_*_test.json` → `results/seeds.json`): ACT **0/10** full task in
all three modes. Sub-goals policy-only: drawer 90 %, plate 40 %, mug 10 %, fork/spoon/pour 0 %; +retry: fork 20 %,
mug 20 %; +fallback: plate 50 %, fork 40 %, spoon 30 %, mug 40 %, pour 10 %. Fallback stays well below the expert
alone (5/10) because a failed policy attempt frequently leaves objects where the expert cannot recover them
(spoon out of the drawer tray, water spilled). `open_drawer` transfers reliably from 60 episodes; the rest do not.

**Plan deviations:** MiniLM language conditioning of ACT (VoiceSort's `text_embed.py`) was not implemented — each
per-skill dataset holds one instruction class, so the plan step (not the text) selects the skill; language selection
is exercised by the planner/instruction-swap path and would be the multi-task SmolVLA's job. Local SmolVLA training
was not attempted on the 6 GB card.

**Tests:** `tests/test_policies.py` (configs/notebook, step→skill mapping, checkpoint loads and acts).

---

## Phase 8 — OpenVINO bench

**Built** (`bench/`): `export_ir.py` — each ACT policy wrapped so it takes exactly the LeRobot-normalised inputs
(state + 3 cameras, batch 1, static shapes) → `ov.convert_model` → fp32 and fp16 IR, checked against PyTorch on a
real dataset frame (`results/ir_export.json`: max |Δ| ~1e-6 fp32, ~5e-4 fp16, 133 → 67 MB). `quantize.py` — NNCF
post-training INT8 with 300 real calibration frames per skill (MIXED preset, transformer model type; 35 MB, mean |Δ|
vs fp32 0.004–0.008 in normalised action units). `run.py` — p50/p95 latency and throughput per skill × precision ×
device with `PERFORMANCE_HINT=LATENCY`, lscpu model and OpenVINO device names captured, the VLM's tok/s and TTFT
pulled from `results/planner_eval.json`, every table captioned "measured on i7-13650HX CPU + UHD iGPU; same IR runs
on Core Ultra NPU with -d NPU and static shapes, not measured here." `preserve.py` — the 10 test seeds re-run with
the ACT policies executed through each IR (`OVActPolicy`, same pre/post-processors) in policy-only mode.
`export_smolvla.py` — SmolVLA vision encoder + connector → IR from `lerobot/smolvla_base` (frozen in fine-tuning,
so identical for the pending Kaggle checkpoint); the action expert is exported from the fine-tuned checkpoint when
it exists.

**Measured:** `results/bench.json` / `results/bench.md`, `results/preserve.json`, `results/ir_export.json`,
`results/smolvla_ir.json` — see README "OpenVINO" and docs/EVIDENCE.md for the rendered numbers.

**Plan deviations:** NPU rows are absent (no NPU on this machine) — static batch-1 shapes are exported so `-d NPU`
needs no re-export; the SmolVLA action expert IR waits for the Kaggle checkpoint; the VLM planner runs on CPU (iGPU
plugin crashes on the second generate, `docs/BLOCKERS.md`) and its iGPU single-call number (9 tok/s, 24–55 s TTFT)
is reported from that one measurement.

---

## Phase 9 — Eval suite

**Built** (`eval/`): `run_seeds.py` (full 7-step task on `test_ranges` seeds 0–9 through the runtime executor:
expert / ACT policy-only / +retry / +fallback / SmolVLA-when-present; per-skill outcome and which stage won; aggregate
`results/seeds.json` with explicit "pending SmolVLA run" rows), `heatmap.py` (seeds × axis, one axis randomised per
column plus the all-axes column, PNG + JSON), `instruction_swap.py` (arm / object / order swaps → confusion matrix),
`camera_vs_oracle.py` (pixel heuristic and VLM yes/no vs the sim oracle after every skill: agreement, precision,
recall, latency), `planner_eval.py`, `voice_test.py`. `make eval` runs them all.

**Measured:** rendered into README / docs/EVIDENCE.md from `results/seeds_*.json`, `heatmap_expert.json`,
`instruction_swap.json`, `camera_vs_oracle.json`.

**Plan deviations:** the seeds × axis heatmap is produced for the scripted expert only — for the ACT policies it
would be 70 full-task policy rollouts (~6 h on this laptop); the ACT rows exist for the all-axes column
(`seeds_act_*`). Four evaluation jobs were run in parallel on this machine, which doubled per-seed wall time
(recorded in each results file's `seconds`). A nominal-layout bug (bottle 9 cm from the held mug → pour grasp
collides with arm B) surfaced through the single-axis columns and was fixed before the final runs.

---

## Phase 10 — Packaging + presentation

**Built:** `Dockerfile` (python:3.11-slim + ffmpeg/EGL libs, pinned requirements, scene built at image build),
`.github/workflows/ci.yml` (offline tests + verifier injection + audit chain), `environment.yml`/`requirements.txt`,
`Makefile` with every target real, `docs/render_readme.py` rendering README.md, EVIDENCE.md, CHALLENGE_CHECKLIST.md,
MODEL_CARD.md and PITCH.md from `results/*.json` (the enforcement of "every claim backed by a file in results/"),
`hosting/app.py` (Gradio: replay of recorded runs, live parser → planner → verifier on the seed-0 scene, verifier
playground, evidence tab) with its own `hosting/requirements.txt`, `runtime/demo.py --video` for the presentation
montage, `docs/PITCH.md` (pitch, scale slide, shot list).

**Pending on the user:** HF token (dataset + checkpoints + IR push), Kaggle SmolVLA run, Space deployment, the
video recording itself, and the lablab submission (docs/KAGGLE_TODO.md).

## Phase 11 — beyond the baseline (started 16 Sep 2026, 23:00 IST)
With the demo video (4:23), deck and HF Space done, the goal became measurable leads over comparable projects; their READMEs
(TableMind, duet, so101-AI-Infra, AuraManip, PegBit, intel-bimanual-vla, ai-packing-assistant) were reviewed. Gaps they
lead on: NPU/Core Ultra numbers (hardware we do not have), per-skill learned-policy success (PegBit 19/20 mug within
1.5 cm), concurrent two-arm execution (duet), headline success on short scripted tasks. None uses Anomalib.

Done tonight: mid-task perturbation recovery (`eval/recovery.py`, 4/4), final-state verification in the runtime,
`eval/skill_eval.py` (per-skill policy-only, 20 held-out seeds, zone error in cm), Anomalib PatchCore table-state check
(`anomaly/`: dataset generator, trainer/OpenVINO export in `.venv-anomalib`, OpenVINO-only checker wired into the runtime
as `--anomaly`), README/EVIDENCE rows that render "pending" until each results file exists.
Running unattended: ACT retrain on 1050 episodes (12k steps/skill) → per-skill eval → full-task ACT evals → bench;
PatchCore train → score → results/anomaly.json.
Next: Qwen3-VL-4B planner comparison, SmolVLA on Kaggle (user), heatmap for ACT, HF pushes of new checkpoints/IR.

## 17 Sep — ACT retrain (1050 episodes) evaluated; bench contaminated
Full-task ACT on the 1050-episode checkpoints: policy-only 0/10, +retry 0/10, +expert fallback 0/10 (per-sub-goal with fallback:
drawer 70 %, plate 80 %, fork 40 %, spoon 60 %, mug 40 %, pour 0 %). `make bench` re-exported/quantised the new checkpoints but its
latency run overlapped a 50k-step training and an eval on the same machine (fp32/CPU 110 ms vs 49 ms idle), so
`results/bench.json` keeps the idle measurement from 16 Sep (same IR architecture and shapes); rerun `python -m bench.run` on an
idle machine after the 50k runs finish. Preservation at fp32/fp16/int8 is unchanged (Δ 0).
50k-step ACT per skill: plate 8/20 → 15/20 (median 1.85 cm) — training length, not episode count, was the bottleneck; the other
skills are training at 50k steps now.

## 17 Sep — full task 5/10 → 7/10 held-out (8/10 training)
Three expert fixes from per-seed failure diagnostics: (1) the pour aimed the spout 1.5 cm *above* the mug rim, so spheres
leaving the near-horizontal tube at ~0.5 m/s with a sideways component bounced off the rim — the lip now dips just inside the
opening, the roll is finer (16 chunks) and the tilt is held until the flow stops (up to 4 s, one extra 20° tip if nothing comes);
(2) `water_in_mug` no longer counts spheres that are inside the mug's cylinder but still inside the bottle tube (the dipped lip
made the old count stop the pour early); (3) `place` shakes a piece off the open jaw when it rides up (fork on seed 5), and the
plate pick retries two other rim points (plate at the table edge on seed 4). A scoring bug surfaced with the clear-table goals:
`run_seeds` required *every* subgoal key to be true, including fork_stowed / drawer_closed — it now scores `oracles.FULL_TASK`
and the saved rows were re-scored. Pour-amount after the fix: normal 5/5, full 5/5, little 4/5 within ±2 spheres.
Remaining misses: place_mug after the pour (3 seeds), pour (3), one fork and one plate pick.

## 17 Sep (later) — hold, pour and set-down fixes
Per-seed traces of the remaining full-task misses: the mug's base was held 3 cm up, 6 mm over a placed plate's rim 8 cm away
(`POUR_POSE` z → 5 cm); the side grasp 2 cm above the mug base let the mug tilt 15–20° in the jaw (grasp now at mid-body,
`MUG_GRASP_Z` 3.4 cm); water spheres that entered the mug bounced back out because a leftover line in `build_scene.py` reset their
contact `solref` from overdamped to critically damped right after setting it (removed — this changes the physics for future
demos; the 1050-episode dataset was recorded with the bouncier water, which only matters for the 150 pour episodes); the pour lip
now sits at the rim plane; the mug's handle could hook the opening jaw on release (the gripper now slides away from the handle
before rising); cutlery and the plate are nudged apart inside their zones. 32 fast tests pass; the 10-seed evals are re-running.
