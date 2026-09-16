# Thali — voice-controlled two-arm robot that sets the table and pours a drink

**For people who can talk but can't reach.** Two simulated SO-101 arms in MuJoCo open the drawer, lay out cutlery
and the plate, hold the mug steady and pour — from a spoken command, on the laptop already in the house.

> _Every number on this page is rendered from a file in [`results/`](results/) by `python -m docs.render_readme`.
> Nothing here is typed by hand._

**Hardware used for every measurement:** Dell G15 5530, Intel Core i7-13650HX (Raptor Lake) with its UHD iGPU,
OpenVINO devices `['CPU', 'GPU']`. **No NPU, not a Core Ultra.** ACT baselines trained on the laptop's RTX 3050 6 GB;
SmolVLA fine-tune on Kaggle (pending — see docs/KAGGLE_TODO.md).

## Architecture

```
mic ──► Speechmatics realtime ──► ASR-tolerant parser ──► local Qwen2-VL-2B INT4 (OpenVINO, CPU) ──► verifier ──► per-arm queues
        partials · end-of-turn      (English / Hinglish /    overhead frame + scene text → JSON plan     ALLOW/REORDER/BLOCK     (drawer before cutlery,
        speaker focus · vocab        Devanagari)              repair → retry → rule fallback            hash-chained audit log   hold before pour)
                                                                                                                                      │
        Speechmatics TTS ◄── "Pouring now. Say stop anytime." ◄── camera yes/no check + sim oracle ◄── skill (per-skill ACT → retry → scripted mink-IK expert) ◄──┘
        barge-in: "stop" / "other arm" on partials pauses the arm within one 20 ms control step
```

## Task completion (rubric: task completion + bimanual, 30)

Full command: _"Open the top drawer, pick up the plate with arm A, place it on the table, pick up the mug with arm B,
pour water into the mug with arm A"_ → 7 skills: open_drawer(A) · fork(A) · **spoon handed A→B via the table** · plate(A) ·
**B holds the mug while A pours** · B sets the mug down. Success = all six sub-goals true in the sim oracle.

| executor | test split (held-out, seeds 0–9) | train split |
|---|---|---|
| scripted expert (mink IK) | 5/10 | 7/10 |
| ACT, policy only | pending | |
| ACT + retry | pending | |
| ACT + retry + expert fallback | pending | |
| SmolVLA multi-task, policy only | pending SmolVLA run (docs/KAGGLE_TODO.md) | |
| SmolVLA + retry + fallback | pending SmolVLA run (docs/KAGGLE_TODO.md) | |

Per-sub-goal rates, per-seed rows and which stage won each skill: [`results/seeds.json`](results/seeds.json).
The runtime executes one skill at a time; the queues decide which arm goes next (both arms do not move simultaneously).

## VLA / multi-modal (20)

- **Local VLM planner**: Qwen2-VL-2B-Instruct exported to OpenVINO INT4 (`planner/export.sh`), fed the overhead frame + a
  structured scene description + the transcript. On 8 commands: **4/8** plans accepted
  straight from the VLM, 4/8 from the rule fallback after the verifier rejected the VLM's answer,
  **100% verifier-approved**, 44.6 tok/s and
  1130 ms time-to-first-token on the CPU ([`results/planner_eval.json`](results/planner_eval.json)).
- **Instruction swap** (same seed, arm / object / order swapped): **7/10** variants encoded correctly (7 plans from the VLM, the rest from the rule fallback).

arm swaps — rows: requested, columns: what the plan encoded

| | plate_arm_A | mug_arm_B | hold_B_pour_A | hold_A_pour_B |
|---|---|---|---|---|
| plate_arm_A | 1 | 0 | 0 | 0 |
| mug_arm_B | 0 | 1 | 0 | 0 |
| hold_B_pour_A | 0 | 0 | 1 | 0 |
| hold_A_pour_B | 0 | 0 | 1 | 0 |

object swaps — rows: requested, columns: what the plan encoded

| | fork | spoon | plate | mug |
|---|---|---|---|---|
| fork | 1 | 0 | 0 | 0 |
| spoon | 0 | 1 | 0 | 0 |
| plate | 0 | 0 | 1 | 0 |
| mug | 0 | 0 | 0 | 1 |

order swaps: plate_then_mug ✗, mug_then_plate ✗
- **Camera-based state check vs sim oracle** (4 seeds × 8 checkpoints × 6 questions):
  pixels: agreement 78%, precision 76%, recall 81%, 0.002 s/question
  ([`results/camera_vs_oracle.json`](results/camera_vs_oracle.json)).
- **Replan**: a failed skill re-enters the planner with the failure reason and the remaining steps (`Planner.replan`), re-verified; see `results/demo_bargein_stop.json`.
- **Training data**: 420 scripted-expert episodes / 299233 frames at 50 Hz, 3 cameras,
  10 paraphrases per skill, 22 deliberate-miss recovery episodes ([`results/demos.json`](results/demos.json)).

## Robustness (15)

Six randomisation axes (placement, mass, friction, shape, lighting, background) with a held-out **test split** (ranges 1.5×
wider, 2 unseen table textures, 1 unseen mug shape). Success per seed × axis (test split, each column randomises only that axis):

**scripted expert** ([`results/heatmap_expert.json`](results/heatmap_expert.json), hardest axis: shape)

| seed | placement | mass | friction | shape | lighting | background | all |
|---|---|---|---|---|---|---|---|
| 0 | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ |
| 1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 3 | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| 4 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 5 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 6 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 7 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 9 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| **rate** | 100% | 100% | 100% | 80% | 100% | 100% | 50% |

**ACT + retry + fallback** (pending)

pending

Policy-only vs +retry vs +fallback is the table above; the expert's own ceiling is 4/10 on test, 7/10 on train.

## OpenVINO (20)

_measured on i7-13650HX CPU + UHD iGPU; same IR runs on Core Ultra NPU with -d NPU and static shapes, not measured here._

| precision / device | mean p50 ms over skills | skills |
|---|---|---|
| fp32/CPU | 114.39 | 1 |
| fp32/GPU | 555.63 | 1 |
| fp16/CPU | 111.51 | 1 |
| fp16/GPU | 556.94 | 1 |
| int8/CPU | 36.95 | 1 |
| int8/GPU | 421.94 | 1 |

Per-skill rows in [results/bench.md](results/bench.md). Devices skipped: NPU.

**Does optimisation preserve success?** The 10 test seeds re-run with the ACT policies executed through each IR precision:

pending

Export: `bench/export_ir.py` (ACT → IR, static batch-1 shapes, fp32/fp16), `bench/quantize.py` (NNCF INT8, 300 real calibration
frames), sizes and max |Δ| vs PyTorch in [`results/ir_export.json`](results/ir_export.json). The VLM planner on the **iGPU** loads and answers
once (9 tok/s, 24–55 s TTFT) and then crashes in the GPU plugin on this driver, so it runs on the CPU here (docs/BLOCKERS.md).
**NPU:** none on this machine; the IRs use static shapes so `-d NPU` on a Core Ultra needs no re-export — not measured.

## Speechmatics

`voice/listen.py`: realtime, `enable_partials`, `end_of_utterance_silence_trigger=0.6`, `diarization="speaker"` with **speaker focus**
(the operator is whoever says _"Thali, listen"_ — or the first speaker; everyone else is logged and ignored), custom dictionary
(arm A / arm B / mug / fork / spoon / plate / drawer), `language="hi"` session for Hindi/Hinglish, TTS confirmations, barge-in on partials.

| sample | WER | normalised WER | skills recovered | speakers | ignored |
|---|---|---|---|---|---|
| normal.wav (en) | 0.00 | 0.00 | ✓ | S1 | — |
| tired.wav (en) | 0.44 | 0.44 | ✓ | S1 | — |
| hindi.wav (hi) | 1.00 | 0.43 | ✓ | S1 | — |
| noisy.wav (en) | 0.31 | 0.31 | ✓ | S1, S2 | And try things and listen to podcasts an… |

Skill sequence recovered on 100% of samples; first partial → parsed command
0.45–1.262 s.
hindi.wav: Speechmatics heard _"arm A se"_ as _"आराम से"_ ("gently") — kept as measured. End-to-end on seed 3 with the noisy sample
(podcast speaker S2 ignored): speech-end → plan-ready **7.478 s**, → arm-moves
**10.447 s** ([`results/demo_seed3.json`](results/demo_seed3.json)).
Barge-in: "stop" 3 s into a skill pauses within one control step, "continue" resumes ([`results/demo_bargein_stop.json`](results/demo_bargein_stop.json)).

## Verifier + audit (innovation)

Deterministic plan verifier (`verifier/rules.py`): reach (mink IK), grasp preconditions, shared-workspace reservation, drawer-before-cutlery,
pour-only-if-the-other-arm-holds-the-mug, joint-velocity limit with a gentle mode → ALLOW / REORDER (fixed plan returned) / BLOCK.
Injection: **20/20 unsafe plans caught**,
6/6 sane plans passed
([`results/verifier_injection.json`](results/verifier_injection.json)). Every plan, verdict and skill result is a sha256-chained JSON line;
`make verify-log` recomputes the chain ([`results/audit.jsonl`](results/audit.jsonl)).

## Reproduce

```bash
uv venv .venv --python 3.11 && uv pip install -r requirements.txt && uv pip install -e .   # or: docker build -t thali .
cp .env.example .env   # SPEECHMATICS_API_KEY, HF_TOKEN
make scene      # rebuild the MuJoCo scene + reach envelope       make demos     # 420 expert episodes -> LeRobot dataset (4 shards)
make train      # per-skill ACT on the local GPU                   make eval      # 10-seed tables, swap matrix, camera-vs-oracle, heatmap
make bench      # IR export, NNCF INT8, CPU/GPU latency, preservation
make demo VOICE=voice/test_samples/noisy.wav SEED=3   # voice -> plan -> verify -> arms, with TTS
make verify-log LOG=results/audit.jsonl              make test   # pytest
```
`planner/export.sh` exports the VLM (needs ~6 GB free, sets `TMPDIR` to disk). Kaggle SmolVLA: `docs/KAGGLE_TODO.md`.
Seeds: env `reset(seed)` is byte-identical per seed (`tests/test_randomize.py`); training seed 1000.

## Limitations

- Pour is the weakest skill (expert 60% on test): the water is 20 free spheres and the spout must tip past ~92°.
- 420 episodes (60/skill) rather than the planned 150–300: recording ran at ~20 frames/s on this laptop.
- One skill executes at a time; per-arm queues schedule, they do not run the arms concurrently.
- The 2B VLM needs the verifier + rule fallback for about half the commands; a 4B model was not tried.
- iGPU unstable for the VLM on this driver; no NPU to measure.

## Attribution

Env skeleton [huggingface/gym-aloha](https://github.com/huggingface/gym-aloha) (Apache-2.0) · SO-101 MJCF [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (Apache-2.0, `souschef_env/assets/so101/LICENSE`) · SO-ARM100 from [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) · IK [kevinzakka/mink](https://github.com/kevinzakka/mink) · training/eval [huggingface/lerobot](https://github.com/huggingface/lerobot) · gripper findings and prop geometry from VectorForge ([sadishihab/bimanual-vla](https://github.com/sadishihab/bimanual-vla)) · drawer pattern from SentinelEdge · scene composition pattern from inzuppato and ashish-doing · JSON repair from ManipulaX · parser design from duet (MIT) · verifier loop shape from TaskForge / RePlanTable / Sovereign · Speechmatics SDKs (MIT).
