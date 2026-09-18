# Thali — project write-up (Intel Physical AI online track)

**Problem.** People with stroke, tremor, arthritis or age can speak but cannot lay a table or pour a drink. **Thali** is a voice-controlled bimanual robot (two SO-101 arms, MuJoCo) that sets the table and pours, running on an Intel laptop with OpenVINO.

Demo video: `video/thali_demo.mp4` (5:36). Repo: https://github.com/Prashant-thakur77/THALI. Dataset: https://huggingface.co/datasets/Prashant-77/thali_all.

## Architecture

![architecture](architecture.png)

```mermaid
flowchart LR
  M[Mic / WAV] --> S[Speechmatics realtime<br/>partials · end-of-turn 0.6 s<br/>speaker diarization → operator focus<br/>custom vocab · hi session]
  S --> P[Parser<br/>ASR-tolerant, Hinglish, Devanagari]
  P --> V[Qwen2-VL-2B INT4 · OpenVINO CPU<br/>overhead frame + scene text → JSON plan<br/>schema repair → retry → rule fallback]
  V --> R[Verifier<br/>reach (IK) · grasp preconditions · workspace reservation<br/>drawer-before-cutlery · hold-before-pour · velocity limits<br/>ALLOW / REORDER / BLOCK]
  R --> Q[Per-arm queues<br/>dependencies, idle arm first]
  Q --> K[Skill executor<br/>ACT / SmolVLA → retry → scripted mink-IK expert]
  K --> C[Camera yes/no check + sim oracle]
  C -->|next| Q
  C -->|fail| V
  R --> A[(sha256-chained audit log)]
  K --> A
  S -.barge-in on partials.-> K
  K --> T[Speechmatics TTS]
```

Closed loop per command: speech → plan → verify → execute skill → check → next / replan → speak. Barge-in ("stop", "other arm", "continue") is matched on partial transcripts and polled every 20 ms control step.

## Workload placement (what runs where, and why)

| Component | Device | Why |
|---|---|---|
| Qwen2-VL-2B planner (INT4 IR, openvino-genai `VLMPipeline`) | **Intel CPU** | 44.6 tok/s, 1.1 s TTFT. The Raptor Lake iGPU answers once (9 tok/s, 24–55 s TTFT) then faults in the GPU plugin — documented in `docs/BLOCKERS.md`. On a Core Ultra this is the iGPU/NPU workload. |
| ACT skill policies (fp32/fp16/INT8 IR, static batch-1 shapes) | **Intel CPU** (INT8), iGPU measured | 17 ms per 50-step action chunk INT8 vs 49 ms fp32; static shapes so `-d NPU` needs no re-export. |
| SmolVLA vision encoder (SigLIP + connector, IR) | CPU / iGPU measured | 417 ms CPU; frozen in fine-tuning, so the same IR serves the fine-tuned checkpoint. |
| Camera state check (pixel differencing; VLM optional) | CPU | 4 ms/question, 84 % agreement with the oracle; the 2B VLM is 38 % and 1.6 s — reported, not used by default. |
| Table-state anomaly check (Anomalib PatchCore → OpenVINO IR) | CPU / iGPU | learned "does the table look disturbed" signal after every skill on the overhead frame; scored on held-out layouts in `results/anomaly.json`. |
| Verifier, queues, audit log, parser | CPU | deterministic, microseconds. |
| MuJoCo simulation + rendering | CPU + display GL | 5 ms physics + 6 ms per 240×320 camera. |
| Speechmatics STT / TTS | cloud | the only off-device component; speech-end → arms moving 10.4 s end to end incl. the VLM plan and first TTS synthesis. |
| ACT training | NVIDIA RTX 3050 (laptop) | 8 000 steps × 7 skills, 29 min each. SmolVLA fine-tune: Kaggle T4 (notebook in `policies/`). |

Hardware: Dell G15 5530, Intel Core i7-13650HX + UHD iGPU, OpenVINO 2026.3, devices `['CPU','GPU']`. **No NPU.**

## Optimisation choices

- **INT4 weight-only** export of the VLM (`optimum-cli … --weight-format int4 --group-size 128`, 1.8 GB) — fits the laptop, 44.6 tok/s on CPU.
- **NNCF post-training INT8** of every ACT policy with 300 real, LeRobot-normalised calibration frames (MIXED preset, transformer model type): 133 → 35 MB, 2.9× faster on CPU, mean |Δ| 0.004–0.008 in normalised action units, **success identical at fp32 / fp16 / int8** on the 10 test seeds (`results/preserve.json`).
- **Static shapes everywhere** (batch 1, 3 × 3 × 240 × 320) so the same IR is NPU-ready.
- **Small prompt + scene text** instead of relying on the frame alone: the 2B VLM gets object coordinates as text, which is what keeps plans schema-valid; a **verifier-in-the-loop retry** turns rejected plans into corrected ones, with a rule planner as the deterministic floor.
- **Rendering**: shadows/reflections off (20 → 6 ms per camera); unrecorded prerequisite skills run unrendered; demo recording sharded 4-way.
- **Grasp physics** instead of teleport grasps: measured jaw pads on the SO-101 mesh, torque-controlled jaws, and a weld that engages only after both pads physically load — a miss stays a miss.

## Validation (all in `results/`, rendered into README.md by `docs/render_readme.py`)

| what | how | number |
|---|---|---|
| full 7-skill task, held-out seeds 0–9 | `eval/run_seeds.py --policy expert` | **9/10** (9/10 train) |
| clear the table (reverse task), held-out | `eval/clear_table.py` | 9/10 |
| follow-ups and corrections | `eval/followups.py` | 20/20 |
| target-volume pour | `eval/pour_amount.py` | 17/18 within ±2 spheres |
| ACT policy-only / +retry / +fallback | `eval/run_seeds.py --policy act` | 0/10 each (per-sub-goal rates in the file) |
| seeds × perturbation axis | `eval/heatmap.py` | 100 % on 5 single axes, shape 80 %, all six 50 % |
| planner | `eval/planner_eval.py` | 4/8 from the VLM, 100 % verifier-approved |
| instruction swap | `eval/instruction_swap.py` | 7/10 |
| camera vs oracle | `eval/camera_vs_oracle.py` | pixels 84 %, VLM 38 % |
| verifier injection | `verifier/inject_bad_plans.py` | 20/20 caught, 6/6 sane passed |
| OpenVINO latency / preservation | `bench/run.py`, `bench/preserve.py` | 49 → 17 ms; Δ 0 |
| voice | `eval/voice_test.py` | 4/4 skill sequences recovered |

Reproduce: `make test · scene · demos · train · eval · bench · demo · verify-log` (README "Reproduce"). 75 tests, CI workflow, Dockerfile.

## Phase 11 additions (17 Sep 2026)

| capability | evidence |
|---|---|
| **Mid-task recovery** — after the plate step passed its check the plate is knocked 10 cm off its zone; the final-state verification notices and redoes exactly that step | 4/4 detected, 4/4 recovered (`results/recovery.json`) |
| **Both arms at once** — independent single-arm steps run simultaneously through a step barrier in the expert (one simulator, two skill threads); a safety rule pairs only steps whose objects/targets are ≥ 15 cm apart and outside the shared zone | drawer+mug: 1065 → 591 sim steps (44 % fewer), plate+mug 30 % fewer, success unchanged (`results/concurrency.json`) |
| **Anomalib table-state check** — PatchCore on the difference between the live overhead frame and the reset reference, nominal set from real expert runs, exported to OpenVINO IR, wired into the loop as `--anomaly` | held-out image AUROC 0.943, 72/75 disturbances flagged, 13/57 false alarms; live: knocked plate flagged in 3/4 runs; 70.23 ms CPU (`results/anomaly_diffreal.json`, `results/recovery_anomaly.json`) |
| **Target-volume pour** — "a little" / normal / "fill it up" → 3 / 6 / 12 water spheres; the wrist roll stops per control step when the oracle counts the target | normal within ±2 on 4/5, little 2/5, full 2/5 (`results/pour_amount.json`) |
| **Per-skill learned-policy evaluation** — each ACT policy alone from task-consistent start states, 20 held-out seeds | 60 ep / 12k steps: drawer 20, plate 8, hold 10 /20 · 1050 ep / 12k steps: fork 6, plate 8 /20 · **50k steps: plate 15/20**, median 1.85 cm — training length, not episode count, was the bottleneck (`results/skill_eval_*.json`) |
| **SmolVLA fine-tune** | running on Kaggle in 12 h T4 sessions (~8 s/step); the step-5000 checkpoint is being evaluated |

Workload placement for the additions: the anomaly IR runs on the Intel CPU (70.23 ms/frame; the iGPU plugin fails on this driver, reported as such); the step barrier and queues are CPU threads; nothing new leaves the device.

## Honest limits

Learned policies still trail the scripted expert (ACT at 50k steps: plate 15/20, fork 16/20 alone; the multi-task SmolVLA is mid-training on Kaggle); handoff, hold and pour still execute one at a time (independent single-arm steps run concurrently); no NPU measured; the VLM planner runs on CPU on this driver.
