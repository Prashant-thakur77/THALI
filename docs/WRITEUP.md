# Thali — project write-up (Intel Physical AI online track)

**Problem.** People with stroke, tremor, arthritis or age can speak but cannot lay a table or pour a drink. **Thali** is a voice-controlled bimanual robot (two SO-101 arms, MuJoCo) that sets the table and pours, running on an Intel laptop with OpenVINO.

Demo video: `video/thali_demo.mp4` (2:34). Repo: https://github.com/Prashant-thakur77/THALI. Dataset: https://huggingface.co/datasets/Prashant-77/thali_all.

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
| full 7-skill task, held-out seeds 0–9 | `eval/run_seeds.py --policy expert` | 5/10 (7/10 train) |
| ACT policy-only / +retry / +fallback | `eval/run_seeds.py --policy act` | 0/10 each (per-sub-goal rates in the file) |
| seeds × perturbation axis | `eval/heatmap.py` | 100 % on 5 single axes, shape 80 %, all six 50 % |
| planner | `eval/planner_eval.py` | 4/8 from the VLM, 100 % verifier-approved |
| instruction swap | `eval/instruction_swap.py` | 7/10 |
| camera vs oracle | `eval/camera_vs_oracle.py` | pixels 84 %, VLM 38 % |
| verifier injection | `verifier/inject_bad_plans.py` | 20/20 caught, 6/6 sane passed |
| OpenVINO latency / preservation | `bench/run.py`, `bench/preserve.py` | 49 → 17 ms; Δ 0 |
| voice | `eval/voice_test.py` | 4/4 skill sequences recovered |

Reproduce: `make test · scene · demos · train · eval · bench · demo · verify-log` (README "Reproduce"). 75 tests, CI workflow, Dockerfile.

## Honest limits

Learned policies are weak (60 episodes/skill at submission; a 1050-episode dataset is on the Hub and the multi-task SmolVLA fine-tune is queued); the pour is the hardest skill; one skill executes at a time; no NPU measured; the VLM planner runs on CPU on this driver.
