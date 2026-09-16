# Thali — voice-controlled two-arm robot that sets the table and pours a drink

**For people who can talk but can't reach.** Two simulated SO-101 arms in MuJoCo open the drawer, lay out cutlery
and the plate, hold the mug steady and pour — from a spoken command, on the laptop already in the house.

> _Every number on this page is rendered from a file in [`results/`](results/) by `python -m docs.render_readme`.
> Nothing here is typed by hand._

**Hardware used for every measurement:** Dell G15 5530, Intel Core i7-13650HX (Raptor Lake) with its UHD iGPU,
OpenVINO devices `['CPU', 'GPU']`. **No NPU, not a Core Ultra.** ACT baselines trained on the laptop's RTX 3050 6 GB;
SmolVLA fine-tune on Kaggle ({{ "checkpoint present" if load("seeds_smolvla_policy_fallback_test.json") else "pending — see docs/KAGGLE_TODO.md" }}).

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
| scripted expert (mink IK) | {{ frac(load("seeds_expert_expert_test.json")) }} | {{ frac(load("seeds_expert_expert_train.json")) }} |
| ACT, policy only | {{ seeds_row("act", "policy_only") }} | |
| ACT + retry | {{ seeds_row("act", "policy_retry") }} | |
| ACT + retry + expert fallback | {{ seeds_row("act", "policy_fallback") }} | |
| SmolVLA multi-task, policy only | {{ seeds_row("smolvla", "policy_only") }} | |
| SmolVLA + retry + fallback | {{ seeds_row("smolvla", "policy_fallback") }} | |

Per-sub-goal rates, per-seed rows and which stage won each skill: [`results/seeds.json`](results/seeds.json).
The runtime executes one skill at a time; the queues decide which arm goes next (both arms do not move simultaneously).

## VLA / multi-modal (20)

- **Local VLM planner**: Qwen2-VL-2B-Instruct exported to OpenVINO INT4 (`planner/export.sh`), fed the overhead frame + a
  structured scene description + the transcript. On 8 commands: **{{ load("planner_eval.json")["accepted_from_vlm"] }}/8** plans accepted
  straight from the VLM, {{ load("planner_eval.json")["accepted_from_rules"] }}/8 from the rule fallback after the verifier rejected the VLM's answer,
  **{{ pct(load("planner_eval.json")["verifier_approved_rate"]) }} verifier-approved**, {{ load("planner_eval.json")["vlm_tokens_per_s"] }} tok/s and
  {{ load("planner_eval.json")["vlm_ttft_ms"] }} ms time-to-first-token on the CPU ([`results/planner_eval.json`](results/planner_eval.json)).
- **Instruction swap** (same seed, arm / object / order swapped): {{ swap_table() }}
- **Camera-based state check vs sim oracle** ({{ load("camera_vs_oracle.json")["seeds"] if load("camera_vs_oracle.json") else "pending" }} seeds × 8 checkpoints × 6 questions):
  {{ ", ".join(f"{k}: agreement {pct(v['agreement'])}, precision {pct(v['precision'])}, recall {pct(v['recall'])}, {v['mean_latency_s']} s/question" for k, v in load("camera_vs_oracle.json")["backends"].items()) if load("camera_vs_oracle.json") else "pending" }}
  ([`results/camera_vs_oracle.json`](results/camera_vs_oracle.json)).
- **Replan**: a failed skill re-enters the planner with the failure reason and the remaining steps (`Planner.replan`), re-verified; see `results/demo_bargein_stop.json`.
- **Training data**: {{ load("demos.json")["total_episodes"] }} scripted-expert episodes / {{ load("demos.json")["total_frames"] }} frames at 50 Hz, 3 cameras,
  10 paraphrases per skill, {{ sum(v["recovery"] for v in load("demos.json")["skills"].values()) }} deliberate-miss recovery episodes ([`results/demos.json`](results/demos.json)).

## Robustness (15)

Six randomisation axes (placement, mass, friction, shape, lighting, background) with a held-out **test split** (ranges 1.5×
wider, 2 unseen table textures, 1 unseen mug shape). Success per seed × axis (test split, each column randomises only that axis):

**scripted expert** ([`results/heatmap_expert.json`](results/heatmap_expert.json), hardest axis: {{ load("heatmap_expert.json")["hardest_axis"] if load("heatmap_expert.json") else "pending" }})

{{ heat_table("expert") }}

**ACT + retry + fallback** {{ "" if load("heatmap_act.json") else "(pending)" }}

{{ heat_table("act") }}

Policy-only vs +retry vs +fallback is the table above; the expert's own ceiling is {{ frac(load("expert_full_task_test.json")) }} on test, {{ frac(load("expert_full_task_train.json")) }} on train.

## OpenVINO (20)

{{ bench_table() }}

**Does optimisation preserve success?** The 10 test seeds re-run with the ACT policies executed through each IR precision:

{{ preserve_table() }}

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
{{ "\n".join(f"| {r['file']} ({r['language']}) | {r['wer']:.2f} | {r['normalized_wer']:.2f} | {'✓' if r['skills_match'] else '✗'} | {', '.join(r['speakers'])} | {r['ignored_other_speaker'][0][:40] + '…' if r['ignored_other_speaker'] else '—'} |" for r in load("voice_test.json")["rows"]) }}

Skill sequence recovered on {{ pct(load("voice_test.json")["skill_match_rate"]) }} of samples; first partial → parsed command
{{ min(x for r in load("voice_test.json")["rows"] for x in r["partial_to_ready_s"]) }}–{{ max(x for r in load("voice_test.json")["rows"] for x in r["partial_to_ready_s"]) }} s.
hindi.wav: Speechmatics heard _"arm A se"_ as _"आराम से"_ ("gently") — kept as measured. End-to-end on seed 3 with the noisy sample
(podcast speaker S2 ignored): speech-end → plan-ready **{{ load("demo_seed3.json")["latency"]["speech_end_to_plan_ready_s"] }} s**, → arm-moves
**{{ load("demo_seed3.json")["latency"]["speech_end_to_arm_moves_s"] }} s** ([`results/demo_seed3.json`](results/demo_seed3.json)).
Barge-in: "stop" 3 s into a skill pauses within one control step, "continue" resumes ([`results/demo_bargein_stop.json`](results/demo_bargein_stop.json)).

## Verifier + audit (innovation)

Deterministic plan verifier (`verifier/rules.py`): reach (mink IK), grasp preconditions, shared-workspace reservation, drawer-before-cutlery,
pour-only-if-the-other-arm-holds-the-mug, joint-velocity limit with a gentle mode → ALLOW / REORDER (fixed plan returned) / BLOCK.
Injection: **{{ load("verifier_injection.json")["caught"] }}/{{ load("verifier_injection.json")["bad_plans"] }} unsafe plans caught**,
{{ load("verifier_injection.json")["good_passed"] }}/{{ load("verifier_injection.json")["good_plans"] }} sane plans passed
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
make verify-log LOG=results/audit.jsonl              make test   # {{ "pytest" }}
```
`planner/export.sh` exports the VLM (needs ~6 GB free, sets `TMPDIR` to disk). Kaggle SmolVLA: `docs/KAGGLE_TODO.md`.
Seeds: env `reset(seed)` is byte-identical per seed (`tests/test_randomize.py`); training seed 1000.

## Limitations

- Pour is the weakest skill (expert {{ pct(load("expert_full_task_test.json")["per_skill_rate"]["5_pour"]) }} on test): the water is 20 free spheres and the spout must tip past ~92°.
- {{ load("demos.json")["total_episodes"] }} episodes (60/skill) rather than the planned 150–300: recording ran at ~20 frames/s on this laptop.
- One skill executes at a time; per-arm queues schedule, they do not run the arms concurrently.
- The 2B VLM needs the verifier + rule fallback for about half the commands; a 4B model was not tried.
- iGPU unstable for the VLM on this driver; no NPU to measure.

## Attribution

Env skeleton [huggingface/gym-aloha](https://github.com/huggingface/gym-aloha) (Apache-2.0) · SO-101 MJCF [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (Apache-2.0, `souschef_env/assets/so101/LICENSE`) · SO-ARM100 from [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) · IK [kevinzakka/mink](https://github.com/kevinzakka/mink) · training/eval [huggingface/lerobot](https://github.com/huggingface/lerobot) · gripper findings and prop geometry from VectorForge ([sadishihab/bimanual-vla](https://github.com/sadishihab/bimanual-vla)) · drawer pattern from SentinelEdge · scene composition pattern from inzuppato and ashish-doing · JSON repair from ManipulaX · parser design from duet (MIT) · verifier loop shape from TaskForge / RePlanTable / Sovereign · Speechmatics SDKs (MIT).
