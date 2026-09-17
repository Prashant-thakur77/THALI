<div align="center">

# Thali

### Say what you want. Two robot arms set your table and pour your drink.

**Voice-controlled bimanual table setting for people who can speak but can't reach — running entirely on an Intel laptop.**

Speechmatics realtime speech · local Qwen2-VL planner on OpenVINO · deterministic safety verifier with a tamper-evident audit log · learned SmolVLA / ACT skills with a scripted-IK fallback · two SO-101 arms in MuJoCo

[Results](#results-at-a-glance) · [How it works](#how-it-works) · [Reproduce](#reproduce) · [Evidence index](docs/EVIDENCE.md) · [Rubric checklist](docs/CHALLENGE_CHECKLIST.md)

</div>

---

## Why

Millions of people can talk perfectly well but can't lay a table or pour a glass of water without help — stroke survivors, people with tremors, older adults living alone. Thali gives that back: say *"set the table and pour me some water"* and two arms open the drawer, lay out the cutlery and plate, one holds the mug steady while the other pours. Say **"stop"** mid-motion and it stops. Only the person who said *"Thali, listen"* is obeyed. Nothing leaves the house: speech understanding aside, every model runs on the laptop's Intel CPU/iGPU through OpenVINO.

## Results at a glance

> Every number on this page is rendered from a JSON file in [`results/`](results/) by `python -m docs.render_readme`. Nothing is typed by hand; [docs/EVIDENCE.md](docs/EVIDENCE.md) maps each claim to its file and the script that produced it.

| | measured | evidence |
|---|---|---|
| **Full task, scripted expert** (drawer → fork → spoon handed A→B → plate → hold + pour → mug) | **{{ frac(load("seeds_expert_expert_test.json")) }}** held-out seeds · {{ frac(load("seeds_expert_expert_train.json")) }} train | `results/seeds_expert_*` |
| **Full task, ACT policies** (policy-only / +retry / +expert fallback) | {{ frac(load("seeds_act_policy_only_test.json")) }} / {{ frac(load("seeds_act_policy_retry_test.json")) }} / {{ frac(load("seeds_act_policy_fallback_test.json")) }} | `results/seeds_act_*` |
| **Full task, multi-task SmolVLA** | {{ frac(load("seeds_smolvla_policy_fallback_test.json")) if load("seeds_smolvla_policy_fallback_test.json") else "training on Kaggle — pending" }} | `results/seeds.json` |
| **Robustness**, one perturbation axis at a time (10 seeds each) | placement {{ pct(load("heatmap_expert.json")["per_axis_success_rate"]["placement"]) }} · mass {{ pct(load("heatmap_expert.json")["per_axis_success_rate"]["mass"]) }} · friction {{ pct(load("heatmap_expert.json")["per_axis_success_rate"]["friction"]) }} · shape {{ pct(load("heatmap_expert.json")["per_axis_success_rate"]["shape"]) }} · lighting {{ pct(load("heatmap_expert.json")["per_axis_success_rate"]["lighting"]) }} · background {{ pct(load("heatmap_expert.json")["per_axis_success_rate"]["background"]) }} · all six {{ pct(load("heatmap_expert.json")["per_axis_success_rate"]["all"]) }} | `results/heatmap_expert.json` |
| **Local VLM planner** (Qwen2-VL-2B, INT4, OpenVINO CPU) | {{ load("planner_eval.json")["accepted_from_vlm"] }}/8 plans straight from the model, **{{ pct(load("planner_eval.json")["verifier_approved_rate"]) }} verifier-approved**, {{ load("planner_eval.json")["vlm_tokens_per_s"] }} tok/s, {{ load("planner_eval.json")["vlm_ttft_ms"] }} ms to first token | `results/planner_eval.json` |
| **Per-skill ACT, policy only** (20 held-out seeds each, from task-consistent start states; 60-episode checkpoints) | {{ skill_row("act_60ep") }} | `results/skill_eval_act_60ep.json` |
| **Per-skill ACT, policy only — retrained on 1050 episodes** | {{ skill_row("act_1050ep") }} | `results/skill_eval_act_1050ep.json` |
| **Per-skill ACT, policy only — 50k training steps** (same 150 episodes; the 12k-step rows above are the baseline) | {{ skill_rows_50k() }} | `results/skill_eval_act_plate_50k.json`, `results/skill_eval_act_50k.json` |
| **Per-skill SmolVLA, policy only** (multi-task, language-conditioned; Kaggle T4 fine-tune in 4 500-step sessions — this row is the **step-5 000 checkpoint of 20 000**, re-evaluated as sessions land) | {{ skill_row("smolvla_5k") }} | `results/skill_eval_smolvla_5k.json` |
| **Table-state anomaly check** (Anomalib PatchCore → OpenVINO IR, overhead camera, held-out layouts) | {{ anomaly_row() }} | `results/anomaly.json` |
| **Clear the table** (reverse task on a set table: fork back to the drawer, spoon handed B→A and back, drawer closed — two new skills `put_in_drawer` / `close_drawer` through planner, verifier, queues and expert) | {{ clear_row() }} | `results/clear_table.json` |
| **Target-volume pour** ("a little" / normal / "fill it up" → 3 / 6 / 12 water spheres; the roll stops when the oracle counts the target) | {{ pour_amount_row() }} | `results/pour_amount.json` |
| **Both arms at once** (independent steps driven through the expert's step barrier; same seeds, same commands) | {{ concurrency_row() }} | `results/concurrency.json` |
| **Mid-task perturbation recovery** (plate knocked 10 cm off its zone after its step passed; final-state check must notice and redo it) | {{ load("recovery.json")["detected"] }}/{{ load("recovery.json")["total"] }} detected · {{ load("recovery.json")["recovered"] }}/{{ load("recovery.json")["total"] }} recovered | `results/recovery.json` |
| **Instruction swap** (arm / object / order) | {{ load("instruction_swap.json")["correct"] }}/{{ load("instruction_swap.json")["total"] }} encoded correctly | `results/instruction_swap.json` |
| **Camera state check vs sim oracle** | {{ pct(load("camera_vs_oracle.json")["backends"]["pixels"]["agreement"]) }} agreement (pixels) · {{ pct(load("camera_vs_oracle.json")["backends"]["vlm"]["agreement"]) }} (2B VLM) | `results/camera_vs_oracle.json` |
| **Safety verifier** | **{{ load("verifier_injection.json")["caught"] }}/{{ load("verifier_injection.json")["bad_plans"] }} unsafe plans blocked**, {{ load("verifier_injection.json")["good_passed"] }}/{{ load("verifier_injection.json")["good_plans"] }} sane plans passed, audit chain verified | `results/verifier_injection.json` |
| **OpenVINO** ACT policy call, CPU | fp32 {{ load("bench.json")["summary"]["fp32/CPU"]["mean_p50_ms"] }} ms → **INT8 {{ load("bench.json")["summary"]["int8/CPU"]["mean_p50_ms"] }} ms** p50; success identical at every precision | `results/bench.json`, `results/preserve.json` |
| **Voice** (4 samples: clear, tired, Hindi, noisy room) | skill sequence recovered on **{{ pct(load("voice_test.json")["skill_match_rate"]) }}**; background speaker ignored; speech-end → arms moving **{{ load("demo_seed3.json")["latency"]["speech_end_to_arm_moves_s"] }} s** | `results/voice_test.json`, `results/demo_seed3.json` |
| **Barge-in** | "stop" pauses within one 20 ms control step, "continue" resumes | `results/demo_bargein_stop.json` |

**Hardware for every number:** Dell G15 5530 — Intel Core i7-13650HX (Raptor Lake) + UHD iGPU, OpenVINO `['CPU','GPU']`. **No NPU; not a Core Ultra.** ACT trained on the laptop's RTX 3050 6 GB; SmolVLA fine-tuned on Kaggle.

## Try it live

**https://huggingface.co/spaces/Prashant-77/thali → "Open Thali Live"** — the whole loop in a browser: type a command or pick a Speechmatics voice sample, watch both arms execute it in the simulator (front + overhead cameras streamed live), and see the plan, the verifier's verdict, each skill's oracle / camera / PatchCore checks and the sha256-chained audit records as they happen. "Stop" and "Continue" inject barge-in partials exactly as the microphone path does; "clear the table" runs as a follow-up on the set table. The interactive site is served from the lab machine (Intel i7-13650HX, the same machine every number was measured on) through a Cloudflare tunnel, so it includes the OpenVINO Qwen2-VL planner and the PatchCore check; one simulator is shared by every visitor and a run takes 1–4 minutes. If the machine is off the button will not answer — the video on the landing page shows the same loop. Run it yourself: `web/serve.sh` (site + tunnel), or `uvicorn web.server:app --port 7860`, or `docker build -f Dockerfile.web`.

## How it works

![architecture](docs/architecture.png)

Full write-up (architecture, workload placement, optimisation choices): [docs/WRITEUP.md](docs/WRITEUP.md). Demo video: `video/thali_demo.mp4` (4:23).

```
 mic ─► Speechmatics realtime ─► parser ─► local VLM planner ─► verifier ─► per-arm queues ─► skill ─► camera check + oracle ─► next / replan
        partials, end-of-turn      English    Qwen2-VL-2B INT4      ALLOW        drawer before    ACT / SmolVLA      "is the drawer open?"
        speaker diarization        Hinglish   overhead frame +      REORDER      cutlery, hold    then retry, then    yes/no on the frame,
        custom dictionary          Devanagari scene text → JSON     BLOCK        before pour      scripted IK expert  sim oracle as truth
                 ▲                                                    │
                 └── barge-in on partials: "stop" / "other arm" pauses the arm mid-motion; "continue" resumes
                     Speechmatics TTS confirms each step: "Pouring now. Say stop anytime."
```

**The task.** *"Open the top drawer, pick up the plate with arm A, place it on the table, pick up the mug with arm B, pour water into the mug with arm A"* → seven skills: open drawer (A) · fork to the left of the plate (A) · **spoon handed from A to B via the table** (the spoon's spot is out of A's reach) · plate (A) · **B lifts and holds the mug while A pours** · B sets the mug down. Success = all six sub-goals true in the simulator's ground truth. Handoff pose and both arms' reach envelopes are measured, not assumed (`results/reach_envelope.json`: {{ load("reach_envelope.json")["overlap"]["0.03"]["ik_feasible_cells"] }} table cells reachable top-down by *both* arms).

**Bimanual coordination.** Each verified step is queued to the arm that performs it with dependencies (drawer before cutlery, hold before pour, receiver free before a handoff); the idle arm's runnable step is dispatched first. The shared centre of the table is a reservation one arm holds at a time. Skills execute one at a time.

**Verifier.** Before anything moves, the plan is simulated step by step against the scene: reachability (IK on the arm model), grasp preconditions, workspace reservation, drawer-before-cutlery, pour-only-if-the-other-arm-holds-the-mug, joint-velocity limits (halved in **gentle mode** — "gently", "dheere"). Fixable ordering mistakes are repaired and returned (REORDER); anything else is refused with the reason spoken back. Every plan, verdict and skill outcome is a sha256-chained JSON line — edit, delete or reorder one and `make verify-log` names the first bad record.

**Voice.** Speechmatics realtime with partials, end-of-utterance at 0.6 s, speaker diarization, a custom dictionary (arm A, arm B, mug, fork, spoon, plate, drawer) and a Hindi session. **Speaker focus:** the operator is whoever says *"Thali, listen"* (or the first speaker); everyone else is logged and ignored — in the noisy sample a podcast playing in the background is diarized as a second speaker and dropped. The parser tolerates real ASR output: no punctuation, fillers, *"arm eight"* for *arm A*, Hinglish (*"plate ko arm A se uthao"*) and Devanagari (*"टॉप ड्रॉअर खोलो"*).

| sample | WER | normalised WER | skills recovered | speakers | ignored |
|---|---|---|---|---|---|
{{ "\n".join(f"| {r['file']} ({r['language']}) | {r['wer']:.2f} | {r['normalized_wer']:.2f} | {'✓' if r['skills_match'] else '✗'} | {', '.join(r['speakers'])} | {r['ignored_other_speaker'][0][:40] + '…' if r['ignored_other_speaker'] else '—'} |" for r in load("voice_test.json")["rows"]) }}

In hindi.wav the service heard *"arm A se"* as *"आराम से"* ("gently") — reported as measured. First partial → parsed command {{ min(x for r in load("voice_test.json")["rows"] for x in r["partial_to_ready_s"]) }}–{{ max(x for r in load("voice_test.json")["rows"] for x in r["partial_to_ready_s"]) }} s.

**Planner.** Qwen2-VL-2B-Instruct exported to OpenVINO INT4 (`planner/export.sh`) receives the overhead frame, a structured scene description and the transcript, and must answer with JSON that passes the schema *and* the verifier; a rejected answer gets one retry with the reasons, then a deterministic rule planner takes over. Replanning after a failed skill re-enters the same loop with the failure reason. {{ swap_table() }}

**Policies.** {{ load("demos.json")["total_episodes"] }} scripted-expert demonstrations ({{ load("demos.json")["total_frames"] }} frames, 3 cameras, 10 instruction paraphrases per skill, {{ sum(v["recovery"] for v in load("demos.json")["skills"].values()) }} deliberate-miss recovery episodes) recorded as a LeRobot v3 dataset. Per-skill ACT baselines train on the laptop; the multi-task, language-conditioned SmolVLA fine-tunes on Kaggle (`policies/kaggle_smolvla.ipynb`). At run time: learned policy → retry → scripted expert, and the table above reports each stage separately. Per-sub-goal, ACT + fallback reaches {{ ", ".join(f"{k} {pct(v)}" for k, v in load("seeds_act_policy_fallback_test.json")["per_subgoal_rate"].items()) }}.

**Robustness.** Six randomisation axes — placement, mass, friction, shape, lighting, background — with a held-out test split (ranges 1.5× wider, two unseen table textures, one unseen mug shape). Success per seed × axis on the test split:

{{ heat_table("expert") }}

![heatmap](results/heatmap_expert.png)

**OpenVINO.** Every ACT policy is exported to IR with static batch-1 shapes (fp32 / fp16) and quantised to INT8 with NNCF on 300 real frames; the SmolVLA vision encoder is exported the same way. Latency of one policy call (an action chunk of 50 steps):

{{ bench_table() }}

Does optimisation change behaviour? The ten test seeds re-run with the policies executed through each IR:

{{ preserve_table() }}

Sub-goal rates through each IR ({{ "; ".join(f"{k}: " + ", ".join(f"{g} {pct(r)}" for g, r in v["per_subgoal_rate"].items() if r > 0) for k, v in load("preserve.json")["results"].items() if k != "torch") }}) — the same skills succeed at every precision. SmolVLA vision encoder: {{ ", ".join(f"{r['precision']}/{r['device']} {r['p50_ms']} ms" for r in load("smolvla_ir.json")["bench"] if "p50_ms" in r) }}. The VLM planner runs on the CPU; on this machine's iGPU it answers once and then the GPU plugin faults (`docs/BLOCKERS.md`). The IRs need no re-export for an NPU (`-d NPU`); none is present here, so it is not measured.

## Reproduce

```bash
git clone https://github.com/Prashant-thakur77/THALI && cd THALI
uv venv .venv --python 3.11 && uv pip install -r requirements.txt && uv pip install -e .    # or: docker build -t thali .
cp .env.example .env            # SPEECHMATICS_API_KEY, HF_TOKEN

make test                       # 75 tests
make scene                      # rebuild the MuJoCo scene + reach/handoff envelope
make demos EPISODES=150         # scripted-expert demonstrations → LeRobot dataset (4 parallel shards), pushed to the Hub
make train                      # per-skill ACT on the local GPU · SmolVLA: policies/kaggle_smolvla.ipynb
make eval                       # 10-seed tables (expert / policy / retry / fallback), swap matrix, camera-vs-oracle, heatmap
make bench                      # IR export, NNCF INT8, CPU + iGPU latency, precision preservation
make demo VOICE=voice/test_samples/noisy.wav SEED=3     # voice → plan → verify → arms, with TTS
make verify-log LOG=results/audit.jsonl                  # recompute the audit hash chain
python -m docs.render_readme    # regenerate this page and docs/ from results/
```

`planner/export.sh` exports the VLM (~6 GB free disk). Every `reset(seed)` is byte-identical per seed; training seed 1000. Datasets and checkpoints: `Prashant-77/thali_all`, `Prashant-77/thali_smolvla` on the Hub. Hosted demo: `hosting/app.py` (Gradio — replay recorded runs, live planner + verifier).

## Limitations

- Pouring is the hardest skill (expert {{ pct(load("expert_full_task_test.json")["per_skill_rate"]["5_pour"]) }} on the held-out split): water is 20 free spheres and the spout must tip past ~92°.
- The per-skill ACT baselines do not transfer beyond `open_drawer`; the multi-task SmolVLA is the intended policy and its rows fill in when the Kaggle run lands.
- The 2B planner needs the verifier and rule fallback for about half of the commands.
- Both arms move at once only for independent single-arm steps whose objects and targets are ≥ 15 cm apart and outside the shared handoff/pour zone (`ArmQueues.ready_pair`); handoff, hold and pour are still one skill at a time, and the learned-policy executors run sequentially.
- Measured on a Raptor Lake laptop: no NPU rows, VLM on CPU.

## License

Code: Apache-2.0. The SO-101 arm model (`souschef_env/assets/so101/`, Apache-2.0) and the environment/IK/training libraries (gym-aloha skeleton, mink, LeRobot — Apache-2.0; Speechmatics SDKs — MIT) keep their licences.
