# Thali — pitch, business slide, video plan

## One sentence
**Thali — voice-controlled two-arm robot that sets the table and pours drinks for people who can't use their hands.**

## Short description (lablab field)
Thali listens to you and sets your table: two SO-101 arms open the drawer, lay out cutlery and plates, hold your mug and pour — for anyone with a stroke, Parkinson's, arthritis or age who can still speak but can't reach. Runs entirely on an Intel laptop with OpenVINO; voice by Speechmatics.

## Long description — opening
Millions of people can talk perfectly well but can't set a table or pour a glass of water without help — stroke survivors, people with tremors, older adults living alone. Thali gives that back. Say what you want and two robot arms do it: one holds the mug steady while the other pours, one passes the plate to the side you can't reach. Say "stop" and it stops. It runs on the laptop already in the house, so nothing leaves the home.

## How each part serves that person
- **Two arms** = the steady hands they lost: hold + pour, handoff across the table (via the table, inside the measured overlap of both arms' reach — `results/reach_envelope.json`).
- **Realtime Speechmatics** = "stop" works mid-motion (barge-in on partials), only the operator's voice commands the robot (speaker focus: the background speaker in noisy.wav is ignored — `results/voice_test.json`), tired and Hindi speech still recovers the skill sequence (100% of samples).
- **Local OpenVINO** = private and offline: planner + policies run on the Intel CPU/iGPU, 44.6 tok/s for the planner on a laptop CPU.
- **Verifier** = never pours without the mug held, never swings through the shared workspace: 20/20 unsafe plans caught, every decision hash-chained.
- **Gentle mode** = halves joint-velocity limits when a person is seated ("gently", "dheere").
- **10 seeds × 6 perturbation axes** = real tables are messy; the expert's hardest axis is **shape** (`results/heatmap_expert.json`).

## Scale slide (sources as cited in the plan; not measured here)
- ~1.3 billion people live with significant disability (WHO); ~1 in 4 adults will have a stroke, upper-limb weakness among the most common lasting effects.
- Existing feeding-assist robots are single-arm, pre-programmed, thousands of dollars; two SO-101 arms are ~$200 plus a laptop.
- Same build → elder care at home, hospital isolation wards, hospitality/kitchen automation. Why local inference: privacy, no connectivity dependence, no per-call cost.

## Video (≤ 3 min) — shot list
0:00 a person at a table, then the one-sentence pitch + architecture · 0:10 live command → Speechmatics partials → JSON plan → verifier ALLOW rows · 0:40 execution with the HUD (skill, oracle ✓, camera ✓) — `python -m runtime.demo --seed 3 --voice voice/test_samples/noisy.wav --tts --video results/demo_seed3.mp4` · 1:30 barge-in "stop, use the other arm" (`--barge-in "other arm@4"`), second speaker ignored, Hindi command (`--voice voice/test_samples/hindi.wav --language hi`) · 2:00 10-seed 2×5 montage with counter → 8/10 expert / 0/10 ACT+fallback · 2:30 bench table on the Intel box (`results/bench.md`), preservation table, 20/20 caught · 2:50 repo + HF links. Hardware caption on every table: measured on i7-13650HX CPU + UHD iGPU; same IR runs on Core Ultra NPU with -d NPU and static shapes, not measured here.
