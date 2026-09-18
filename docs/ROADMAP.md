# Thali roadmap — beyond the dinner table

Status on 17 Sep 2026: Phase 11 work running (ACT retrain on 1050 episodes, SmolVLA on Kaggle,
per-skill evals, PatchCore table check, concurrent two-arm execution). This is the plan for what comes next, ranked by
how much it moves the project versus what it costs. Every item ends with the number it would put in the README.

How a new skill enters the system (the same five files every time):
1. `expert/primitives.py` — a scripted primitive (IK legs, jaw commands, retries).
2. `souschef_env/oracles.py` — a ground-truth predicate for "did it happen" (+ `scene_description.py` if a new object).
3. `verifier/rules.py` — preconditions and a bad-plan injection for it; `planner/schema.json` + prompts — the planner can name it.
4. `expert/make_demos.py` — instruction paraphrases + demos (150 episodes ≈ 25 min of recording per skill, 4 shards).
5. `policies/` — retrain the per-skill ACT (42 min) and the multi-task SmolVLA (Kaggle, 6 h); `eval/skill_eval.py` scores it.

## Tier 1 — more of the same table (days, high judge impact)

| item | what it adds | needs | result row |
|---|---|---|---|
| **Clear the table** — *done 17 Sep* | reverse task: fork back to the drawer, spoon handed B→A and back, drawer closed (`put_in_drawer`, `close_drawer`; verifier orders close after stow) | done: `results/clear_table.json` **9/10** held-out (the miss is the set-up) | next: demos + ACT for the two skills; mug to the cabinet |
| **Two place settings** | second plate/fork/spoon set; zones per seat; the planner must count | assets + zones, planner prompt with seats, workspace rule for the far seat (handoff needed) | 2-seat task N/10 |
| **Glass + jug / bottle cap** | pour into a glass, one arm holds the bottle while the other unscrews the cap (true bimanual, not handoff) | cap joint on the bottle, twist primitive with torque limit | cap-off N/10, pour-into-glass N/10 |
| **Target-volume pour** — *done 17 Sep* | "a little" / normal / "fill it up" → 3 / 6 / 12 spheres; the roll stops per control step when the oracle counts target − 1 | done: `results/pour_amount.json` — normal within ±2 on 4/5, little 2/5 (overshoot: water leaves in bursts), full 2/5 (two layouts saturate at 6 spheres) | next: slower final roll chunk near the exit angle; bottle neck geometry for "full" |
| **Tray carry (two-arm lift)** | both arms lift one object (tray with the mug on it) and carry it together | coordinated two-arm IK legs through the step barrier (now possible), tray asset | tray delivered without spill N/10 |

## Tier 2 — interaction (days)

| item | what it adds | needs | result row |
|---|---|---|---|
| **Follow-ups & corrections** — *done 17 Sep* | "again", "a bit more", "no, the other side", "the other arm" resolved against the last executed step (`runtime/followups.py`), verified, presets in both web UIs | done: `results/followups.json` **20/20** resolved to the expected plan, 14/16 verifier-approved (the two refusals are correct: the other arm cannot reach) | next: "left of the plate" relative positions; spoken clarification questions |
| **Per-person preferences** | two diarised speakers, each with a seat and preferences ("I don't take water") | speaker → seat map, plan per seat | 2-speaker sessions N/10 |
| **Ask when ambiguous** | "which mug?" when two match; refuse with a reason when an item is missing (already partly there) | clarification state + TTS question, resume on answer | ambiguous commands: N asked, 0 wrong |
| **Kitchen timer & reminders** | "tell me when the tea has steeped 3 minutes" — voice-only skills mixed with arm skills | scheduler in the state machine | — (demo value) |

## Tier 3 — perception & robustness (days–weeks)

| item | what it adds | needs | result row |
|---|---|---|---|
| **Anomaly check on difference images** — *done 17 Sep* | full frame (AUROC 0.76, 38/60 false alarms) and table crop (0.78, 42/60) both failed because the held-out split randomises the table texture; PatchCore on \|frame − reset reference\| crops cancels the texture (AUROC 0.94); a nominal set generated from *real* expert runs (post-skill arm poses, real drawer travel) makes it usable live: **AUROC 0.943, 72/75 flagged, 13/57 false alarms; in the loop 3/4 knocked plates flagged, 1/4 false alarm; 70 ms CPU** | next: more nominal seeds (60 → 300) to cut false alarms; 2× crop resolution for small knocks | live 4/4 flagged, 0 false alarms |
| **Unseen objects** | new mug/plate meshes and colours never in the demos; the policy and the planner must cope | 5 extra assets in the test split only | per-skill success on unseen shapes |
| **Clutter & moved cabinet** | distractor objects, cabinet position randomised ±10 cm | randomiser axes 7–8, heatmap rerun | heatmap columns |
| **Camera check with a small VLM** | replace the pixel heuristic with a fine-tuned SmolVLM yes/no on 2 k labelled frames | frames from the oracle, 1 h fine-tune | camera vs oracle 84 % → > 95 % |

## Tier 4 — learned policies that win (weeks)

| item | what it adds | needs | result row |
|---|---|---|---|
| **Longer training before more data** | finding 17 Sep: 60 → 1050 episodes (and 8k → 12k steps) moved per-skill ACT only on the fork (2 → 6/20); drawer 20/20, plate 8/20, mug 1/20, hold 10/20, handoff 0/20, pour 1/20 are unchanged, so the bottleneck is not the episode count. ACT normally needs 50k–100k steps — **confirmed**: plate 8/20 → **15/20** at 50k steps on the same 150 episodes (3.5 h on the RTX 3050). 50k runs for fork, mug, hold, handoff and pour are queued (17 h) | then temporal ensembling, larger chunk, 2× image resolution, 100k steps | every skill ≥ 15/20 |
| **SmolVLA: finish on one machine** | finding 18 Sep: 5k → 9.5k → 14k steps on a Kaggle T4 at batch 16 took the drawer skill 1 → 12 → 14/20 and hold 3 → 4 → 8/20; the last 6k steps on the RTX 3050 at batch 4 with a fresh optimizer *lost* ground (11/20, 3/20). Continue on Kaggle after the quota reset from the step-14 000 checkpoint, batch 16, with the optimizer state | next Kaggle session(s) | drawer ≥ 18/20, hold ≥ 12/20 |
| **3 000 episodes/skill + augmentation** | after the above: ACT/SmolVLA that beat the expert's speed and match its success | 5 h recording (4 shards), Kaggle runs | per-skill policy-only ≥ 18/20 |
| **Policy-first execution** | run the learned policy by default and the expert only as fallback in the demo video | after the row above | full task policy-only N/10 |
| **Core Ultra / NPU** | ACT + PatchCore IR timed on NPU with `-d NPU`; the export is already static-shape | access to a Core Ultra machine (or Intel DevCloud) | NPU ms per call |
| **Real SO-101 arms** | sim-to-real of drawer + plate skills with the same LeRobot dataset format | 2 arms (~$220), camera mounts, 200 real demos | real-world drawer N/10 |

## Tier 5 — different tables

| item | why | needs |
|---|---|---|
| **Packing / shipping bench** (the other Intel prompt) | same stack: boxes instead of plates, "fragile on top" as a verifier rule | assets + 4 primitives + rules |
| **Medication tray** | assistive-care fit: pill cups to seats, verifier blocks a wrong seat | assets + per-person rules |
| **Café counter** | cup, saucer, spoon, milk pour — a public-facing demo | assets |

## Suggested order for the next two weeks
1. ~~Difference-image anomaly retrain~~ done (AUROC 0.94); optional: spill sensitivity.
2. ~~Clear-the-table~~ done (9/10). Live site: done (Hugging Face Docker Space `thali-live`).
3. ~~Follow-ups/corrections~~ done (20/20); ask-when-ambiguous next.
4. 3 000-episode recording in the background throughout; SmolVLA retrain at the end (1 week wall, mostly unattended).
5. Two-arm tray carry (3 days) — the first skill that only two arms can do.
