# Thali roadmap — beyond the dinner table

Status on 17 Sep 2026: submission in; post-deadline work running (ACT retrain on 1050 episodes, SmolVLA on Kaggle,
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
| **Clear the table** | reverse task: cutlery back to the drawer, mug to the cabinet, close the drawer | 3 primitives (put-in-drawer, close_drawer, place-in-cabinet), oracles, 3×150 demos | "clear the table" N/10 held-out |
| **Two place settings** | second plate/fork/spoon set; zones per seat; the planner must count | assets + zones, planner prompt with seats, workspace rule for the far seat (handoff needed) | 2-seat task N/10 |
| **Glass + jug / bottle cap** | pour into a glass, one arm holds the bottle while the other unscrews the cap (true bimanual, not handoff) | cap joint on the bottle, twist primitive with torque limit | cap-off N/10, pour-into-glass N/10 |
| **Target-volume pour** — *done 17 Sep* | "a little" / normal / "fill it up" → 3 / 6 / 12 spheres; the roll stops per control step when the oracle counts target − 1 | done: `results/pour_amount.json` — normal within ±2 on 4/5, little 2/5 (overshoot: water leaves in bursts), full 2/5 (two layouts saturate at 6 spheres) | next: slower final roll chunk near the exit angle; bottle neck geometry for "full" |
| **Tray carry (two-arm lift)** | both arms lift one object (tray with the mug on it) and carry it together | coordinated two-arm IK legs through the step barrier (now possible), tray asset | tray delivered without spill N/10 |

## Tier 2 — interaction (days)

| item | what it adds | needs | result row |
|---|---|---|---|
| **Follow-ups & corrections** | "a bit more", "no, the other side", "left of the plate" resolved against the last step | dialogue state in the runtime, relative-position parsing, verifier re-check | 20 scripted follow-ups → N correct |
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
| **3 000 episodes/skill + augmentation** | ACT/SmolVLA that beat the expert's speed and match its success | 5 h recording (4 shards), Kaggle runs | per-skill policy-only ≥ 18/20 |
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
2. Clear-the-table (2 days) — doubles the task list, reuses everything. (target-volume pour: done, needs finer control for 'little')
3. Follow-ups/corrections + ask-when-ambiguous (2 days) — the strongest voice-track story.
4. 3 000-episode recording in the background throughout; SmolVLA retrain at the end (1 week wall, mostly unattended).
5. Two-arm tray carry (3 days) — the first skill that only two arms can do.
