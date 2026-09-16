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
