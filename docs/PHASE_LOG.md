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
