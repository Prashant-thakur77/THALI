# Thali — make targets (plan §1). All stubs until their phase lands.
SEEDS  ?= 10
DEVICE ?= CPU
PY     ?= .venv/bin/python
export MUJOCO_GL ?= glfw   # EGL is broken on this box, see docs/BLOCKERS.md

.PHONY: demos train eval bench demo verify-log test scene

demos:        ## Phase 2 — scripted-expert demos -> LeRobotDataset
	@echo "demos: not implemented"

train:        ## Phase 3 — train ACT baselines locally / SmolVLA on Kaggle
	@echo "train: not implemented"

eval:         ## Phase 9 — 10-seed eval on test_ranges (SEEDS=$(SEEDS))
	@echo "eval: not implemented"

bench:        ## Phase 8 — OpenVINO bench (DEVICE=$(DEVICE); CPU|GPU only, no NPU on this box)
	@echo "bench: not implemented"

demo:         ## Phase 7 — voice -> planner -> verifier -> arms
	@echo "demo: not implemented"

verify-log:   ## Phase 5 — recompute the hash chain of the audit log
	@echo "verify-log: not implemented"

test:         ## run the pytest suite
	$(PY) -m pytest -q

scene:        ## Phase 1 — rebuild assets/dinner_table.xml and the reach envelope
	$(PY) -m souschef_env.build_scene
	$(PY) -m souschef_env.reach
