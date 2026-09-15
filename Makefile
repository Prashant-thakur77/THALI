# Thali — make targets (plan §1). All stubs until their phase lands.
SEEDS  ?= 10
DEVICE ?= CPU
PY     ?= .venv/bin/python

.PHONY: demos train eval bench demo verify-log

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
