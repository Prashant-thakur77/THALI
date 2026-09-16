# Thali — make targets (plan §1). All stubs until their phase lands.
SEEDS  ?= 10
DEVICE ?= CPU
PY     ?= .venv/bin/python
export MUJOCO_GL ?= glfw   # EGL is broken on this box, see docs/BLOCKERS.md

.PHONY: demos train eval bench demo verify-log test scene

EPISODES ?= 60
demos:        ## Phase 2 — scripted-expert demos (4 parallel shards, EPISODES=$(EPISODES) per skill) -> LeRobotDataset, pushed if HF_TOKEN is set
	rm -rf data/lerobot/shard_1 data/lerobot/shard_2 data/lerobot/shard_3 data/lerobot/shard_4
	$(PY) -m expert.make_demos --episodes $(EPISODES) --skills open_drawer pick_place_fork  --root data/lerobot/shard_1 --repo-id shard/shard_1 --results data/lerobot/shard_1/demos_shard.json & \
	$(PY) -m expert.make_demos --episodes $(EPISODES) --skills pick_place_plate pick_place_mug --root data/lerobot/shard_2 --repo-id shard/shard_2 --results data/lerobot/shard_2/demos_shard.json & \
	$(PY) -m expert.make_demos --episodes $(EPISODES) --skills handoff_spoon hold_mug --root data/lerobot/shard_3 --repo-id shard/shard_3 --results data/lerobot/shard_3/demos_shard.json & \
	$(PY) -m expert.make_demos --episodes $(EPISODES) --skills pour --root data/lerobot/shard_4 --repo-id shard/shard_4 --results data/lerobot/shard_4/demos_shard.json & \
	wait
	$(PY) -m expert.merge_demos --shards data/lerobot/shard_1 data/lerobot/shard_2 data/lerobot/shard_3 data/lerobot/shard_4 --out data/lerobot/thali_all --repo-id Prashant-77/thali_all --push
	$(PY) -m expert.sweep --seeds 10 --split train
	$(PY) -m expert.sweep --seeds 10 --split test

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
