#!/usr/bin/env bash
# Per-skill ACT baselines on the local RTX 3050 6 GB (plan Phase 3.2): batch 8, AMP, chunk 50, 8k steps each.
# Episodes per skill come from results/demos.json so the split is exactly what the recorder wrote.
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=glfw
STEPS=${STEPS:-8000}
SKILLS=${SKILLS:-"open_drawer pick_place_fork pick_place_plate pick_place_mug handoff_spoon hold_mug pour"}
for skill in $SKILLS; do
  out=outputs/act_$skill
  if [ -f "$out/checkpoints/last/pretrained_model/config.json" ]; then echo "skip $skill (done)"; continue; fi
  rm -rf "$out"
  EP=$(.venv/bin/python -c "import json; print(json.dumps(json.load(open('results/demos.json'))['skills']['$skill']['episodes']))")
  echo "=== ACT $skill: $STEPS steps"
  .venv/bin/lerobot-train --policy.type=act --policy.device=cuda --policy.use_amp=true \
    --dataset.repo_id=Prashant-77/thali_all --dataset.root=data/lerobot/thali_all --dataset.episodes="$EP" \
    --policy.chunk_size=50 --policy.n_action_steps=50 --batch_size=8 --steps=$STEPS --log_freq=500 \
    --save_freq=$STEPS --eval_freq=0 --num_workers=4 --output_dir="$out" --job_name="act_$skill" \
    --wandb.enable=false --policy.push_to_hub=false --seed=1000
done
echo "all ACT skills trained"
