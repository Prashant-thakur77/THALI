# Training SmolVLA on Kaggle

The multi-task SmolVLA is the one policy that does not train on the laptop (RTX 3050, 6 GB). It fine-tunes
`lerobot/smolvla_base` on a free Kaggle GPU from `policies/kaggle_smolvla.ipynb`; everything else in the repo runs
locally. Checkpoints live on the Hub at [`Prashant-77/thali_smolvla`](https://huggingface.co/Prashant-77/thali_smolvla)
(the best one is `step_14000/`), so the runtime and `eval/` never need Kaggle.

## Dataset
`make demos` records 1050 scripted-expert episodes (742 837 frames, 3 cameras) as a LeRobot v3 dataset under
`data/lerobot/thali_all/` and pushes it to [`Prashant-77/thali_all`](https://huggingface.co/datasets/Prashant-77/thali_all)
when `HF_TOKEN` is in `.env`. The notebook streams it from there.

## The notebook
1. Kaggle → New Notebook → File → Import `policies/kaggle_smolvla.ipynb`.
2. Settings: Accelerator **GPU T4 ×2** (or P100), Internet **on**, Persistence **on**.
3. Attach a private dataset holding `hf_token.txt` (a write token for `Prashant-77`); the notebook finds it with a glob under `/kaggle/input`.
4. Run all. A session trains up to 4500 steps (a Kaggle session is capped at 12 h) and resumes from the newest
   checkpoint on the Hub, so the 20 000-step run is a chain of sessions. Each session pushes its newest checkpoint and a
   `TRAINING_STEP.txt` marker.

Pins that matter: `transformers==4.57.6`, `huggingface_hub==0.35.3` (transformers 5 removed an import LeRobot uses);
`--rename_map` maps the dataset cameras `overhead/wrist_a/wrist_b` onto the base model's `camera1/2/3`
(`runtime/executors.py` applies the same map at inference).

From the command line instead of the browser (`pip install kaggle`, token in `~/.kaggle/`):
```bash
kaggle kernels push -p kg_kernel/            # kernel-metadata.json: enable_gpu, enable_internet, dataset_sources
kaggle kernels status prashantthakur77/thali
```

## Evaluating a checkpoint
```bash
python -m eval.skill_eval --policy smolvla --tag step_14000        # per skill, results/skill_eval_smolvla_*.json
python -m eval.run_seeds --policy smolvla --mode policy_only --seeds 10 --split test   # full task, results/seeds_smolvla_*_test.json
python -m docs.render_readme                                        # rows in README / EVIDENCE / MODEL_CARD
```
The per-skill and full-task rows in the README come from these files and nothing else.
