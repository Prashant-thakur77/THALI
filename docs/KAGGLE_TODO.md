# Things that need your Hugging Face / Kaggle accounts

Nothing here blocks the local pipeline; each item is a one-line command once the credential exists.

## 1. Push the demo dataset (Phase 2)
No `HF_TOKEN` was available on this machine, so `Prashant-77/thali_all` is recorded locally under
`data/lerobot/thali_all/` — **done**: 1050 episodes, 742 837 frames are on the Hub as of 16 Sep 2026.

```bash
echo "HF_TOKEN=hf_..." >> .env          # write access to Prashant-77
.venv/bin/python -c "
from dotenv import load_dotenv; load_dotenv()
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('Prashant-77/thali_all', root='data/lerobot/thali_all')
ds.push_to_hub(tags=['thali','so101','bimanual','mujoco'], private=False)"
```
(`make demos` does the same push automatically when `HF_TOKEN` is set.)

## 2. SmolVLA fine-tune on Kaggle (Phase 3)
Needs item 1 first (the notebook streams the dataset from the Hub).
1. kaggle.com → New Notebook → File → Import `policies/kaggle_smolvla.ipynb`.
2. Settings: Accelerator **GPU T4 ×2** (or P100), Internet **on**, Persistence on.
3. Add-ons → Secrets → `HF_TOKEN` = a write token for `Prashant-77`.
4. Run all. ~5–6 h for 20 000 steps at batch 16; checkpoints every 5 000 steps are pushed to
   `Prashant-77/thali_smolvla` (`--policy.push_to_hub=true`), the last cell re-uploads the final one.
5. Back here: `make eval POLICY=smolvla` (or `python -m eval.run_seeds --policy smolvla`) adds the SmolVLA rows to
   `results/seeds.json`. Until the checkpoint exists on the Hub those rows read **"pending SmolVLA run"**.
