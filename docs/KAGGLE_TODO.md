# Things that need your Hugging Face / Kaggle accounts

Nothing here blocks the local pipeline; each item is a one-line command once the credential exists.

## 1. Push the demo dataset (Phase 2)
No `HF_TOKEN` was available on this machine, so `Prashant-77/thali_all` is recorded locally under
`data/lerobot/thali_all/` (2.8 GB, 420 episodes, 299 233 frames; see `results/demos.json`) but **not pushed**.

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
Filled in by Phase 3: upload `policies/kaggle_smolvla.ipynb`, attach the dataset, run, and the notebook pushes
`Prashant-77/thali_smolvla`. Until that checkpoint exists on the Hub, every SmolVLA row in the results is marked
"pending SmolVLA run".
