"""Merge per-skill demo shards (recorded in parallel by make_demos.py) into one dataset and one results file.

    python -m expert.merge_demos --shards data/lerobot/shard_* --out data/lerobot/thali_all --repo-id Prashant-77/thali_all [--push]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "lerobot" / "thali_all")
    ap.add_argument("--repo-id", default=f"{os.environ.get('HF_USER', 'Prashant-77')}/thali_all")
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "demos.json")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if args.out.exists():
        shutil.rmtree(args.out)
    shards = sorted(args.shards)
    aggregate_datasets([f"shard/{s.name}" for s in shards], args.repo_id, roots=shards, aggr_root=args.out)
    ds = LeRobotDataset(args.repo_id, root=args.out)
    # merge the shard summaries; episode indices are renumbered in shard order
    summary = {"repo_id": args.repo_id, "root": str(args.out), "fps": ds.fps, "total_episodes": ds.num_episodes,
               "total_frames": ds.num_frames, "skills": {}, "shards": [str(s) for s in shards]}
    offset = 0
    for s in shards:
        part = json.loads((s / "demos_shard.json").read_text())
        for k, v in part["skills"].items():
            v = dict(v)
            v["episodes"] = [e + offset for e in v["episodes"]]
            summary["skills"][k] = v
        offset += part["total_episodes"]
        for key in ("cameras", "image_hw", "split", "recovery_fraction"):
            summary[key] = part[key]
    summary["elapsed_s_wall"] = max(json.loads((s / "demos_shard.json").read_text())["elapsed_s"] for s in shards)
    args.results.write_text(json.dumps(summary, indent=2))
    print(f"merged {ds.num_episodes} episodes / {ds.num_frames} frames -> {args.out}")
    if args.push:
        if not (os.environ.get("HF_TOKEN") or (Path.home() / ".cache" / "huggingface" / "token").exists()):
            print("No HF token: dataset NOT pushed (see docs/KAGGLE_TODO.md)")
        else:
            ds.push_to_hub(tags=["thali", "so101", "bimanual", "mujoco"], private=False)
            print("pushed", args.repo_id)


if __name__ == "__main__":
    main()
