"""Train Anomalib PatchCore on nominal overhead frames and export it to OpenVINO IR.

Runs in the separate ``.venv-anomalib`` environment (Anomalib pins its own torch/lightning).  Only the exported IR +
metadata are used by the runtime (``anomaly/check.py``), which needs nothing but openvino.

    .venv-anomalib/bin/python -m anomaly.train_patchcore --backbone wide_resnet50_2 --coreset 0.1
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import os
VARIANT = os.environ.get("THALI_ANOMALY_VARIANT", "")          # "" = full frame, "crop" = table crop (anomaly/crop.py)
DATA = ROOT / "anomaly" / (f"data_{VARIANT}" if VARIANT else "data")
OUT = ROOT / "anomaly" / (f"model_{VARIANT}" if VARIANT else "model")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="wide_resnet50_2")
    ap.add_argument("--coreset", type=float, default=0.1)
    ap.add_argument("--size", type=int, nargs=2, default=(240, 320))
    a = ap.parse_args()
    from anomalib.data import Folder
    from anomalib.deploy import ExportType
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    manifest = json.loads((DATA / "manifest.json").read_text())
    abnormal = [f"test/{k}" for k in manifest["test_bad"]]
    dm = Folder(name="thali_table", root=str(DATA), normal_dir="train/good", normal_test_dir="test/good", abnormal_dir=abnormal,
                train_batch_size=16, eval_batch_size=16, num_workers=4)
    model = Patchcore(backbone=a.backbone, layers=["layer2", "layer3"], pre_trained=True, coreset_sampling_ratio=a.coreset, num_neighbors=9)
    engine = Engine(default_root_dir=str(OUT / "runs"), max_epochs=1, accelerator="auto", devices=1)
    t0 = time.time()
    engine.fit(model=model, datamodule=dm)
    fit_s = time.time() - t0
    metrics = engine.test(model=model, datamodule=dm)
    ir_dir = OUT / "openvino"
    if ir_dir.exists():
        shutil.rmtree(ir_dir)
    engine.export(model=model, export_type=ExportType.OPENVINO, export_root=str(OUT), input_size=tuple(a.size))
    summary = {"backbone": a.backbone, "layers": ["layer2", "layer3"], "coreset_sampling_ratio": a.coreset, "input_size": list(a.size),
               "train_good": manifest["train_good"], "test_good": manifest["test_good"], "test_bad": manifest["test_bad"],
               "fit_s": round(fit_s, 1), "lightning_test_metrics": {k: float(v) for m in metrics for k, v in m.items()}}
    (OUT / "train_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
