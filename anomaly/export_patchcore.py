"""Export the trained PatchCore checkpoint to OpenVINO IR (separate from training so a failed export never costs a re-fit).

    .venv-anomalib/bin/python -m anomaly.export_patchcore
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import os
VARIANT = os.environ.get("THALI_ANOMALY_VARIANT", "")          # "" = full frame, "crop" = table crop (anomaly/crop.py)
OUT = ROOT / "anomaly" / (f"model_{VARIANT}" if VARIANT else "model")


def main() -> None:
    from anomalib.deploy import ExportType
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    ckpt = sorted(OUT.rglob("model.ckpt"))[-1]
    import torch, anomalib
    torch.serialization.add_safe_globals([anomalib.PrecisionType])
    model = Patchcore.load_from_checkpoint(str(ckpt), weights_only=False)
    ir_dir = OUT / "openvino"
    if ir_dir.exists():
        shutil.rmtree(ir_dir)
    path = Engine(default_root_dir=str(OUT / "runs")).export(model=model, export_type=ExportType.OPENVINO, export_root=str(OUT),
                                                            input_size=(240, 320), ckpt_path=str(ckpt))
    summary = json.loads((OUT / "train_summary.json").read_text()) if (OUT / "train_summary.json").exists() else {}
    summary["exported_ir"] = str(path)
    (OUT / "train_summary.json").write_text(json.dumps(summary, indent=1))
    print("exported", path)


if __name__ == "__main__":
    main()
