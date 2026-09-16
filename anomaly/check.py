"""OpenVINO-only inference for the PatchCore table-state check, plus the held-out scorer.

``TableAnomalyCheck`` loads the exported IR (CPU by default) and returns an anomaly score and a verdict for an overhead
frame; the runtime calls it after each skill as a second, learned "did something go wrong on the table" signal next to
the pixel/VLM state check.  ``python -m anomaly.check --score`` runs every test frame through the IR, computes image
AUROC, the per-anomaly-type detection rate at the exported threshold, and the IR latency on CPU and iGPU, and writes
results/anomaly.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "anomaly" / "model"
DATA = ROOT / "anomaly" / "data"


def _find_ir(d: Path) -> Path:
    xml = sorted(d.rglob("*.xml"))
    if not xml:
        raise FileNotFoundError(f"no OpenVINO IR under {d}; run anomaly/train_patchcore.py first")
    return xml[0]


class TableAnomalyCheck:
    name = "patchcore_openvino"

    def __init__(self, device: str = "CPU", model_dir: Path = MODEL_DIR):
        import openvino as ov
        xml = _find_ir(model_dir)
        meta_path = xml.with_name("metadata.json")
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        core = ov.Core()
        self.model = core.read_model(xml)
        self.compiled = core.compile_model(self.model, device)
        self.req = self.compiled.create_infer_request()
        shp = self.model.inputs[0].get_partial_shape()
        self.h, self.w = int(shp[2].get_length()), int(shp[3].get_length())
        # anomalib 2.x exports the post-processor inside the graph: pred_score is min-max normalised so that the
        # learned image threshold maps to 0.5, and pred_label is the thresholded decision.
        self.threshold = float(self.meta.get("image_threshold", 0.5))
        self.out_score = next((o for o in self.compiled.outputs if "pred_score" in o.get_any_name()), self.compiled.outputs[0])
        self.device = device

    def _prep(self, frame: np.ndarray) -> np.ndarray:
        """RGB uint8 HxWx3 -> [1,3,H,W] float in [0,1]; the ImageNet normalisation lives inside the exported graph."""
        from PIL import Image
        img = Image.fromarray(frame).resize((self.w, self.h))
        x = np.asarray(img, dtype=np.float32) / 255.0
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

    def score(self, frame: np.ndarray) -> float:
        outs = self.req.infer({0: self._prep(frame)})
        return float(np.asarray(outs[self.out_score]).reshape(-1)[0])

    def ask(self, frame: np.ndarray) -> bool:
        """True when the table looks nominal."""
        return self.score(frame) < self.threshold


def _auroc(scores_pos: list[float], scores_neg: list[float]) -> float:
    pos, neg = np.asarray(scores_pos), np.asarray(scores_neg)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def score_all(devices: tuple[str, ...] = ("CPU", "GPU")) -> dict:
    from PIL import Image
    chk = TableAnomalyCheck("CPU")
    manifest = json.loads((DATA / "manifest.json").read_text())
    good = [float(chk.score(np.asarray(Image.open(p).convert("RGB")))) for p in sorted((DATA / "test" / "good").glob("*.png"))]
    per_type, bad_all = {}, []
    for k in manifest["test_bad"]:
        s = [float(chk.score(np.asarray(Image.open(p).convert("RGB")))) for p in sorted((DATA / "test" / k).glob("*.png"))]
        per_type[k] = {"n": len(s), "detected": int(sum(x >= chk.threshold for x in s)), "auroc_vs_good": round(_auroc(s, good), 3), "median_score": round(float(np.median(s)), 4)}
        bad_all += s
    fp = int(sum(x >= chk.threshold for x in good))
    # threshold-free operating point: the score cut that lets through 90% of nominal frames, and what it catches
    thr10 = float(np.quantile(good, 0.90))
    tpr10 = {k: int(sum(x >= thr10 for x in [r for r in bad_all[i * per_type[k]["n"]:(i + 1) * per_type[k]["n"]]])) for i, k in enumerate(per_type)}
    out = {"model": "PatchCore (Anomalib) → OpenVINO IR", "threshold": chk.threshold, "input_hw": [chk.h, chk.w],
           "test_good": len(good), "false_positives": fp, "image_auroc": round(_auroc(bad_all, good), 3),
           "detected": int(sum(x >= chk.threshold for x in bad_all)), "test_bad_total": len(bad_all), "per_type": per_type,
           "at_10pct_fpr": {"threshold": round(thr10, 4), "detected": int(sum(x >= thr10 for x in bad_all)), "per_type": tpr10}, "latency_ms": {}}
    frame = np.asarray(Image.open(sorted((DATA / "test" / "good").glob("*.png"))[0]).convert("RGB"))
    for dev in devices:
        try:
            c = TableAnomalyCheck(dev)
            for _ in range(5):
                c.score(frame)
            t = [];
            for _ in range(30):
                t0 = time.perf_counter(); c.score(frame); t.append((time.perf_counter() - t0) * 1000)
            out["latency_ms"][dev] = {"p50": round(float(np.median(t)), 2), "mean": round(float(np.mean(t)), 2)}
        except Exception as e:  # device missing / plugin error: report, don't hide
            out["latency_ms"][dev] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    res = score_all()
    (ROOT / "results" / "anomaly.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_type"}, indent=1)); print(json.dumps(res["per_type"], indent=1))
