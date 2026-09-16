"""Seeds x perturbation-axis heatmap (plan Phase 9.1).

For each of the six randomisation axes, the full task runs on seeds 0-9 of the test split with ONLY that axis
randomised (plus the all-axes column), giving a 10 x 7 success matrix per policy.  Writes
results/heatmap_<policy>.json and results/heatmap_<policy>.png (matplotlib, no seaborn).

    python -m eval.heatmap --policy expert
    python -m eval.heatmap --policy act --mode policy_fallback
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from souschef_env.randomize import AXES
from eval.run_seeds import run

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="expert")
    ap.add_argument("--mode", default="policy_fallback")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    mode = "expert" if args.policy == "expert" else args.mode
    cols = [*(((a,), a) for a in AXES), (AXES, "all")]
    matrix: dict[str, list[bool]] = {}
    per_axis: dict[str, float] = {}
    for axes, label in cols:
        out = run(args.policy, mode, args.seeds, args.split, tuple(axes), args.device, tag=f"heatmap_{args.policy}_{label}")
        if out is None:
            print("checkpoint missing; aborting heatmap")
            return
        matrix[label] = [r["success"] for r in out["rows"]]
        per_axis[label] = out["success_rate"]
    res = {"policy": args.policy, "mode": mode, "split": args.split, "seeds": args.seeds, "columns": [c[1] for c in cols],
           "matrix": matrix, "per_axis_success_rate": per_axis,
           "hardest_axis": min((a for a in AXES), key=lambda a: per_axis[a])}
    (ROOT / "results" / f"heatmap_{args.policy}.json").write_text(json.dumps(res, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        M = np.array([[1.0 if matrix[c][s] else 0.0 for c in res["columns"]] for s in range(args.seeds)])
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(res["columns"])), [f"{c}\n{per_axis[c]:.0%}" for c in res["columns"]])
        ax.set_yticks(range(args.seeds), [f"seed {s}" for s in range(args.seeds)])
        ax.set_title(f"Thali full task, {args.policy} ({mode}), {args.split} split: success per seed x perturbation axis")
        for s in range(args.seeds):
            for j, c in enumerate(res["columns"]):
                ax.text(j, s, "✓" if matrix[c][s] else "✗", ha="center", va="center", fontsize=11)
        fig.tight_layout()
        fig.savefig(ROOT / "results" / f"heatmap_{args.policy}.png", dpi=120)
    except Exception as e:  # pragma: no cover
        print("no plot:", e)
    print(json.dumps(per_axis, indent=1), "hardest:", res["hardest_axis"])


if __name__ == "__main__":
    main()
