"""Target-volume pour: does "a little" / "normal" / "full" deliver the requested number of water spheres?

For each seed arm B holds the mug and arm A pours with the requested amount; the expert stops the roll as soon as the
oracle counts the target spheres in the mug (constants.POUR_TARGET_SPHERES: little 3, normal 6, full 12).  Reports
the spheres actually delivered, the absolute error and the hit rate within +/-2 spheres.  Writes results/pour_amount.json.

    python -m eval.pour_amount --seeds 0 1 2 3 4 --amounts little normal full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

import souschef_env  # noqa: F401
from souschef_env import constants as C
from expert.primitives import Expert

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--amounts", nargs="+", default=["little", "normal", "full"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "pour_amount.json")
    a = ap.parse_args()
    env = gym.make("souschef_env/Thali-v0", disable_env_checker=True).unwrapped
    rows = []
    for amount in a.amounts:
        for seed in a.seeds:
            env.reset(seed=seed, options={"split": a.split})
            env.render_enabled = False
            ex = Expert(env)
            h = ex.hold_mug("b")
            if not h.ok:
                rows.append({"amount": amount, "seed": seed, "hold_ok": False}); continue
            r = ex.pour("a", amount)
            rows.append({"amount": amount, "seed": seed, "hold_ok": True, "pour_ok": bool(r.ok), "target": r.detail.get("target_spheres"),
                         "poured": int(r.detail.get("poured", 0)), "aborted": bool(r.detail.get("aborted")), "sim_steps": int(r.steps)})
            print(f"{amount} seed {seed}: target {rows[-1]['target']} poured {rows[-1]['poured']} ok={r.ok}", flush=True)
    summary = {}
    for amount in a.amounts:
        rs = [r for r in rows if r["amount"] == amount and r.get("hold_ok")]
        if not rs:
            continue
        err = [abs(r["poured"] - r["target"]) for r in rs]
        summary[amount] = {"target": C.POUR_TARGET_SPHERES[amount], "n": len(rs), "mean_poured": round(float(np.mean([r["poured"] for r in rs])), 1),
                           "mean_abs_error": round(float(np.mean(err)), 2), "within_2": int(sum(e <= 2 for e in err)), "reached_target": int(sum(r["pour_ok"] for r in rs))}
    out = {"split": a.split, "seeds": a.seeds, "targets": C.POUR_TARGET_SPHERES, "summary": summary, "rows": rows}
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
