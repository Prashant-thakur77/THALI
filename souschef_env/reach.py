"""Measure each arm's reach envelope and compute the handoff overlap (plan Phase 1.3).

Stage 1: a forward-kinematics sweep of the four joints that move the gripper
rasterises site positions into a 1 cm grid at the grasp height band.
Stage 2: every cell in the *intersection* of the two arms' grids is checked
with top-down IK for both arms (VectorForge's lesson: position reach is far
more generous than top-down reach that can actually grasp).  The handoff pose
is the centroid of the doubly-IK-feasible region.  Everything is written to
results/reach_envelope.json; nothing here is hand-typed.

Run:  python -m souschef_env.reach
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from souschef_env import constants as C
from souschef_env.ik import ArmIK

CELL = 0.01
GRID_X = (-0.45, 0.45)
GRID_Y = (-0.36, 0.36)
SWEEP = (("shoulder_pan", 25), ("shoulder_lift", 21), ("elbow_flex", 21), ("wrist_flex", 13))
GRASP_HEIGHTS = (0.03, 0.06)   # cutlery/plate rim and mug/bottle body grasp heights (m above the table)
Z_TOL = 0.03
IK_YAWS = (0.0, np.pi / 3, 2 * np.pi / 3)


def fk_points(model: mujoco.MjModel, arm: str) -> np.ndarray:
    data = mujoco.MjData(model)
    p = C.ARM_PREFIX[arm]
    site = model.site(p + "gripperframe").id
    adr, grids = [], []
    for joint, n in SWEEP:
        jid = model.joint(p + joint).id
        adr.append(model.jnt_qposadr[jid])
        lo, hi = model.jnt_range[jid]
        grids.append(np.linspace(lo * 0.98, hi * 0.98, n))
    out = np.empty((int(np.prod([n for _, n in SWEEP])), 3))
    for i, vals in enumerate(itertools.product(*grids)):
        data.qpos[adr] = vals
        mujoco.mj_kinematics(model, data)
        out[i] = data.site_xpos[site]
    return out


def occupancy(points: np.ndarray, height: float) -> np.ndarray:
    nx = int(round((GRID_X[1] - GRID_X[0]) / CELL)) + 1
    ny = int(round((GRID_Y[1] - GRID_Y[0]) / CELL)) + 1
    band = points[np.abs(points[:, 2] - height) <= Z_TOL]
    grid = np.zeros((nx, ny), dtype=bool)
    ix = np.round((band[:, 0] - GRID_X[0]) / CELL).astype(int)
    iy = np.round((band[:, 1] - GRID_Y[0]) / CELL).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    grid[ix[ok], iy[ok]] = True
    # close 1-cell holes left by the finite sweep
    closed = grid.copy()
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        closed |= np.roll(np.roll(grid, dx, 0), dy, 1)
    return closed


def cell_xy(ix: int, iy: int) -> tuple[float, float]:
    return GRID_X[0] + ix * CELL, GRID_Y[0] + iy * CELL


def measure(out_path: Path) -> dict:
    t0 = time.time()
    model = mujoco.MjModel.from_xml_path(str(C.ARMS_XML))
    ik = ArmIK(model)
    result: dict = {"cell_m": CELL, "sweep": dict(SWEEP), "heights_m": list(GRASP_HEIGHTS), "arms": {}, "overlap": {}}
    pts = {a: fk_points(model, a) for a in C.ARMS}
    for a in C.ARMS:
        r = np.hypot(pts[a][:, 0] - C.ARM_BASE_POS[a][0], pts[a][:, 1] - C.ARM_BASE_POS[a][1])
        result["arms"][a] = {"base_xy_m": list(C.ARM_BASE_POS[a][:2]), "fk_samples": int(len(pts[a])),
                             "max_radius_m": float(r.max()), "position_cells": {}}
    for h in GRASP_HEIGHTS:
        occ = {a: occupancy(pts[a], h) for a in C.ARMS}
        for a in C.ARMS:
            result["arms"][a]["position_cells"][str(h)] = int(occ[a].sum())
        both = occ["a"] & occ["b"]
        cells = np.argwhere(both)
        feasible, tested = [], 0
        # only test cells in the middle band where a handoff makes sense (|y| small) to keep IK cost bounded
        for ix, iy in cells:
            x, y = cell_xy(ix, iy)
            if abs(y) > 0.12 or abs(x) > 0.15:
                continue
            tested += 1
            ok = True
            for arm in C.ARMS:
                for yaw in IK_YAWS:
                    if not ik.solve_top_down(arm, np.array([x, y, h]), yaw, max_iters=40).ok:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                feasible.append((x, y))
        feas = np.array(feasible) if feasible else np.zeros((0, 2))
        entry = {"position_overlap_cells": int(both.sum()), "ik_tested_cells": tested, "ik_feasible_cells": int(len(feas)),
                 "ik_feasible_area_m2": float(len(feas) * CELL * CELL)}
        if len(feas):
            c = feas.mean(axis=0)
            # handoff = feasible cell nearest the centroid
            best = feas[np.argmin(np.hypot(*(feas - c).T))]
            entry.update({"bbox_x_m": [float(feas[:, 0].min()), float(feas[:, 0].max())],
                          "bbox_y_m": [float(feas[:, 1].min()), float(feas[:, 1].max())],
                          "handoff_xy_m": [float(best[0]), float(best[1])]})
        result["overlap"][str(h)] = entry
    # the handoff pose: at the lower grasp height (objects are handed over resting on the table -- via-table)
    h0 = str(GRASP_HEIGHTS[0])
    if "handoff_xy_m" in result["overlap"][h0]:
        result["handoff_pose_m"] = [*result["overlap"][h0]["handoff_xy_m"], GRASP_HEIGHTS[0]]
        result["handover_mode"] = "via_table"
    result["elapsed_s"] = round(time.time() - t0, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    r = measure(root / "results" / "reach_envelope.json")
    for a in C.ARMS:
        print(f"arm {a}: max radius {r['arms'][a]['max_radius_m']:.3f} m, cells {r['arms'][a]['position_cells']}")
    for h, e in r["overlap"].items():
        print(f"h={h}: position overlap {e['position_overlap_cells']} cells, IK-feasible {e['ik_feasible_cells']}/{e['ik_tested_cells']} -> handoff {e.get('handoff_xy_m')}")
    print("handoff pose:", r.get("handoff_pose_m"), r.get("handover_mode"), f"({r['elapsed_s']} s)")
