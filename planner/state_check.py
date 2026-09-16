"""Camera-based state check (plan §0: "after each skill a camera-based state check (VLM yes/no)").

Asks the local VLM one yes/no question about the overhead frame per sub-goal; the sim oracle is the ground
truth it is scored against in eval/camera_vs_oracle.py.  Without the model the check degrades to a pixel
heuristic (colour-blob positions in the overhead view), so the runtime loop still runs and is still
measurable; results record which backend produced each judgement.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np

from souschef_env import constants as C

QUESTIONS = {
    "drawer_open": "Is the wooden drawer at the top of the image pulled open (its front is away from the cabinet)? Answer yes or no.",
    "plate_placed": "Is the white plate resting on the table near the centre-bottom of the image, between the two arms? Answer yes or no.",
    "fork_placed": "Is there a fork lying on the table to the left of the plate area (not in the drawer)? Answer yes or no.",
    "spoon_placed": "Is there a spoon lying on the table to the right of the plate area (not in the drawer)? Answer yes or no.",
    "mug_placed": "Is the blue mug resting on the table on the right side, near the right arm? Answer yes or no.",
    "poured": "Are there blue water drops inside the blue mug? Answer yes or no.",
}


def overhead_to_table(u: float, v: float, w: int = C.IMAGE_WIDTH, h: int = C.IMAGE_HEIGHT) -> tuple[float, float]:
    """Pixel -> table xy for the fixed overhead camera (pos (0,-0.05,0.95), fovy 52, looking straight down)."""
    fy = (h / 2) / np.tan(np.radians(52 / 2))
    z = 0.95
    x = (u - w / 2) / fy * z
    y = -(v - h / 2) / fy * z - 0.05
    return float(x), float(y)


def _px(x: float, y: float) -> tuple[int, int]:
    """Table xy -> overhead pixel (inverse of overhead_to_table)."""
    fy = (C.IMAGE_HEIGHT / 2) / np.tan(np.radians(52 / 2)) / 0.95
    return int(C.IMAGE_WIDTH / 2 + x * fy), int(C.IMAGE_HEIGHT / 2 - (y + 0.05) * fy)


def _patch(img: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    u0, v1 = _px(x0, y0)
    u1, v0 = _px(x1, y1)
    return img[max(0, v0):max(0, v1), max(0, u0):max(0, u1)]


class PixelHeuristic:
    """Camera-only judgements on the overhead frame, deliberately simple: the no-model fallback.

    Colour blobs locate the plate (white) and the mug (blue).  Everything else is *change* against the frame
    captured at reset (``set_reference``): the region in front of the cabinet changes when the drawer slides
    out, a cutlery zone changes when a piece lands in it, the mug's interior changes when water is in it.
    Lighting is randomised per episode but constant within one, which is what makes differencing robust
    where absolute colour thresholds were not.
    """

    name = "pixels"
    DIFF = 18.0  # mean |Δ| over a patch (0-255) that counts as "changed"

    def __init__(self) -> None:
        self.ref: np.ndarray | None = None

    def set_reference(self, image: np.ndarray) -> None:
        self.ref = np.asarray(image).astype(np.int16)

    @staticmethod
    def _mask_centroid(img: np.ndarray, lo: tuple, hi: tuple) -> tuple[float, float, int] | None:
        m = np.all((img >= lo) & (img <= hi), axis=-1)
        n = int(m.sum())
        if n < 15:
            return None
        vs, us = np.nonzero(m)
        return float(us.mean()), float(vs.mean()), n

    def _changed(self, img: np.ndarray, box: tuple[float, float, float, float]) -> float:
        if self.ref is None:
            return 0.0
        a = _patch(np.asarray(img).astype(np.int16), *box)
        b = _patch(self.ref, *box)
        if a.size == 0 or a.shape != b.shape:
            return 0.0
        return float(np.abs(a - b).mean())

    def ask(self, image: np.ndarray, key: str) -> bool:
        img = np.asarray(image)
        if key == "mug_placed":
            c = self._mask_centroid(img, (20, 60, 130), (110, 150, 255))
            if c is None:
                return False
            x, y = overhead_to_table(c[0], c[1])
            (zx, zy), r = C.ZONES["mug"]
            return bool(np.hypot(x - zx, y - zy) < r + 0.03)
        if key == "plate_placed":
            c = self._mask_centroid(img, (200, 200, 190), (255, 255, 255))
            if c is None:
                return False
            x, y = overhead_to_table(c[0], c[1])
            (zx, zy), r = C.ZONES["plate"]
            return bool(np.hypot(x - zx, y - zy) < r + 0.03)
        if key == "drawer_open":
            cx, cy, _ = C.CABINET_POS
            # the strip the drawer front slides into: y from (front, closed) - 0.13 to (front, closed) - 0.02
            return self._changed(img, (cx - 0.10, cy - 0.078 - 0.13, cx + 0.10, cy - 0.078 - 0.02)) > self.DIFF
        if key in ("fork_placed", "spoon_placed"):
            (zx, zy), r = C.ZONES["fork" if key == "fork_placed" else "spoon"]
            return self._changed(img, (zx - r, zy - r, zx + r, zy + r)) > self.DIFF * 0.6
        if key == "poured":
            c = self._mask_centroid(img, (20, 60, 130), (110, 150, 255))
            if c is None:
                return False
            u, v = int(c[0]), int(c[1])
            patch = img[max(0, v - 5):v + 5, max(0, u - 5):u + 5]
            light_blue = np.all((patch >= (60, 130, 200)) & (patch <= (170, 215, 255)), axis=-1)
            return bool(light_blue.sum() > 3)
        raise KeyError(key)


class VLMStateCheck:
    name = "vlm"

    def __init__(self, vlm) -> None:  # planner.plan.OpenVinoVLM
        self.vlm = vlm
        self.last: dict = {}

    def ask(self, image: np.ndarray, key: str) -> bool:
        t0 = time.perf_counter()
        text = self.vlm(image, QUESTIONS[key] + " Reply with a single word.")
        self.last = {"latency_s": time.perf_counter() - t0, "raw": text[:80]}
        return bool(re.search(r"\byes\b", text.lower())) and not re.search(r"\bno\b", text.lower()[:10])


def make_state_check(vlm=None):
    return VLMStateCheck(vlm) if vlm is not None else PixelHeuristic()
