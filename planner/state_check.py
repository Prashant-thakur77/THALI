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


class PixelHeuristic:
    """Colour-blob judgements on the overhead frame. Deliberately crude: it is the no-model fallback."""

    name = "pixels"

    @staticmethod
    def _mask_centroid(img: np.ndarray, lo: tuple, hi: tuple) -> tuple[float, float, int] | None:
        m = np.all((img >= lo) & (img <= hi), axis=-1)
        n = int(m.sum())
        if n < 15:
            return None
        vs, us = np.nonzero(m)
        return float(us.mean()), float(vs.mean()), n

    def ask(self, image: np.ndarray, key: str) -> bool:
        img = np.asarray(image)
        if key == "mug_placed":
            c = self._mask_centroid(img, (20, 60, 130), (110, 150, 255))  # blue mug
            if c is None:
                return False
            x, y = overhead_to_table(c[0], c[1])
            (zx, zy), r = C.ZONES["mug"]
            return bool(np.hypot(x - zx, y - zy) < r + 0.03)
        if key == "plate_placed":
            c = self._mask_centroid(img, (200, 200, 190), (255, 255, 255))  # white plate
            if c is None:
                return False
            x, y = overhead_to_table(c[0], c[1])
            (zx, zy), r = C.ZONES["plate"]
            return bool(np.hypot(x - zx, y - zy) < r + 0.03)
        if key == "drawer_open":
            # drawer front (lighter wood) moves toward -y: count light-wood pixels in the band just below the cabinet
            cx, cy, _ = C.CABINET_POS
            band = [overhead_to_table(u, v) for u, v in ((0, 0),)]  # noqa: F841 (documenting the frame)
            fy = (C.IMAGE_HEIGHT / 2) / np.tan(np.radians(26)) / 0.95
            v0 = int(C.IMAGE_HEIGHT / 2 - (cy - 0.10 + 0.05) * fy)
            v1 = int(C.IMAGE_HEIGHT / 2 - (cy - 0.10 - 0.03 + 0.05) * fy)
            u0 = int(C.IMAGE_WIDTH / 2 + (cx - 0.11) * fy)
            u1 = int(C.IMAGE_WIDTH / 2 + (cx + 0.11) * fy)
            patch = img[max(0, v0):max(0, v1), max(0, u0):max(0, u1)]
            if patch.size == 0:
                return False
            wood = np.all((patch >= (110, 70, 30)) & (patch <= (190, 130, 90)), axis=-1)
            return bool(wood.mean() > 0.25)
        if key in ("fork_placed", "spoon_placed"):
            (zx, zy), r = C.ZONES["fork" if key == "fork_placed" else "spoon"]
            fy = (C.IMAGE_HEIGHT / 2) / np.tan(np.radians(26)) / 0.95
            u = int(C.IMAGE_WIDTH / 2 + zx * fy)
            v = int(C.IMAGE_HEIGHT / 2 - (zy + 0.05) * fy)
            k = int(r * fy)
            patch = img[max(0, v - k):v + k, max(0, u - k):u + k]
            if patch.size == 0:
                return False
            grey = np.all((patch >= (150, 150, 150)) & (patch <= (230, 230, 240)), axis=-1) & (np.abs(patch[..., 0].astype(int) - patch[..., 2].astype(int)) < 25)
            return bool(grey.sum() > 25)
        if key == "poured":
            c = self._mask_centroid(img, (20, 60, 130), (110, 150, 255))
            if c is None:
                return False
            u, v = int(c[0]), int(c[1])
            patch = img[max(0, v - 6):v + 6, max(0, u - 6):u + 6]
            light_blue = np.all((patch >= (60, 130, 200)) & (patch <= (160, 210, 255)), axis=-1)
            return bool(light_blue.sum() > 4)
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
