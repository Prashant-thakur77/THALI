"""Table crop for the overhead camera (240x320): drop the randomised background and lighting falloff around the
table so the anomaly detector scores the table, not the room.  The box is the table's footprint in the fixed
overhead view (souschef_env/assets: overhead camera looks straight down), with a small margin."""

from __future__ import annotations

import numpy as np
from PIL import Image

CROP = (40, 10, 280, 210)   # left, top, right, bottom in the 320x240 frame
OUT_SIZE = (320, 240)       # resized back to the model's input size (w, h)


def crop_table(frame: np.ndarray) -> np.ndarray:
    img = Image.fromarray(np.asarray(frame)).crop(CROP).resize(OUT_SIZE, Image.BILINEAR)
    return np.asarray(img)


def diff_from_reference(frame: np.ndarray, reference: np.ndarray, gain: float = 3.0) -> np.ndarray:
    """|frame - reference| on the table crop, amplified and clipped to uint8.

    The runtime captures the reference at reset (the same frame the pixel state check uses), so table texture,
    lighting falloff and background cancel; what remains is everything that moved -- expected placements as well as
    disturbances, and PatchCore learns which of those are nominal."""
    a = crop_table(frame).astype(np.float32)
    b = crop_table(reference).astype(np.float32)
    return np.clip(np.abs(a - b) * gain, 0, 255).astype(np.uint8)
