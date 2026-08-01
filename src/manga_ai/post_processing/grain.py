from __future__ import annotations

import numpy as np
from PIL import Image


def apply_grain(image: Image.Image, *, strength: float, seed: int) -> Image.Image:
    if strength <= 0:
        return image.convert("L")

    array = np.asarray(image.convert("L"), dtype=np.int16)
    rng = np.random.default_rng(seed)
    amplitude = max(1, int(255 * strength))
    noise = rng.integers(-amplitude, amplitude + 1, size=array.shape)
    return Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8))
