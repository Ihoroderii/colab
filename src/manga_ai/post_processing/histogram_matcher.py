from __future__ import annotations

import numpy as np
from PIL import Image

try:
    from skimage.exposure import match_histograms
except Exception:  # pragma: no cover - optional dependency fallback
    match_histograms = None


def match_panel_histogram(panel: Image.Image, reference: Image.Image | None) -> Image.Image:
    if reference is None or match_histograms is None:
        return panel.convert("L")

    panel_array = np.asarray(panel.convert("L"), dtype=np.uint8)
    reference_array = np.asarray(reference.convert("L"), dtype=np.uint8)
    result = match_histograms(panel_array, reference_array, channel_axis=None)
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
