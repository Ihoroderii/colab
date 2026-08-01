from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class StyleMetrics:
    mean_brightness: float
    contrast: float
    black_ratio: float
    white_ratio: float
    edge_density: float = 0.0

    @property
    def brightness(self) -> float:
        return self.mean_brightness


PanelMetrics = StyleMetrics


def _edge_density(image: np.ndarray) -> float:
    vertical = np.abs(np.diff(image, axis=0))
    horizontal = np.abs(np.diff(image, axis=1))
    return float(((vertical > 35).mean() + (horizontal > 35).mean()) / 2.0)


def calculate_metrics(path: Path | str) -> StyleMetrics:
    image = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return StyleMetrics(
        mean_brightness=float(image.mean()),
        contrast=float(image.std()),
        black_ratio=float((image < 40).mean()),
        white_ratio=float((image > 230).mean()),
        edge_density=_edge_density(image),
    )


def calculate_image_metrics(image: Image.Image) -> StyleMetrics:
    array = np.asarray(image.convert("L"), dtype=np.float32)
    return StyleMetrics(
        mean_brightness=float(array.mean()),
        contrast=float(array.std()),
        black_ratio=float((array < 40).mean()),
        white_ratio=float((array > 230).mean()),
        edge_density=_edge_density(array),
    )
