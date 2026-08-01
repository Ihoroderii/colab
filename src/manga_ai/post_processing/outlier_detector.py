from __future__ import annotations

from .metrics import StyleMetrics


def is_style_outlier(panel: StyleMetrics, reference: StyleMetrics) -> bool:
    return any(
        [
            abs(panel.mean_brightness - reference.mean_brightness) > 25,
            abs(panel.contrast - reference.contrast) > 20,
            abs(panel.black_ratio - reference.black_ratio) > 0.15,
        ]
    )


def should_reprocess(panel: StyleMetrics, reference: StyleMetrics) -> bool:
    return is_style_outlier(panel, reference)
