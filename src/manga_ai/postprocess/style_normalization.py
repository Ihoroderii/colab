"""Compatibility wrapper for the split post_processing package."""

from __future__ import annotations

from manga_ai.post_processing.config import MangaStyleConfig, StyleNormalizationSettings
from manga_ai.post_processing.grain import apply_grain
from manga_ai.post_processing.histogram_matcher import match_panel_histogram
from manga_ai.post_processing.line_enhancer import enhance_lines
from manga_ai.post_processing.metrics import PanelMetrics, StyleMetrics, calculate_image_metrics, calculate_metrics
from manga_ai.post_processing.outlier_detector import is_style_outlier, should_reprocess
from manga_ai.post_processing.pipeline import normalize_completed_page, normalize_panel_image, normalize_panel_path
from manga_ai.post_processing.resize import resize_panel
from manga_ai.post_processing.screentones import apply_screentones, create_tone_masks
from manga_ai.post_processing.tone_normalizer import apply_gamma, normalize_tone

__all__ = [
    "MangaStyleConfig",
    "StyleNormalizationSettings",
    "PanelMetrics",
    "StyleMetrics",
    "apply_gamma",
    "apply_grain",
    "apply_screentones",
    "calculate_image_metrics",
    "calculate_metrics",
    "create_tone_masks",
    "enhance_lines",
    "is_style_outlier",
    "match_panel_histogram",
    "normalize_completed_page",
    "normalize_panel_image",
    "normalize_panel_path",
    "normalize_tone",
    "resize_panel",
    "should_reprocess",
]
