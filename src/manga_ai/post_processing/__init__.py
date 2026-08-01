"""Deterministic post-processing for generated manga panels."""

from .config import MangaStyleConfig, StyleNormalizationSettings
from .metrics import StyleMetrics, PanelMetrics, calculate_metrics
from .outlier_detector import is_style_outlier, should_reprocess
from .pipeline import normalize_completed_page, normalize_panel_image, normalize_panel_path

__all__ = [
    "MangaStyleConfig",
    "StyleNormalizationSettings",
    "StyleMetrics",
    "PanelMetrics",
    "calculate_metrics",
    "is_style_outlier",
    "should_reprocess",
    "normalize_completed_page",
    "normalize_panel_image",
    "normalize_panel_path",
]
