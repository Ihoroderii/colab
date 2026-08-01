from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageEnhance

from .config import MangaStyleConfig
from .grain import apply_grain
from .histogram_matcher import match_panel_histogram
from .image_loader import load_image, save_png
from .line_enhancer import enhance_lines
from .resize import resize_panel
from .screentones import apply_screentones
from .tone_normalizer import autocontrast_luma, normalize_tone


def normalize_panel_image(
    image: Image.Image,
    reference: Image.Image | None = None,
    config: MangaStyleConfig | None = None,
) -> Image.Image:
    config = config or MangaStyleConfig()
    panel = resize_panel(image.convert("L"), config.target_size)
    reference_luma = resize_panel(reference.convert("L"), config.target_size) if reference is not None else None

    panel = normalize_tone(
        panel,
        gamma=config.gamma,
        contrast=config.contrast,
        autocontrast_cutoff=config.autocontrast_cutoff,
    )
    panel = match_panel_histogram(panel, reference_luma)
    panel = apply_screentones(panel, config)
    panel = enhance_lines(
        panel,
        radius=config.sharpen_radius,
        percent=config.sharpen_percent,
        threshold=config.sharpen_threshold,
    )
    panel = apply_grain(panel, strength=config.grain_strength, seed=config.grain_seed)
    return panel.convert("RGB")


def normalize_panel_path(
    source_path: Path | str,
    reference_path: Path | str | None,
    output_path: Path | str,
    config: MangaStyleConfig | None = None,
) -> None:
    reference = load_image(reference_path) if reference_path and Path(reference_path).exists() else None
    result = normalize_panel_image(load_image(source_path), reference, config)
    save_png(result, output_path)


def normalize_completed_page(
    source_path: Path | str,
    output_path: Path | str,
    config: MangaStyleConfig | None = None,
) -> None:
    config = config or MangaStyleConfig()
    page = load_image(source_path, mode="L")
    page = autocontrast_luma(page, config.page_autocontrast_cutoff)
    page = ImageEnhance.Contrast(page).enhance(config.page_contrast)
    save_png(page, output_path)
