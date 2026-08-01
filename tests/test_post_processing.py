from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from manga_ai.post_processing import (
    MangaStyleConfig,
    calculate_metrics,
    normalize_completed_page,
    normalize_panel_image,
    normalize_panel_path,
    should_reprocess,
)
from manga_ai.post_processing.histogram_matcher import match_panel_histogram
from manga_ai.post_processing.line_enhancer import enhance_lines
from manga_ai.post_processing.resize import resize_panel
from manga_ai.post_processing.screentones import create_tone_masks
from manga_ai.post_processing.tone_normalizer import normalize_tone
from manga_ai.postprocess.style_normalization import StyleNormalizationSettings


def _sample_panel(size: tuple[int, int] = (320, 480)) -> Image.Image:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 40, size[0] - 30, size[1] - 40), outline="black", width=6)
    draw.line((40, size[1] - 80, size[0] - 40, 80), fill="black", width=4)
    draw.ellipse((110, 110, 210, 210), outline="black", width=5, fill=(190, 190, 190))
    return image


def test_module_steps_are_composable() -> None:
    config = MangaStyleConfig(width=128, height=192, grain_strength=0)
    panel = resize_panel(_sample_panel(), config.target_size)
    toned = normalize_tone(panel, gamma=config.gamma, contrast=config.contrast)
    matched = match_panel_histogram(toned, toned)
    masks = create_tone_masks(matched, config)
    enhanced = enhance_lines(
        matched,
        radius=config.sharpen_radius,
        percent=config.sharpen_percent,
        threshold=config.sharpen_threshold,
    )

    assert panel.size == (128, 192)
    assert toned.mode == "L"
    assert matched.mode == "L"
    assert sorted(masks) == ["black", "dark_tone", "light_tone"]
    assert enhanced.size == (128, 192)


def test_pipeline_normalizes_panel_and_page(tmp_path: Path) -> None:
    config = MangaStyleConfig(width=128, height=192, grain_strength=0)
    panel = _sample_panel()
    reference = Image.new("RGB", (128, 192), "white")

    result = normalize_panel_image(panel, reference, config)
    assert result.mode == "RGB"
    assert result.size == (128, 192)

    source_path = tmp_path / "panel.png"
    out_path = tmp_path / "panel_normalized.png"
    page_path = tmp_path / "page.png"
    page_out = tmp_path / "page_normalized.png"
    panel.save(source_path)

    normalize_panel_path(source_path, None, out_path, config)
    normalize_completed_page(out_path, page_out, config)

    assert out_path.exists()
    assert page_out.exists()
    assert Image.open(page_out).mode in ("L", "RGB")


def test_metrics_and_compatibility_imports(tmp_path: Path) -> None:
    path = tmp_path / "panel.png"
    _sample_panel().save(path)
    metrics = calculate_metrics(path)
    assert metrics.mean_brightness > 0
    assert metrics.brightness == metrics.mean_brightness
    assert metrics.contrast > 0
    assert StyleNormalizationSettings().target_size == (768, 1024)
    assert should_reprocess(metrics, metrics) is False
