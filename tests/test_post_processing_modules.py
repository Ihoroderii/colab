from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from manga_ai.post_processing.ai_restyler import AIRestyleRequest, AIRestyler
from manga_ai.post_processing.config import MangaStyleConfig
from manga_ai.post_processing.grain import apply_grain
from manga_ai.post_processing.histogram_matcher import match_panel_histogram
from manga_ai.post_processing.image_loader import load_image, save_png
from manga_ai.post_processing.line_enhancer import enhance_lines
from manga_ai.post_processing.metrics import calculate_image_metrics, calculate_metrics
from manga_ai.post_processing.outlier_detector import is_style_outlier, should_reprocess
from manga_ai.post_processing.pipeline import normalize_completed_page, normalize_panel_image, normalize_panel_path
from manga_ai.post_processing.resize import resize_panel
from manga_ai.post_processing.screentones import apply_screentones, create_tone_masks, dot_pattern
from manga_ai.post_processing.tone_normalizer import apply_gamma, autocontrast_luma, normalize_tone


def _sample_image(size: tuple[int, int] = (96, 144)) -> Image.Image:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 12, size[0] - 10, size[1] - 12), outline="black", width=4)
    draw.rectangle((20, 30, size[0] - 20, size[1] // 2), fill=(120, 120, 120))
    draw.line((8, size[1] - 20, size[0] - 8, 20), fill="black", width=3)
    return image


def test_config_target_size_and_serialization() -> None:
    config = MangaStyleConfig(width=320, height=480, gamma=0.9)

    assert config.target_size == (320, 480)
    assert config.to_dict()["gamma"] == 0.9


def test_image_loader_loads_and_saves_png(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "panel.png"
    save_png(_sample_image(), path)

    loaded = load_image(path, mode="L")

    assert path.exists()
    assert loaded.mode == "L"


def test_resize_panel_fits_exact_target_size() -> None:
    resized = resize_panel(_sample_image((120, 80)), (64, 96))

    assert resized.size == (64, 96)


def test_tone_normalizer_outputs_luma_and_applies_gamma() -> None:
    image = Image.new("L", (2, 1))
    image.putdata([64, 192])

    gamma_result = apply_gamma(image, gamma=0.5)
    autocontrast_result = autocontrast_luma(image, cutoff=0)
    normalized = normalize_tone(_sample_image(), gamma=0.96, contrast=1.08)

    assert gamma_result.mode == "L"
    assert list(gamma_result.getdata()) != [64, 192]
    assert autocontrast_result.mode == "L"
    assert normalized.mode == "L"


def test_histogram_matcher_returns_luma_image() -> None:
    panel = Image.new("L", (32, 32), 80)
    reference = Image.new("L", (32, 32), 200)

    matched = match_panel_histogram(panel, reference)

    assert matched.mode == "L"
    assert matched.size == panel.size


def test_line_enhancer_preserves_size_and_mode() -> None:
    image = _sample_image().convert("L")

    enhanced = enhance_lines(image, radius=1.2, percent=110, threshold=3)

    assert enhanced.mode == image.mode
    assert enhanced.size == image.size


def test_screentones_create_masks_patterns_and_output() -> None:
    config = MangaStyleConfig(width=64, height=64)
    image = Image.new("L", (64, 64), 140)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 20, 64), fill=20)
    draw.rectangle((21, 0, 42, 64), fill=80)

    masks = create_tone_masks(image, config)
    pattern = dot_pattern((64, 64), step=8, radius=2)
    toned = apply_screentones(image, config)

    assert sorted(masks) == ["black", "dark_tone", "light_tone"]
    assert all(mask.mode == "L" for mask in masks.values())
    assert pattern.mode == "L"
    assert toned.mode == "L"
    assert toned.size == image.size


def test_grain_is_deterministic_for_same_seed() -> None:
    image = Image.new("L", (32, 32), 128)

    first = apply_grain(image, strength=0.02, seed=123)
    second = apply_grain(image, strength=0.02, seed=123)
    different = apply_grain(image, strength=0.02, seed=124)

    assert ImageChops.difference(first, second).getbbox() is None
    assert ImageChops.difference(first, different).getbbox() is not None


def test_metrics_for_path_and_image() -> None:
    image = _sample_image()
    metrics = calculate_image_metrics(image)

    assert metrics.mean_brightness > 0
    assert metrics.contrast > 0
    assert 0 <= metrics.black_ratio <= 1
    assert 0 <= metrics.white_ratio <= 1
    assert 0 <= metrics.edge_density <= 1


def test_metrics_for_file_path(tmp_path: Path) -> None:
    path = tmp_path / "panel.png"
    _sample_image().save(path)

    metrics = calculate_metrics(path)

    assert metrics.brightness == metrics.mean_brightness
    assert metrics.contrast > 0


def test_outlier_detector_accepts_same_metrics_and_rejects_large_difference() -> None:
    reference = calculate_image_metrics(Image.new("L", (32, 32), 220))
    same = calculate_image_metrics(Image.new("L", (32, 32), 220))
    outlier = calculate_image_metrics(Image.new("L", (32, 32), 20))

    assert is_style_outlier(same, reference) is False
    assert should_reprocess(outlier, reference) is True


def test_pipeline_normalizes_panel_image_and_paths(tmp_path: Path) -> None:
    config = MangaStyleConfig(width=64, height=96, grain_strength=0)
    source_path = tmp_path / "source.png"
    reference_path = tmp_path / "reference.png"
    panel_out = tmp_path / "panel_out.png"
    page_out = tmp_path / "page_out.png"
    _sample_image().save(source_path)
    Image.new("RGB", (64, 96), "white").save(reference_path)

    result = normalize_panel_image(_sample_image(), Image.open(reference_path), config)
    normalize_panel_path(source_path, reference_path, panel_out, config)
    normalize_completed_page(panel_out, page_out, config)

    assert result.mode == "RGB"
    assert result.size == (64, 96)
    assert panel_out.exists()
    assert page_out.exists()


def test_ai_restyler_boundary_raises_until_provider_is_configured() -> None:
    async def _call() -> None:
        restyler = AIRestyler()
        await restyler.restyle(AIRestyleRequest(image=b"image-bytes"))

    try:
        asyncio.run(_call())
    except NotImplementedError as exc:
        assert "not configured" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("AIRestyler should raise until a provider is configured")
