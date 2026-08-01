from __future__ import annotations

from PIL import Image, ImageChops

from manga_ai.effects.style import ManhwaStyler
from manga_ai.effects.text import ManhwaEffects


def _image(size: tuple[int, int] = (160, 120), color: str = "gray") -> Image.Image:
    return Image.new("RGB", size, color)


def assert_changed(before: Image.Image, after: Image.Image) -> None:
    assert before.size == after.size
    assert ImageChops.difference(before.convert("RGB"), after.convert("RGB")).getbbox() is not None


def test_manhwa_effects_screentone_speed_lines_lighting_and_text() -> None:
    base = _image()

    toned = ManhwaEffects.add_screentone(base, pattern_type="dots", opacity=0.4, scale=1.0)
    speed = ManhwaEffects.add_speed_lines(base, direction="horizontal", intensity=0.5)
    lighting = ManhwaEffects.add_dramatic_lighting(base, style="contrast", intensity=0.5)
    text = ManhwaEffects.add_text_effects(base, "BOOM", (20, 20), effect_type="emphasis")

    assert toned.mode == "RGBA"
    assert speed.mode == "RGBA"
    assert lighting.size == base.size
    assert text.size == base.size
    assert_changed(base, toned)
    assert_changed(base, speed)
    assert_changed(base, text)


def test_manhwa_styler_border_square_text_and_style() -> None:
    styler = ManhwaStyler()
    base = _image((120, 80))

    bordered = ManhwaStyler.add_border(base, border_width=4, border_color="#000000")
    square = ManhwaStyler.pad_to_square(base, size=140, fill_color="#ffffff")
    text_panel = styler.add_text(base, "Hello", speaker="Kai", bubble_type="speech", position=(10, 10))
    styled = styler.apply_style(base)
    adjusted = styler.adjust_panel(Image.new("RGB", (200, 400), "white"), target_width=300)

    assert bordered.size == (128, 88)
    assert square.size == (140, 140)
    assert text_panel.size == base.size
    assert styled.size == base.size
    assert adjusted.width == 300
    assert_changed(base, text_panel)
