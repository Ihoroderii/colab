from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops

from manga_ai.config import Config
from manga_ai.runners import main as runner


def test_runner_style_and_bubble_heuristics() -> None:
    config = Config()
    config.scenario.tone = "dramatic"
    config.scenario.genre = "action"
    style = runner._infer_global_style(config, [{"scene": ["night", "shadow"]}])

    assert style["canvas_bg"] == "#0f0f12"
    assert style["panel_gap"] == 48
    assert style["apply_dramatic_lighting"] is True
    assert runner._is_sfx_text("BOOM") is True
    assert runner._infer_bubble_type("Narrator", "Later", "", []) == "narration"
    assert runner._infer_bubble_type("Kai", "HEY!", "", []) == "sfx"
    assert runner._infer_bubble_type("Kai", "(I know.)", "", []) == "thought"
    assert runner._infer_bubble_type("Kai", "quiet...", "dramatic", []) == "whisper"


def test_runner_reference_image_helpers(tmp_path: Path) -> None:
    source = tmp_path / "reference.png"
    Image.new("RGB", (100, 200), "blue").save(source)

    loaded = runner._load_reference_image(str(source))
    fit = runner._prepare_reference_image(loaded, (300, 300), "fit")
    crop = runner._prepare_reference_image(loaded, (300, 300), "crop")

    assert loaded.size == (100, 200)
    assert fit.size == (296, 296)
    assert crop.size == (296, 296)
    assert runner._round_to_multiple(301, 8) == 296


def test_runner_pose_drawing_and_width_estimation() -> None:
    config = Config()
    pose = runner._text_to_pose("standing")
    image = runner._draw_pose(pose, size=128)
    width = runner._estimate_base_width(config.style, "long scene " * 50, "speech " * 30)

    assert image.size == (128, 128)
    assert width >= config.style.panel_width


def test_runner_style_normalization_helper(tmp_path: Path) -> None:
    config = Config()
    config.style_normalization.enabled = True
    config.style_normalization.width = 64
    config.style_normalization.height = 96
    reference = tmp_path / "style.png"
    Image.new("RGB", (64, 96), "white").save(reference)
    config.style_normalization.reference_path = str(reference)

    result = runner._apply_reference_style_normalization(config, Image.new("RGB", (128, 128), "gray"))

    assert result.size == (64, 96)


def test_runner_configured_bubbles_internal_backend_changes_image() -> None:
    config = Config()
    config.bubbles.backend = "internal"
    styler = runner.ManhwaStyler()
    image = Image.new("RGB", (220, 180), "white")

    result = runner._apply_configured_bubbles(
        config,
        styler,
        image,
        "Hello",
        "Kai",
        "speech",
        (20, 20),
    )

    assert result.size == image.size
    assert ImageChops.difference(image, result).getbbox() is not None


def test_runner_generate_panel_api_backend_is_mockable(monkeypatch) -> None:
    config = Config()
    config.model.image_backend = "api"
    config.bubbles.backend = "internal"
    config.style.auto_panel_width = False
    config.style.square_panels = False
    config.style.panel_border_width = 0
    config.style.apply_dramatic_lighting = False
    config.style_normalization.enabled = False

    monkeypatch.setattr(
        runner,
        "generate_image_with_api",
        lambda *args, **kwargs: Image.new("RGB", (128, 192), "white"),
    )

    panel, meta = runner._generate_panel(
        config,
        context="Classroom",
        scene_prompt="Kai stands near the window",
        speaker="Kai",
        speech_text="Hello",
    )

    assert panel.width == config.style.panel_width
    assert meta["generation_mode"] == "api_txt2img"
    assert meta["bubble_backend"] == "internal"
