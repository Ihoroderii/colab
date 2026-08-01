from __future__ import annotations

import json
import os
import shutil
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from manga_ai.config import Config
from manga_ai.effects.style import ManhwaStyler
from manga_ai.integrations.bubbles import BubbleCharacter, BubbleLine, apply_manhwa_bubbles
from manga_ai.models.demo import CameraPlan, CharacterState, DialogueLine, PanelPlan, ScenePlan
from manga_ai.pipelines.assemble import ManhwaAssembler
from manga_ai.pipelines.blender_control import render_control_image
from manga_ai.pipelines.image_api import generate_image_with_api
from manga_ai.post_processing import (
    StyleNormalizationSettings,
    calculate_metrics,
    normalize_completed_page,
    normalize_panel_image,
    should_reprocess,
)


CHARACTER_PROMPTS: dict[str, str] = {
    "Kai": (
        "Kai, teenage boy protagonist, short messy black hair, dark school jacket, "
        "focused eyes, calm but determined expression"
    ),
    "Maria": (
        "Maria, teenage girl, long white hair, neat school uniform, sharp intelligent eyes, "
        "quiet serious expression"
    ),
}

DEMO_NEGATIVE_PROMPT = (
    "low quality, blurry, distorted hands, extra limbs, bad anatomy, messy text, watermark, "
    "logo, gore, blood, wounds, weapons, explicit violence"
)


def build_default_scene_plan(scenario: str) -> ScenePlan:
    """Create a deterministic 3-panel classroom demo plan from a short scenario."""
    scenario = scenario.strip() or "Kai and Maria discover a secret promise in their classroom."
    return ScenePlan(
        scenario=scenario,
        episode_title="Demo 01: Classroom Oath",
        panels=[
            PanelPlan(
                panel_number=1,
                title="Establishing classroom tension",
                location="classroom",
                action=(
                    "After class, Kai stands by the left side of a classroom table while Maria waits near "
                    "the window on the right. The room is quiet and tense."
                ),
                camera=CameraPlan(shot="wide", angle_degrees=35, target="both characters", mood="quiet tension"),
                characters=[
                    CharacterState(name="Kai", position=(-1.4, 0.0, 0.0), pose="standing", facing="right", expression="determined"),
                    CharacterState(name="Maria", position=(1.4, 0.25, 0.0), pose="standing", facing="left", expression="serious"),
                ],
                props=["classroom table", "window", "chairs", "blackboard"],
                dialogue=[
                    DialogueLine(speaker=None, text="After class, the promise returns.", kind="narration", position=(24, 24)),
                ],
                seed=4101,
            ),
            PanelPlan(
                panel_number=2,
                title="Kai close-up",
                location="classroom",
                action=(
                    "Close-up on Kai as he lowers his voice. The table edge and window light frame his face."
                ),
                camera=CameraPlan(shot="close_up", angle_degrees=35, target="Kai", mood="resolve"),
                characters=[
                    CharacterState(name="Kai", position=(-0.35, 0.0, 0.0), pose="standing", facing="camera", expression="determined"),
                    CharacterState(name="Maria", position=(1.45, 0.3, 0.0), pose="standing", facing="left", expression="watching"),
                ],
                props=["classroom table", "window"],
                dialogue=[
                    DialogueLine(speaker="Kai", text="I won't run from it this time.", kind="speech", position=(36, 36)),
                ],
                seed=4102,
            ),
            PanelPlan(
                panel_number=3,
                title="Maria answer",
                location="classroom",
                action=(
                    "Over Maria's shoulder toward Kai. Maria answers with a calm but serious expression while "
                    "the city light cuts through the classroom window."
                ),
                camera=CameraPlan(shot="over_shoulder", angle_degrees=35, target="Kai from Maria shoulder", mood="dramatic"),
                characters=[
                    CharacterState(name="Kai", position=(-1.1, 0.15, 0.0), pose="standing", facing="right", expression="surprised"),
                    CharacterState(name="Maria", position=(1.2, 0.0, 0.0), pose="standing", facing="left", expression="serious"),
                ],
                props=["classroom table", "window", "chairs"],
                dialogue=[
                    DialogueLine(speaker="Maria", text="Then keep your oath, Kai.", kind="speech", position=(32, 34)),
                ],
                seed=4103,
            ),
        ],
    )


def build_panel_prompt(scene: ScenePlan, panel: PanelPlan) -> str:
    character_text = "; ".join(
        CHARACTER_PROMPTS.get(ch.name, f"{ch.name}, consistent manga character design")
        for ch in panel.characters
    )
    camera_text = {
        "wide": "wide establishing shot",
        "medium": "medium cinematic shot",
        "close_up": "close-up facial expression shot",
        "over_shoulder": "over-the-shoulder composition",
    }[panel.camera.shot]
    props = ", ".join(panel.props)
    return (
        f"{scene.episode_title}. Scenario: {scene.scenario}\n"
        f"Panel {panel.panel_number}: {panel.action}\n"
        f"Characters: {character_text}\n"
        f"Setting and props: {panel.location}, {props}\n"
        f"Camera: {camera_text}, {panel.camera.angle_degrees:g} degree angle, target {panel.camera.target}\n"
        "Style: detailed black-and-white manga panel, clean ink lineart, screentones, dramatic shadows, "
        "professional manga background detail, expressive faces, correct school clothes, no text inside the art, "
        "no speech bubbles, preserve the rough composition from the input image."
    )


def _panel_to_control_dict(scene: ScenePlan, panel: PanelPlan) -> dict[str, Any]:
    camera_words = {
        "wide": "wide establishing classroom room 35 degrees",
        "medium": "medium classroom shot 35 degrees",
        "close_up": "close-up face expression classroom",
        "over_shoulder": "over shoulder classroom 35 degrees",
    }[panel.camera.shot]
    speaker = next((line.speaker for line in panel.dialogue if line.speaker), panel.characters[0].name if panel.characters else "Narrator")
    speech = " ".join(line.text for line in panel.dialogue)
    return {
        "context": f"{scene.episode_title}. {panel.title}. {panel.action}",
        "scene": [panel.location, camera_words, *panel.props],
        "speaker": speaker,
        "speech": speech,
    }


def _wrap_dialogue(text: str) -> str:
    return "\n".join(textwrap.wrap(text, width=24, break_long_words=False)) or text


def _character_bubble_boxes(panel: PanelPlan, image_size: tuple[int, int]) -> list[BubbleCharacter]:
    width, height = image_size
    if not panel.characters:
        return []
    xs = [ch.position[0] for ch in panel.characters]
    min_x = min(xs)
    max_x = max(xs)
    span = max(0.1, max_x - min_x)
    boxes = []
    for ch in panel.characters:
        norm_x = 0.5 if span <= 0.1 else (ch.position[0] - min_x) / span
        center_x = int(width * (0.28 + norm_x * 0.44))
        head_y = int(height * 0.32)
        body_top = int(height * 0.24)
        body_bottom = int(height * 0.72)
        box_w = int(width * 0.16)
        boxes.append(
            BubbleCharacter(
                name=ch.name,
                bbox=(center_x - box_w // 2, body_top, center_x + box_w // 2, body_bottom),
                head=(center_x, head_y),
            )
        )
    return boxes


def _draw_dialogue(config: Config, styler: ManhwaStyler, image: Image.Image, panel_plan: PanelPlan) -> Image.Image:
    lines = panel_plan.dialogue
    if getattr(config.bubbles, "backend", "internal") == "manhwa_bubbles":
        try:
            return apply_manhwa_bubbles(
                image,
                [
                    BubbleLine(
                        speaker=line.speaker,
                        text=line.text,
                        kind=line.kind,
                        position_hint="left" if line.position[0] < image.width // 2 else "right",
                    )
                    for line in lines
                ],
                characters=_character_bubble_boxes(panel_plan, image.size),
                config=config,
                panel_id=panel_plan.panel_number,
                seed=panel_plan.seed,
            )
        except Exception:
            if not getattr(config.bubbles, "fallback_to_internal", True):
                raise

    panel = image.convert("RGB")
    for index, line in enumerate(lines):
        x, y = line.position
        if index:
            y += index * 96
        panel = styler.add_text(
            panel,
            _wrap_dialogue(line.text),
            speaker=line.speaker,
            bubble_type=line.kind,
            position=(x, y),
        )
    return panel


def _fallback_manga_from_control(control: Image.Image, error: Exception) -> Image.Image:
    img = control.convert("L").convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle((18, img.height - 82, img.width - 18, img.height - 18), fill="white", outline="black", width=2)
    draw.text((30, img.height - 68), "Image API unavailable; using control render.", fill="black")
    return img


def _save_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _prepare_demo_config(config: Config, output_dir: str, image_model: str) -> None:
    config.output.output_dir = output_dir
    config.scenario.protagonist = "Kai"
    config.scenario.antagonist = "Maria"
    config.model.image_backend = "api"
    config.image_api.provider = "cloudflare"
    config.image_api.model = image_model
    config.image_api.cloudflare_use_multipart = False
    config.blender.enabled = True
    config.blender.fallback_to_pil = True
    config.blender.render_width = 768
    config.blender.render_height = 1024
    config.image_api.size = f"{config.blender.render_width}x{config.blender.render_height}"
    config.style_normalization.enabled = True
    config.style_normalization.width = config.blender.render_width
    config.style_normalization.height = config.blender.render_height


def _normalization_settings(config: Config) -> StyleNormalizationSettings:
    norm = getattr(config, "style_normalization", None)
    return StyleNormalizationSettings(
        style_id=getattr(norm, "style_id", "manga_bw_v1"),
        width=int(getattr(norm, "width", 768)),
        height=int(getattr(norm, "height", 1024)),
        gamma=float(getattr(norm, "gamma", 0.96)),
        contrast=float(getattr(norm, "contrast", 1.08)),
        autocontrast_cutoff=float(getattr(norm, "autocontrast_cutoff", 1.0)),
        sharpen_radius=float(getattr(norm, "sharpen_radius", 1.2)),
        sharpen_percent=int(getattr(norm, "sharpen_percent", 110)),
        sharpen_threshold=int(getattr(norm, "sharpen_threshold", 3)),
        grain_strength=float(getattr(norm, "grain_strength", 0.02)),
        screentone_style=getattr(norm, "screentone_style", "dots_medium"),
        page_autocontrast_cutoff=float(getattr(norm, "page_autocontrast_cutoff", 0.3)),
        page_contrast=float(getattr(norm, "page_contrast", 1.02)),
    )


def generate_demo(
    scenario: str,
    config: Config | None = None,
    *,
    output_dir: str = "output",
    image_model: str = "@cf/runwayml/stable-diffusion-v1-5-img2img",
) -> str:
    """Run the v0.1 three-panel manga demo and return the final page path."""
    config = config or Config.from_env()
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    _prepare_demo_config(config, output_dir, image_model)

    scene_plan = build_default_scene_plan(scenario)
    _save_json(os.path.join(output_dir, "scene_plan.json"), scene_plan.model_dump(mode="json"))

    styler = ManhwaStyler(font_path=(config.style.font_paths or [None])[0])
    norm_settings = _normalization_settings(config)
    style_reference_path = Path(getattr(config.style_normalization, "reference_path", "references/manga_style_reference.png"))
    explicit_reference_exists = style_reference_path.exists()
    reference_image = Image.open(style_reference_path).convert("RGB") if explicit_reference_exists else None
    panel_records: list[dict[str, Any]] = []
    manga_panels: list[Image.Image] = []

    for panel in scene_plan.panels:
        control_result = render_control_image(config, _panel_to_control_dict(scene_plan, panel), panel.panel_number, output_dir)
        if control_result is None:
            raise RuntimeError("Blender control rendering is disabled and no fallback control image was produced.")

        rough_path = os.path.join(output_dir, f"panel_{panel.panel_number:02d}_blender.png")
        raw_path = os.path.join(output_dir, f"panel_{panel.panel_number:02d}_manga_raw.png")
        normalized_path = os.path.join(output_dir, f"panel_{panel.panel_number:02d}_manga_normalized.png")
        manga_path = os.path.join(output_dir, f"panel_{panel.panel_number:02d}_manga.png")
        metadata_path = os.path.join(output_dir, f"panel_{panel.panel_number:02d}_metadata.json")
        shutil.copyfile(control_result.image_path, rough_path)

        prompt = build_panel_prompt(scene_plan, panel)
        api_error = None
        try:
            manga_img = generate_image_with_api(
                config,
                prompt,
                DEMO_NEGATIVE_PROMPT,
                reference_image=control_result.image,
                width=config.blender.render_width,
                height=config.blender.render_height,
                strength=0.45,
                guidance=7.5,
                seed=panel.seed,
                num_steps=24,
            )
        except Exception as exc:
            api_error = str(exc)
            manga_img = _fallback_manga_from_control(control_result.image, exc)

        manga_img.save(raw_path, quality=config.output.image_quality)

        metadata = {
            "panel": panel.model_dump(mode="json"),
            "prompt": prompt,
            "negative_prompt": DEMO_NEGATIVE_PROMPT,
            "control": {
                "source": control_result.source,
                "image_path": rough_path,
                "metadata_path": control_result.metadata_path,
                "fallback_reason": control_result.fallback_reason,
                "scene": control_result.scene,
            },
            "image_api": {
                "provider": config.image_api.provider,
                "model": config.image_api.model,
                "size": config.image_api.size,
                "strength": 0.45,
                "guidance": 7.5,
                "seed": panel.seed,
                "num_steps": 24,
                "api_error": api_error,
            },
        }
        panel_records.append(
            {
                "panel": panel,
                "raw_image": manga_img,
                "raw_path": raw_path,
                "normalized_path": normalized_path,
                "manga_path": manga_path,
                "metadata_path": metadata_path,
                "metadata": metadata,
            }
        )

    if reference_image is None and panel_records:
        reference_image = panel_records[0]["raw_image"]

    reference_metrics = None
    if explicit_reference_exists:
        reference_metrics = calculate_metrics(style_reference_path)

    for record in panel_records:
        normalized_img = normalize_panel_image(record["raw_image"], reference_image, norm_settings)
        normalized_img.save(record["normalized_path"], quality=config.output.image_quality)

        final_panel = _draw_dialogue(config, styler, normalized_img, record["panel"])
        final_panel.save(record["manga_path"], quality=config.output.image_quality)
        manga_panels.append(final_panel)

        metrics_path = Path(record["normalized_path"])
        metrics = calculate_metrics(metrics_path)
        needs_reprocess = should_reprocess(metrics, reference_metrics) if reference_metrics is not None else False
        record["metadata"]["style_normalization"] = {
            "enabled": True,
            "settings": norm_settings.to_dict(),
            "reference_path": str(style_reference_path) if explicit_reference_exists else record["raw_path"],
            "reference_is_explicit": explicit_reference_exists,
            "raw_path": record["raw_path"],
            "normalized_path": record["normalized_path"],
            "metrics": metrics.__dict__,
            "needs_reprocess": needs_reprocess,
        }
        record["metadata"]["bubbles"] = {
            "backend": getattr(config.bubbles, "backend", "internal"),
            "project_path": getattr(config.bubbles, "project_path", "../bubble"),
            "fallback_to_internal": getattr(config.bubbles, "fallback_to_internal", True),
        }
        _save_json(record["metadata_path"], record["metadata"])

    page_path = os.path.join(output_dir, "page_01.png")
    pre_page_normalization_path = os.path.join(output_dir, "page_01_before_page_normalization.png")
    ManhwaAssembler.assemble_panels(
        manga_panels,
        output_path=pre_page_normalization_path,
        panel_gap=config.style.panel_gap,
        background_color=config.style.canvas_bg,
    )
    normalize_completed_page(Path(pre_page_normalization_path), Path(page_path), norm_settings)
    return page_path
