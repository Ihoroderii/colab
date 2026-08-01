"""Package runner for the manhwa generator.
Run with: python -m manga_ai
"""
from __future__ import annotations
import os, datetime, json, random, logging
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageOps

from ..config import Config
from ..cli import parse_args, update_config_from_args
from ..pipelines.diffusion import get_cached_img2img_pipeline, get_cached_pipeline, select_device_and_dtype
from ..pipelines.scenario import generate_scenario, synthesize_prose_story, generate_story, panels_from_story
from ..pipelines.blender_control import render_control_image
from ..pipelines.image_api import generate_image_with_api
from ..pipelines.llm_api import create_llm_client
from ..effects.style import ManhwaStyler
from ..effects.text import ManhwaEffects
from ..integrations.bubbles import BubbleCharacter, BubbleLine, apply_manhwa_bubbles
from ..pipelines.assemble import ManhwaAssembler
from ..utils.prompts import get_manhwa_prompts
from ..utils.validator import PanelValidator
from ..utils.export import export_webtoon
from ..utils.detect import detect_faces_and_people
from ..post_processing import (
    StyleNormalizationSettings,
    normalize_completed_page,
    normalize_panel_image,
)

logger = logging.getLogger(__name__)


def _style_normalization_settings(config: Config) -> StyleNormalizationSettings:
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


def _apply_reference_style_normalization(config: Config, image: Image.Image) -> Image.Image:
    if not getattr(config.style_normalization, "enabled", False):
        return image
    reference_path = Path(getattr(config.style_normalization, "reference_path", "references/manga_style_reference.png"))
    if not reference_path.exists():
        logger.warning("Style normalization enabled but reference image not found: %s", reference_path)
        return image
    try:
        result = normalize_panel_image(
            image,
            Image.open(reference_path).convert("RGB"),
            _style_normalization_settings(config),
        )
        logger.info("Applied style normalization using %s", reference_path)
        return result
    except Exception as e:
        logger.warning("Style normalization failed, using unnormalized panel: %s", e)
        return image


def _apply_configured_bubbles(
    config: Config,
    styler: ManhwaStyler,
    image: Image.Image,
    text: str,
    speaker: str | None,
    bubble_type: str,
    position: tuple[int, int],
) -> Image.Image:
    if getattr(config.bubbles, "backend", "internal") == "manhwa_bubbles":
        try:
            x, y = position
            head_x = max(20, min(image.width - 20, x + min(180, image.width // 4)))
            head_y = max(80, min(image.height - 80, y + min(260, image.height // 3)))
            box_w = max(90, image.width // 7)
            box_h = max(220, image.height // 3)
            characters = []
            if speaker and speaker.lower() not in ("narrator", "narration"):
                characters.append(
                    BubbleCharacter(
                        name=speaker,
                        bbox=(
                            max(0, head_x - box_w // 2),
                            max(0, head_y - 40),
                            min(image.width, head_x + box_w // 2),
                            min(image.height, head_y + box_h),
                        ),
                        head=(head_x, head_y),
                    )
                )
            return apply_manhwa_bubbles(
                image,
                [
                    BubbleLine(
                        speaker=speaker,
                        text=text,
                        kind=bubble_type,
                        position_hint="left" if position[0] < image.width // 2 else "right",
                    )
                ],
                characters=characters,
                config=config,
            )
        except Exception as e:
            logger.warning("manhwa_bubbles backend failed, using internal bubble renderer: %s", e)
            if not getattr(config.bubbles, "fallback_to_internal", True):
                raise
    return styler.add_text(image, text, speaker=speaker, bubble_type=bubble_type, position=position)


def _init_logging():
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _init_config() -> Config:
    cfg = Config.from_env()
    args = parse_args()
    return update_config_from_args(cfg, args)


def _infer_global_style(config: Config, scenes) -> dict:
    # Analyze tags and tone to set style params
    tone = (getattr(config.scenario, "tone", "") or "").lower()
    tags = []
    for sc in scenes:
        sc_tags = sc.get("scene", [])
        if isinstance(sc_tags, list):
            tags.extend([str(t).lower() for t in sc_tags])
        elif isinstance(sc_tags, str):
            tags.extend([w.strip().lower() for w in sc_tags.split(",")])
    tags_set = set(tags)

    style = {}
    # Background color heuristics
    if any(k in tags_set for k in ["night", "neon", "rain", "shadow"]) or tone in ("dark","dramatic","horror"):
        style["canvas_bg"] = "#0f0f12"
    else:
        style["canvas_bg"] = "#ffffff"

    # Panel gap heuristics
    if "action" in (getattr(config.scenario, "genre", "").lower()):
        style["panel_gap"] = 48
    else:
        style["panel_gap"] = 40

    # Lighting heuristics
    if tone in ("dramatic","dark","horror") or any(k in tags_set for k in ["storm","night","shadow","neon"]):
        style["apply_dramatic_lighting"] = True
        style["lighting_style"] = "dramatic"
        style["lighting_intensity"] = 0.55
    else:
        style["apply_dramatic_lighting"] = False

    # Jitter heuristics (more variation for action)
    style["panel_width_jitter"] = 0.18 if "action" in (getattr(config.scenario, "genre", "").lower()) else 0.12

    return style


def _text_to_pose(scene_text):
    return [(250,100),(250,200),(220,300),(280,300),(210,400),(290,400)]


def _draw_pose(keypoints, size=512):
    skeleton = np.ones((size, size, 3), dtype=np.uint8) * 255
    connections = [(0,1),(1,2),(1,3),(2,4),(3,5)]
    for s, e in connections:
        if s < len(keypoints) and e < len(keypoints):
            cv2.line(skeleton, keypoints[s], keypoints[e], (0,0,0), 4)
    for p in keypoints:
        cv2.circle(skeleton, p, 6, (0,0,255), -1)
    return Image.fromarray(cv2.cvtColor(skeleton, cv2.COLOR_BGR2RGB))


def _round_to_multiple(x: int, base: int = 8) -> int:
    if base <= 1:
        return x
    return max(base, int(x // base * base))


def _resample_lanczos():
    return getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def _load_reference_image(path: str | None) -> Image.Image | None:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reference image does not exist: {path}")
    try:
        img = Image.open(path).convert("RGB")
        logger.info("Loaded reference image: %s size=%s", path, img.size)
        return img
    except Exception as e:
        logger.warning("Could not load reference image %s: %s", path, e)
        return None


def _prepare_reference_image(reference_image: Image.Image, size: tuple[int, int], resize_mode: str) -> Image.Image:
    width = _round_to_multiple(max(256, int(size[0])), 8)
    height = _round_to_multiple(max(256, int(size[1])), 8)
    resample = _resample_lanczos()
    mode = (resize_mode or "fit").lower()
    if mode == "crop":
        return ImageOps.fit(reference_image, (width, height), method=resample, centering=(0.5, 0.5))
    contained = ImageOps.contain(reference_image, (width, height), method=resample)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(contained, ((width - contained.width) // 2, (height - contained.height) // 2))
    return canvas


def _estimate_base_width(style_cfg, scene_prompt: str, speech_text: str) -> int:
    # Heuristic:
    # - Start from declared base (default 800)
    # - Increase width for long scenes or long speech to keep readability and layout balance
    base = getattr(style_cfg, "panel_width", 800)
    scene_len = len(scene_prompt or "")
    speech_len = len(speech_text or "")
    # Scene contribution: up to +25% around 600 chars
    scene_scale = 1.0 + min(scene_len / 600.0, 1.0) * 0.25
    # Speech contribution: up to +20% around 180 chars
    speech_scale = 1.0 + min(speech_len / 180.0, 1.0) * 0.20
    width = int(base * scene_scale * speech_scale)
    return max(384, min(1400, width))


def _is_sfx_text(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return False
    no_space = s.replace(" ", "")
    # short, all-caps onomatopoeia
    if len(no_space) <= 12 and no_space.upper() == no_space and any(ch.isalpha() for ch in no_space):
        return True
    common = {"boom","bang","clang","slam","thud","crack","wham","pow","zap","whoosh","bam","krak"}
    return s.lower() in common


def _infer_bubble_type(speaker: str, speech: str, tone: str, scene_tags=None) -> str:
    s = (speech or "").strip()
    t = (tone or "").lower()
    tags = []
    if scene_tags is not None:
        if isinstance(scene_tags, list):
            tags = [str(x).lower() for x in scene_tags]
        elif isinstance(scene_tags, str):
            tags = [w.strip().lower() for w in scene_tags.split(",")]

    # Explicit narration speaker
    if speaker and speaker.lower() in ("narrator","narration"):
        return "narration"

    # SFX detection first
    if _is_sfx_text(s) or any(k in tags for k in ["sfx","impact","explosion","hit","crash","bang"]):
        return "sfx"

    # Shout for exclamations, all caps, or action-heavy tags
    if s.endswith("!") or s.isupper() or any(k in tags for k in ["action","fight","battle","chase","yell","scream"]):
        return "shout"

    # Whisper for dramatic/quiet tones or stealthy tags with ellipses
    if (t in ("dramatic","dark","horror","mysterious") or any(k in tags for k in ["stealth","quiet","night"])) and "..." in s:
        return "whisper"

    # Thought bubble indicated by parentheses or tags
    if s.startswith("(") and s.endswith(")") or any(k in tags for k in ["thought","internal","monologue"]):
        return "thought"
    return "speech"


def _generate_panel(
    config: Config,
    context: str,
    scene_prompt: str,
    speaker: str,
    speech_text: str,
    reference_image: Image.Image | None = None,
    reference_source: str | None = None,
    img2img_strength: float | None = None,
):
    keypoints = _text_to_pose(scene_prompt)
    pose_image = _draw_pose(keypoints)

    prompt, negative_prompt = get_manhwa_prompts(context, scene_prompt)

    # Decide target sizing strategy
    square_mode = bool(getattr(config.style, "square_panels", False))

    # Compute base width (auto or fixed)
    if getattr(config.style, "auto_panel_width", True):
        base_width = _estimate_base_width(config.style, scene_prompt, speech_text)
    else:
        base_width = getattr(config.style, "panel_width", 800)

    # Apply jitter to target width to introduce natural variation
    jitter = float(getattr(config.style, "panel_width_jitter", 0.0) or 0.0)
    jitter = max(0.0, min(0.95, jitter))
    if jitter > 0:
        # sample in [1-jitter, 1+jitter]
        scale = 1.0 + (random.random() * 2 * jitter - jitter)
        target_width = max(256, int(base_width * scale))
    else:
        target_width = base_width

    # If square mode: set generation width/height to square, accounting for border width
    gen_kwargs = {}
    if square_mode:
        # Prefer a fixed runtime side if provided (same size across panels)
        runtime_side = getattr(config.style, "runtime_square_size", None)
        if runtime_side:
            side = int(runtime_side)
        else:
            side_cfg = getattr(config.style, "square_size", None)
            side = int(side_cfg) if side_cfg else int(target_width)
        bw = int(getattr(config.style, "panel_border_width", 0) or 0)
        gen_side = max(256, side - 2 * max(0, bw))
        # Align to multiples of 8 for SD pipelines
        gen_side = _round_to_multiple(gen_side, 8)
        gen_kwargs.update({"width": gen_side, "height": gen_side})

    seed = None
    effective_strength = None
    if getattr(config.model, "image_backend", "local") == "api":
        result_img = generate_image_with_api(
            config,
            prompt=prompt,
            negative_prompt=negative_prompt,
            reference_image=reference_image,
        )
        generation_mode = "api_img2img" if reference_image is not None else "api_txt2img"
        if reference_image is not None:
            effective_strength = float(
                img2img_strength
                if img2img_strength is not None
                else getattr(config.reference, "img2img_strength", 0.38) or 0.38
            )
    else:
        if reference_image is not None:
            pipeline, (device, dtype) = get_cached_img2img_pipeline(
                config.model.stable_diffusion_model,
                config.model.device,
                (config.model.REMOVED_TOKENtoken or None),
                getattr(config.model, "fallback_diffusion_model", "runwayml/stable-diffusion-v1-5"),
            )
        else:
            pipeline, (device, dtype) = get_cached_pipeline(
                config.model.stable_diffusion_model,
                config.model.device,
                (config.model.REMOVED_TOKENtoken or None),
                getattr(config.model, "fallback_diffusion_model", "runwayml/stable-diffusion-v1-5"),
            )

        seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=device).manual_seed(seed)

        if reference_image is not None:
            if square_mode:
                ref_size = (gen_kwargs.get("width", target_width), gen_kwargs.get("height", target_width))
            else:
                ratio = reference_image.height / max(1, reference_image.width)
                ref_size = (target_width, max(256, int(target_width * ratio)))
            conditioned_image = _prepare_reference_image(
                reference_image,
                ref_size,
                getattr(config.reference, "resize_mode", "fit"),
            )
            effective_strength = float(
                img2img_strength
                if img2img_strength is not None
                else getattr(config.reference, "img2img_strength", 0.38) or 0.38
            )
            effective_strength = max(0.0, min(1.0, effective_strength))
            result_img = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=conditioned_image,
                strength=effective_strength,
                guidance_scale=getattr(config.generation, "guidance_scale", 7.5),
                num_inference_steps=getattr(config.generation, "num_inference_steps", 50),
                generator=generator,
            ).images[0]
            generation_mode = "img2img"
        else:
            result_img = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=getattr(config.generation, "guidance_scale", 7.5),
                num_inference_steps=getattr(config.generation, "num_inference_steps", 50),
                generator=generator,
                **gen_kwargs,
            ).images[0]
            generation_mode = "txt2img"

    styler = ManhwaStyler()
    # If not square mode, do the classic aspect-based resize. Square mode already sized at generation.
    if not square_mode:
        result_img = styler.adjust_panel(result_img, target_width=target_width)
    result_img = styler.apply_style(result_img)
    result_img = _apply_reference_style_normalization(config, result_img)
    # Optional square padding
    if getattr(config.style, "square_panels", False):
        try:
            side = int(getattr(config.style, "square_size", 0) or 0)
            if side <= 0:
                side = int(target_width)
            fill = getattr(config.style, "square_fill_color", "#ffffff")
            result_img = ManhwaStyler.pad_to_square(result_img, size=side, fill_color=fill)
        except Exception as e:
            logger.debug(f"Square pad failed: {e}")
    # Optional lighting effect
    if getattr(config.style, "apply_dramatic_lighting", False):
        try:
            result_img = ManhwaEffects.add_dramatic_lighting(
                result_img,
                style=getattr(config.style, "lighting_style", "dramatic"),
                intensity=float(getattr(config.style, "lighting_intensity", 0.5)),
            )
        except Exception as e:
            logger.warning(f"Lighting effect failed, continuing without it: {e}")
    # Derive simple tags from scene_prompt text for bubble inference
    _tags = []
    if scene_prompt:
        if "," in scene_prompt:
            _tags = [w.strip().lower() for w in scene_prompt.split(",") if w.strip()]
        else:
            _tags = [w.strip().lower() for w in scene_prompt.split() if w.strip()]

    bubble_type = _infer_bubble_type(speaker, speech_text, getattr(config.scenario, "tone", ""), _tags)

    # Detection-assisted bubble placement
    position = (20, 20)
    if getattr(config.style, "auto_bubble_placement", True):
        try:
            faces, people = detect_faces_and_people(result_img)
            W, H = result_img.size
            # Prefer the largest face; otherwise use largest person; fallback top-left
            target = None
            if faces:
                target = max(faces, key=lambda r: r[2] * r[3])
            elif people:
                target = max(people, key=lambda r: r[2] * r[3])
            if target:
                x, y, w, h = target
                # Place bubble slightly above-left of the target box
                px = max(20, x - 10)
                py = max(20, y - 40)
                position = (px, py)

            # Optional debug overlay
            if getattr(config.style, "draw_detection_debug", False):
                dbg = result_img.copy()
                draw = ImageDraw.Draw(dbg)
                for (x, y, w, h) in faces:
                    draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=3)
                for (x, y, w, h) in people:
                    draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=3)
                result_img = dbg
        except Exception as e:
            logger.debug(f"Detection failed or unavailable: {e}")

    result_img = _apply_configured_bubbles(
        config,
        styler,
        result_img,
        speech_text,
        speaker,
        bubble_type,
        position,
    )
    # Add border around panel if configured
    try:
        bw = int(getattr(config.style, "panel_border_width", 0) or 0)
        bc = getattr(config.style, "panel_border_color", "#000000")
        if bw > 0:
            result_img = ManhwaStyler.add_border(result_img, border_width=bw, border_color=bc)
    except Exception as _e:
        logger.debug(f"Border add failed: {_e}")
    # Final square enforcement after all overlays/border
    if getattr(config.style, "square_panels", False):
        try:
            fill = getattr(config.style, "square_fill_color", "#ffffff")
            # If a fixed runtime size is set, use it; otherwise use current max dimension
            final_side = int(getattr(config.style, "runtime_square_size", 0) or 0) or max(result_img.size)
            result_img = ManhwaStyler.pad_to_square(result_img, size=final_side, fill_color=fill)
        except Exception as e:
            logger.debug(f"Final square enforce failed: {e}")

    return result_img, {
        "seed": seed,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "generation_mode": generation_mode,
        "reference_source": reference_source,
        "img2img_strength": effective_strength,
        "bubble_backend": getattr(config.bubbles, "backend", "internal"),
    }


# ---------------------------------------------------------------------------
# PanelSpec-native rendering
# ---------------------------------------------------------------------------

def render_panel_spec(
    config: Config,
    panel_spec: dict,
    ip_adapter_images: list = None,
) -> tuple:
    """Render a single panel from a PanelSpec dict.

    This is the first-class entry point for manga-ai-bot to consume
    the shared PanelSpec contract produced by manga_prep.  It uses
    prompt_positive / prompt_negative directly instead of rebuilding
    prompts through get_manhwa_prompts.

    Args:
        config: manga-ai-bot Config.
        panel_spec: Dict with at least prompt_positive and prompt_negative.
        ip_adapter_images: Optional reference images for IP-Adapter conditioning.

    Returns:
        (PIL.Image, metadata_dict)
    """
    prompt = panel_spec.get("prompt_positive", "")
    negative = panel_spec.get("prompt_negative", "")
    if not prompt:
        raise ValueError("panel_spec must contain a non-empty prompt_positive")

    control_result = None
    if getattr(config.blender, "enabled", False):
        control_panel = {
            "context": panel_spec.get("location_id") or panel_spec.get("setting") or "",
            "scene": [
                panel_spec.get("shot_type", ""),
                panel_spec.get("camera_angle", ""),
                prompt,
            ],
            "speaker": "",
            "speech": "",
        }
        try:
            panel_number = int(str(panel_spec.get("panel_id") or "1").split("_")[-1])
        except Exception:
            panel_number = 1
        control_result = render_control_image(config, control_panel, panel_number, config.output.output_dir)

    seed = random.randint(0, 2**32 - 1)
    if getattr(config.model, "image_backend", "local") == "api":
        result_img = generate_image_with_api(
            config,
            prompt=prompt,
            negative_prompt=negative,
            reference_image=control_result.image if control_result is not None else None,
        )
        generation_mode = "api_img2img" if control_result is not None else "api_txt2img"
    elif control_result is not None:
        pipeline, (device, dtype) = get_cached_img2img_pipeline(
            config.model.stable_diffusion_model,
            config.model.device,
            (config.model.REMOVED_TOKENtoken or None),
            getattr(config.model, "fallback_diffusion_model", "runwayml/stable-diffusion-v1-5"),
        )
        generator = torch.Generator(device=device).manual_seed(seed)
        conditioned_image = _prepare_reference_image(control_result.image, (768, 1024), "fit")
        result_img = pipeline(
            prompt=prompt,
            negative_prompt=negative,
            image=conditioned_image,
            strength=max(0.0, min(1.0, float(getattr(config.blender, "img2img_strength", 0.34)))),
            guidance_scale=getattr(config.generation, "guidance_scale", 7.5),
            num_inference_steps=getattr(config.generation, "num_inference_steps", 50),
            generator=generator,
        ).images[0]
        generation_mode = "img2img"
    else:
        pipeline, (device, dtype) = get_cached_pipeline(
            config.model.stable_diffusion_model,
            config.model.device,
            (config.model.REMOVED_TOKENtoken or None),
            getattr(config.model, "fallback_diffusion_model", "runwayml/stable-diffusion-v1-5"),
        )
        generator = torch.Generator(device=device).manual_seed(seed)
        gen_kwargs = {"width": 768, "height": 1024}

        # IP-Adapter conditioning if pipeline supports it and images are provided
        if ip_adapter_images and hasattr(pipeline, "set_ip_adapter_scale"):
            try:
                pipeline.set_ip_adapter_scale(0.6)
                if len(ip_adapter_images) == 1:
                    gen_kwargs["ip_adapter_image"] = ip_adapter_images[0]
                else:
                    gen_kwargs["ip_adapter_image"] = ip_adapter_images
            except Exception as e:
                logger.debug("IP-Adapter not available: %s", e)

        result_img = pipeline(
            prompt=prompt,
            negative_prompt=negative,
            guidance_scale=getattr(config.generation, "guidance_scale", 7.5),
            num_inference_steps=getattr(config.generation, "num_inference_steps", 50),
            generator=generator,
            **gen_kwargs,
        ).images[0]
        generation_mode = "txt2img"

    styler = ManhwaStyler()
    result_img = styler.apply_style(result_img)
    result_img = _apply_reference_style_normalization(config, result_img)

    meta = {
        "seed": seed,
        "prompt": prompt,
        "negative_prompt": negative,
        "panel_id": panel_spec.get("panel_id"),
        "generation_mode": generation_mode,
        "shot_type": panel_spec.get("shot_type"),
        "camera_angle": panel_spec.get("camera_angle"),
        "characters": [c.get("character_id", "") for c in panel_spec.get("characters", [])],
        "location_id": panel_spec.get("location_id", ""),
        "control_image": control_result.image_path if control_result is not None else None,
        "control_source": control_result.source if control_result is not None else None,
    }
    return result_img, meta


def main():
    _init_logging()
    config = _init_config()
    os.makedirs(config.output.output_dir, exist_ok=True)
    logger.info("Image backend: %s", getattr(config.model, "image_backend", "local"))
    if getattr(config.model, "image_backend", "local") == "api":
        logger.info(
            "Using image API: provider=%s model=%s size=%s quality=%s",
            getattr(config.image_api, "provider", "openai"),
            getattr(config.image_api, "model", "gpt-image-1"),
            getattr(config.image_api, "size", "1024x1024"),
            getattr(config.image_api, "quality", "medium"),
        )
    else:
        logger.info(f"Using diffusion model: {config.model.stable_diffusion_model}")
    logger.info(f"Preferred device: {config.model.device}")
    try:
        logger.info(
            "Panel border: width=%s color=%s",
            getattr(config.style, "panel_border_width", 0),
            getattr(config.style, "panel_border_color", "#000000"),
        )
    except Exception:
        pass
    try:
        logger.info(
            "Square panels: %s | size=%s | fill=%s",
            getattr(config.style, "square_panels", False),
            getattr(config.style, "square_size", None),
            getattr(config.style, "square_fill_color", "#ffffff"),
        )
    except Exception:
        pass

    if getattr(config.style, "square_panels", False) and getattr(config.style, "same_panel_size", True):
        side = getattr(config.style, "square_size", None)
        if not side:
            side = getattr(config.style, "panel_width", 800)
        try:
            side = int(side)
        except Exception:
            side = 800
        config.style.runtime_square_size = side
        logger.info("Runtime square size fixed to: %s", side)

    reference_image = _load_reference_image(getattr(config.reference, "image_path", None))
    if reference_image is not None:
        logger.info(
            "Reference conditioning enabled: strength=%s resize_mode=%s",
            getattr(config.reference, "img2img_strength", 0.38),
            getattr(config.reference, "resize_mode", "fit"),
        )
    if getattr(config.blender, "enabled", False):
        logger.info(
            "Blender control enabled: executable=%s size=%sx%s strength=%s fallback=%s",
            getattr(config.blender, "executable", "blender"),
            getattr(config.blender, "render_width", 768),
            getattr(config.blender, "render_height", 1024),
            getattr(config.blender, "img2img_strength", 0.34),
            getattr(config.blender, "fallback_to_pil", True),
        )
    if getattr(config.style_normalization, "enabled", False):
        logger.info(
            "Style normalization enabled: reference=%s size=%sx%s",
            getattr(config.style_normalization, "reference_path", "references/manga_style_reference.png"),
            getattr(config.style_normalization, "width", 768),
            getattr(config.style_normalization, "height", 1024),
        )
    logger.info(
        "Bubble backend: %s (project_path=%s fallback=%s)",
        getattr(config.bubbles, "backend", "internal"),
        getattr(config.bubbles, "project_path", "../bubble"),
        getattr(config.bubbles, "fallback_to_internal", True),
    )

    try:
        client = create_llm_client(config)
        logger.info("LLM provider: %s", getattr(config.model, "llm_provider", "huggingface"))
    except Exception as e:
        client = None
        logger.warning("LLM client setup failed, using fallback story/scenario generation: %s", e)

    logger.info("Generating ~300-word story first...")
    story_txt = generate_story(config, client, target_words=300)
    # Save prose story
    story_path = os.path.join(config.output.output_dir, "story.txt")
    with open(story_path, "w", encoding="utf-8") as sf:
        sf.write(story_txt)
    logger.info(f"Saved story to {story_path}")
    # Print full story to logs with word count
    try:
        _wc = len(story_txt.split())
        logger.info("Story (~%d words):\n%s", _wc, story_txt)
    except Exception:
        logger.info("Story (raw): %s", story_txt)

    logger.info("Deriving panel scenario from story...")
    scenes = panels_from_story(config, client, story_txt)
    # Print the whole scenario to logs
    try:
        logger.info("Scenario JSON (full):\n%s", json.dumps(scenes, ensure_ascii=False, indent=2))
    except Exception:
        logger.info("Scenario (raw): %s", scenes)

    # Also save a synthesized panel-by-panel prose for reference
    story_panels_txt = synthesize_prose_story(config, scenes)
    story_panels_path = os.path.join(config.output.output_dir, "story_panels.txt")
    with open(story_panels_path, "w", encoding="utf-8") as sf:
        sf.write(story_panels_txt)
    logger.info(f"Saved panel-by-panel prose to {story_panels_path}")

    # Run log
    run_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_log_path = os.path.join(config.output.output_dir, f"run_log_{run_ts}.txt")
    with open(run_log_path, "w", encoding="utf-8") as f:
        f.write("=== Manhwa Generation Run Log ===\n")
        f.write(f"Timestamp: {run_ts}\n")
        f.write(f"Image backend: {getattr(config.model, 'image_backend', 'local')}\n")
        f.write(f"Model: {config.model.stable_diffusion_model}\n")
        f.write(f"Fallback model: {getattr(config.model, 'fallback_diffusion_model', None)}\n")
        f.write(f"Image API provider: {getattr(config.image_api, 'provider', None)}\n")
        f.write(f"Image API model: {getattr(config.image_api, 'model', None)}\n")
        f.write(f"Image API size: {getattr(config.image_api, 'size', None)}\n")
        f.write(f"Image API quality: {getattr(config.image_api, 'quality', None)}\n")
        f.write(f"Device: {config.model.device}\n")
        f.write(f"LLM Model: {config.model.llm_model}\n")
        f.write(f"Reference image: {getattr(config.reference, 'image_path', None)}\n")
        f.write(f"Img2img strength: {getattr(config.reference, 'img2img_strength', None)}\n")
        f.write(f"Blender control enabled: {getattr(config.blender, 'enabled', False)}\n")
        f.write(f"Blender executable: {getattr(config.blender, 'executable', None)}\n")
        f.write(f"Blender control strength: {getattr(config.blender, 'img2img_strength', None)}\n")
        # Story section
        try:
            _wc = len(story_txt.split())
        except Exception:
            _wc = 0
        f.write("\n-- Story (~{} words) --\n".format(_wc))
        f.write(story_txt)
        f.write("\n\n")
        f.write("\n-- Scenario --\n")
        f.write(json.dumps(scenes, ensure_ascii=False, indent=2))
        f.write("\n\n")

    # Optional validator
    validator = None
    if getattr(config, "validation", None) and config.validation.enabled:
        device, _ = select_device_and_dtype(config.model.device)
        validator = PanelValidator(
            device=device,
            blip_model_id=config.validation.blip_model,
            clip_model_id=config.validation.clip_model,
            threshold=config.validation.similarity_threshold,
        )
        validator.load()

    # Auto style based on scenario
    if getattr(config.style, "auto_style", True):
        inferred = _infer_global_style(config, scenes)
        config.style.canvas_bg = inferred.get("canvas_bg", config.style.canvas_bg)
        config.style.panel_gap = inferred.get("panel_gap", config.style.panel_gap)
        config.style.apply_dramatic_lighting = inferred.get("apply_dramatic_lighting", config.style.apply_dramatic_lighting)
        config.style.lighting_style = inferred.get("lighting_style", config.style.lighting_style)
        config.style.lighting_intensity = inferred.get("lighting_intensity", config.style.lighting_intensity)
        config.style.panel_width_jitter = inferred.get("panel_width_jitter", config.style.panel_width_jitter)

    panels, ref_img = [], None
    for i, sc in enumerate(scenes, 1):
        context = sc.get("context", "")
        scene = sc.get("scene", "")
        if isinstance(scene, list):
            scene = ", ".join(scene)
        speaker = sc.get("speaker", "Narrator")
        speech = sc.get("speech", "")

        panel_reference = reference_image
        panel_reference_source = getattr(config.reference, "image_path", None)
        panel_reference_strength = None
        control_result = None
        if getattr(config.blender, "enabled", False):
            try:
                control_result = render_control_image(config, sc, i, config.output.output_dir)
                if control_result is not None:
                    panel_reference = control_result.image
                    panel_reference_source = control_result.image_path
                    panel_reference_strength = float(getattr(config.blender, "img2img_strength", 0.34))
                    logger.info(
                        "Panel %s control image: %s (%s)",
                        i,
                        control_result.image_path,
                        control_result.source,
                    )
            except Exception as e:
                logger.warning("Panel %s control render failed, using normal reference path: %s", i, e)

        try:
            panel, meta = _generate_panel(
                config,
                context,
                scene,
                speaker,
                speech,
                reference_image=panel_reference,
                reference_source=panel_reference_source,
                img2img_strength=panel_reference_strength,
            )
        except Exception as e:
            logger.error(f"Panel {i} generation failed: {e}")
            panel = Image.new("RGB", (768, 1024), color=(32, 32, 32))
            draw = ImageDraw.Draw(panel)
            try:
                draw.text((30, 40), f"Panel {i} failed", fill=(240, 240, 240))
                draw.text((30, 90), f"{speaker}: {speech}", fill=(220, 220, 220))
            except Exception:
                pass
            meta = {
                "seed": None,
                "prompt": None,
                "negative_prompt": None,
                "generation_mode": "failed",
                "error": str(e),
            }

        if validator is not None and validator.available():
            passed, details = validator.validate(panel, ref_img, expected_traits=[speaker] if speaker else None)
            logger.info(f"Validation: passed={passed}, details={details}")
        if ref_img is None:
            ref_img = panel

        if config.output.save_individual_panels:
            panel_path = os.path.join(
                config.output.output_dir,
                config.output.panel_filename_template.format(i)
            )
            panel.save(panel_path, quality=config.output.image_quality)
            logger.info(f"Saved panel to {panel_path} (size={panel.size})")
        else:
            panel_path = None

        panels.append(panel)
        with open(run_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n-- Panel {i} --\n")
            f.write(f"Context: {context}\n")
            f.write(f"Scene: {scene}\n")
            f.write(f"Speaker: {speaker}\n")
            f.write(f"Speech: {speech}\n")
            f.write(f"Prompt: {meta.get('prompt')}\n")
            f.write(f"Negative: {meta.get('negative_prompt')}\n")
            f.write(f"Generation mode: {meta.get('generation_mode')}\n")
            if meta.get("reference_source"):
                f.write(f"Reference source: {meta.get('reference_source')}\n")
            if meta.get("img2img_strength") is not None:
                f.write(f"Img2img strength: {meta.get('img2img_strength')}\n")
            if control_result is not None:
                f.write(f"Control source: {control_result.source}\n")
                f.write(f"Control scene: {control_result.metadata_path}\n")
            if meta.get("error"):
                f.write(f"Error: {meta.get('error')}\n")
            f.write(f"Seed: {meta.get('seed')}\n")
            if panel_path:
                f.write(f"Output: {panel_path}\n")

    chapter_path = os.path.join(config.output.output_dir, config.output.chapter_filename)
    logger.info(f"Assembling final chapter to {chapter_path}")
    assembler = ManhwaAssembler()
    if panels:
        # Force white background for assembly
        final_img = assembler.assemble_panels(
            panels,
            output_path=chapter_path,
            panel_gap=getattr(config.style, "panel_gap", 40),
            background_color="#ffffff",
        )
    else:
        final_img = Image.new("RGB", (768, 1024), color=(20, 20, 20))
        final_img.save(chapter_path)
    if getattr(config.style_normalization, "enabled", False):
        try:
            normalize_completed_page(Path(chapter_path), Path(chapter_path), _style_normalization_settings(config))
            logger.info("Applied page-wide style normalization to %s", chapter_path)
        except Exception as e:
            logger.warning("Page-wide style normalization failed: %s", e)
    logger.info("Generation complete!")

    if getattr(config, "export", None) and config.export.enabled:
        export_dir = os.path.join(config.output.output_dir, config.export.output_subdir)
        try:
            chapter_img = Image.open(chapter_path)
            paths = export_webtoon(
                chapter_img,
                export_dir,
                basename=os.path.splitext(os.path.basename(chapter_path))[0],
                width=config.export.width,
                max_slice_height=config.export.max_slice_height,
                overlap=config.export.overlap,
                fmt=config.export.format,
                quality=config.export.quality,
            )
            logger.info(f"Exported {len(paths)} slices to {export_dir}")
            with open(run_log_path, "a", encoding="utf-8") as f:
                f.write("\n-- Export --\n")
                for p in paths:
                    f.write(f"Slice: {p}\n")
        except Exception as e:
            logger.error(f"Export failed: {e}")


if __name__ == "__main__":
    main()
