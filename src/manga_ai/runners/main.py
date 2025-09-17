"""Package runner for the manhwa generator.
Run with: python -m manga_ai
"""
from __future__ import annotations
import os, datetime, json, random, logging
from typing import Optional

import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw
from openai import OpenAI

from ..config import Config
from ..cli import parse_args, update_config_from_args
from ..pipelines.diffusion import get_cached_pipeline, select_device_and_dtype
from ..pipelines.scenario import generate_scenario, synthesize_prose_story, generate_story, panels_from_story
from ..effects.style import ManhwaStyler
from ..effects.text import ManhwaEffects
from ..pipelines.assemble import ManhwaAssembler
from ..utils.prompts import get_manhwa_prompts
from ..utils.validator import PanelValidator
from ..utils.export import export_webtoon
from ..utils.detect import detect_faces_and_people

logger = logging.getLogger(__name__)


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


def _generate_panel(config: Config, context: str, scene_prompt: str, speaker: str, speech_text: str):
    keypoints = _text_to_pose(scene_prompt)
    pose_image = _draw_pose(keypoints)

    pipeline, (device, dtype) = get_cached_pipeline(
        config.model.stable_diffusion_model,
        config.model.device,
        (config.model.REMOVED_TOKENtoken or None),
    )

    seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(device=device).manual_seed(seed)

    prompt, negative_prompt = get_manhwa_prompts(context, scene_prompt)
    result_img = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=getattr(config.generation, "guidance_scale", 7.5),
        num_inference_steps=getattr(config.generation, "num_inference_steps", 50),
        generator=generator,
    ).images[0]

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

    styler = ManhwaStyler()
    result_img = styler.adjust_panel(result_img, target_width=target_width)
    result_img = styler.apply_style(result_img)
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

    result_img = styler.add_text(result_img, speech_text, speaker=speaker, bubble_type=bubble_type, position=position)

    return result_img, {"seed": seed, "prompt": prompt, "negative_prompt": negative_prompt}


def main():
    _init_logging()
    config = _init_config()
    os.makedirs(config.output.output_dir, exist_ok=True)
    logger.info(f"Using diffusion model: {config.model.stable_diffusion_model}")
    logger.info(f"Preferred device: {config.model.device}")

    client = None
    if config.model.REMOVED_TOKENtoken:
        client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=config.model.REMOVED_TOKENtoken)

    logger.info("Generating ~300-word story first...")
    story_txt = generate_story(config, client, target_words=300)
    # Save prose story
    story_path = os.path.join(config.output.output_dir, "story.txt")
    with open(story_path, "w", encoding="utf-8") as sf:
        sf.write(story_txt)
    logger.info(f"Saved story to {story_path}")

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
        f.write(f"Model: {config.model.stable_diffusion_model}\n")
        f.write(f"Device: {config.model.device}\n")
        f.write(f"LLM Model: {config.model.llm_model}\n")
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

        try:
            panel, meta = _generate_panel(config, context, scene, speaker, speech)
        except Exception as e:
            logger.error(f"Panel {i} generation failed: {e}")
            panel = Image.new("RGB", (768, 1024), color=(32, 32, 32))
            draw = ImageDraw.Draw(panel)
            try:
                draw.text((30, 40), f"Panel {i} failed", fill=(240, 240, 240))
                draw.text((30, 90), f"{speaker}: {speech}", fill=(220, 220, 220))
            except Exception:
                pass
            meta = {"seed": None, "prompt": None, "negative_prompt": None}

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
            logger.info(f"Saved panel to {panel_path}")
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
