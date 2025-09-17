import os
import sys
import datetime
import json
import re
import random
import logging
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
try:
    from IPython.display import display  # type: ignore
except Exception:
    display = None  # type: ignore

# ------------------- Custom Modules -------------------
# Import via shim (works both pre- and post-migration)
from manhwa_utils import ManhwaStyler, ManhwaAssembler, get_manhwa_prompts  # type: ignore
from validator import PanelValidator  # shim that re-exports package class
from manhwa_effects import ManhwaEffects  # shim that re-exports package class
from export import export_webtoon  # shim that re-exports package function

# ------------------- AI/LLM -------------------
from openai import OpenAI
# Use packaged diffusion/scenario helpers
from manga_ai.pipelines.diffusion import get_cached_pipeline, select_device_and_dtype  # type: ignore
from manga_ai.pipelines.scenario import generate_scenario, synthesize_prose_story  # type: ignore
"""
Note: BLIP/CLIP validation imports were removed in this script revision because
the current flow doesn't use them. Re-add when a validator stage is implemented.
"""

from config import Config
from cli import parse_args, update_config_from_args


# =====================================================
# Setup Logging
# =====================================================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =====================================================
# Load Configuration
# =====================================================
def init_config():
    """Initialize configuration from environment and CLI args"""
    config = Config.from_env()
    args = parse_args()
    return update_config_from_args(config, args)

config = init_config()
logger.info(f"Using device: {config.model.device}")
logger.info(f"Using diffusion model: {config.model.stable_diffusion_model}")


# =====================================================
# OpenAI Client
# =====================================================
client = None
if config.model.REMOVED_TOKENtoken:
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=config.model.REMOVED_TOKENtoken,
    )


# =====================================================
# Pose Functions
# =====================================================
def text_to_pose(scene_text):
    # TODO: Replace with real text-to-pose model
    return [(250,100),(250,200),(220,300),(280,300),(210,400),(290,400)]

def draw_pose(keypoints, size=512):
    skeleton = np.ones((size, size, 3), dtype=np.uint8) * 255
    connections = [(0,1),(1,2),(1,3),(2,4),(3,5)]
    for s, e in connections:
        if s < len(keypoints) and e < len(keypoints):
            cv2.line(skeleton, keypoints[s], keypoints[e], (0,0,0), 4)
    for p in keypoints:
        cv2.circle(skeleton, p, 6, (0,0,255), -1)
    return Image.fromarray(cv2.cvtColor(skeleton, cv2.COLOR_BGR2RGB))


# =====================================================
# Speech/Text Effects
# =====================================================
def detect_face(img_pil):
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return (x, y, w, h)
    return None


def analyze_text_style(text: str) -> tuple[str, float]:
    text_lower = text.lower()
    if text.isupper() or text.count("!") > 1:
        return "shout", 1.0
    if any(word in text_lower for word in ["think", "wonder", "ponder"]):
        return "thought", 0.7
    if any(word in text_lower for word in ["boom", "crash", "bang"]):
        return "sound", 0.8
    return "normal", 0.5


def add_speech_effects(img_pil: Image.Image, text: str, speaker: str = None) -> Image.Image:
    effects = ManhwaEffects()
    face_pos = detect_face(img_pil)
    style, intensity = analyze_text_style(text)

    if face_pos:
        x, y, w, h = face_pos
        position = (x + w + 20, y)
    else:
        position = (50, 50)

    # Map our style labels to effect types supported by ManhwaEffects
    effect_type = {
        "shout": "emphasis",
        "thought": "thought",
        "sound": "sound",
        "normal": "emphasis",
    }.get(style, "emphasis")

    size = 30 if style == "shout" else 26
    result = effects.add_text_effects(img_pil, text, position=position, effect_type=effect_type, size=size)
    if speaker:
        result = effects.add_text_effects(result, speaker, position=(position[0], position[1]-25), effect_type="normal", size=24)

    return result


# =====================================================
# Manhwa Panel Generator
# =====================================================
def generate_manhwa_panel(context, scene_prompt, speaker, speech_text):
    # Pose
    keypoints = text_to_pose(scene_prompt)
    pose_image = draw_pose(keypoints)

    # Load pipeline (cached)
    pipeline, (device, dtype) = get_cached_pipeline(
        config.model.stable_diffusion_model,
        config.model.device,
        (config.model.REMOVED_TOKENtoken or None),
    )

    seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(device=device).manual_seed(seed)

    # TODO: Define this function properly
    # Manhwa-optimized prompts from utility
    prompt, negative_prompt = get_manhwa_prompts(context, scene_prompt)

    result_img = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=getattr(config.generation, "guidance_scale", 7.5),
        num_inference_steps=getattr(config.generation, "num_inference_steps", 50),
        generator=generator,
    ).images[0]

    # Apply styling
    styler = ManhwaStyler()
    result_img = styler.adjust_panel(result_img, target_width=getattr(config.style, "panel_width", 800))
    result_img = styler.apply_style(result_img)
    result_img = styler.add_text(result_img, speech_text, speaker=speaker)

    return result_img, {
        "seed": seed,
    }


# =====================================================
# Assembly
# =====================================================
def assemble_panels(panels, output_path="chapter.png"):
    assembler = ManhwaAssembler()
    if not panels:
        # Create a placeholder image to avoid crashing and inform the user
        placeholder = Image.new("RGB", (768, 1024), color=(20, 20, 20))
        draw = ImageDraw.Draw(placeholder)
        msg = "No panels generated. See run log for details."
        try:
            draw.text((30, 40), msg, fill=(240, 240, 240))
        except Exception:
            pass
        placeholder.save(output_path)
        final_img = placeholder
    else:
        final_img = assembler.assemble_panels(
            panels,
            output_path=output_path,
            panel_gap=getattr(config.style, "panel_gap", 40),
            background_color=getattr(config.style, "canvas_bg", "#ffffff"),
        )
    if display is not None:
        display(final_img)


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    os.makedirs(config.output.output_dir, exist_ok=True)

    logger.info("Generating manhwa scenario...")
    scenes = generate_scenario(config, client)
    # Save a prose story derived from scenes
    story_txt = synthesize_prose_story(config, scenes)
    story_path = os.path.join(config.output.output_dir, "story.txt")
    with open(story_path, "w", encoding="utf-8") as sf:
        sf.write(story_txt)
    logger.info(f"Saved story to {story_path}")
    # Prepare run log
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

    panels, ref_img = [], None
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
    for i, sc in enumerate(scenes, 1):
        context = sc.get("context", "")
        scene = sc.get("scene", "")
        # Ensure scene is string for prompt builder (join lists)
        if isinstance(scene, list):
            scene = ", ".join(scene)
        speaker = sc.get("speaker", "Narrator")
        speech = sc.get("speech", "")

        try:
            panel, meta = generate_manhwa_panel(context, scene, speaker, speech)
        except Exception as e:
            logger.error(f"Panel {i} generation failed: {e}")
            # Create a simple text-only placeholder panel to keep flow
            panel = Image.new("RGB", (768, 1024), color=(32, 32, 32))
            draw = ImageDraw.Draw(panel)
            try:
                draw.text((30, 40), f"Panel {i} failed", fill=(240, 240, 240))
                draw.text((30, 90), f"{speaker}: {speech}", fill=(220, 220, 220))
            except Exception:
                pass
            meta = {"seed": None}
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

        # Log per-panel details
        prompt, negative_prompt = get_manhwa_prompts(context, scene)
        with open(run_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n-- Panel {i} --\n")
            f.write(f"Context: {context}\n")
            f.write(f"Scene: {scene}\n")
            f.write(f"Speaker: {speaker}\n")
            f.write(f"Speech: {speech}\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Negative: {negative_prompt}\n")
            f.write(f"Seed: {meta.get('seed')}\n")
            if panel_path:
                f.write(f"Output: {panel_path}\n")

    chapter_path = os.path.join(config.output.output_dir, config.output.chapter_filename)
    logger.info(f"Assembling final chapter to {chapter_path}")
    assemble_panels(panels, chapter_path)
    logger.info("Generation complete! 🎉")

    # Export slicing for publishing
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
