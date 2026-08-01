from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .config import MangaStyleConfig


def create_tone_masks(image: Image.Image, config: MangaStyleConfig) -> dict[str, Image.Image]:
    grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
    return {
        "black": Image.fromarray(np.where(grayscale <= config.black_threshold, 255, 0).astype(np.uint8)),
        "dark_tone": Image.fromarray(
            np.where(
                (grayscale > config.black_threshold) & (grayscale <= config.dark_tone_threshold),
                255,
                0,
            ).astype(np.uint8)
        ),
        "light_tone": Image.fromarray(
            np.where(
                (grayscale > config.dark_tone_threshold) & (grayscale <= config.light_tone_threshold),
                255,
                0,
            ).astype(np.uint8)
        ),
    }


def dot_pattern(size: tuple[int, int], step: int, radius: int, offset: int = 0) -> Image.Image:
    width, height = size
    pattern = Image.new("L", size, 255)
    draw = ImageDraw.Draw(pattern)
    for y in range(offset, height + step, step):
        row_offset = (step // 2) if ((y // step) % 2) else 0
        for x in range(offset + row_offset, width + step, step):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=0)
    return pattern


def apply_screentones(image: Image.Image, config: MangaStyleConfig) -> Image.Image:
    base = image.convert("L")
    masks = create_tone_masks(base, config)
    solid_black = Image.new("L", base.size, 0)
    dark_pattern = dot_pattern(base.size, config.dark_tone_step, config.dark_tone_radius, offset=0)
    light_pattern = dot_pattern(base.size, config.light_tone_step, config.light_tone_radius, offset=3)

    toned = Image.composite(solid_black, base, masks["black"])
    toned = Image.composite(dark_pattern, toned, masks["dark_tone"])
    toned = Image.composite(light_pattern, toned, masks["light_tone"])
    return toned
