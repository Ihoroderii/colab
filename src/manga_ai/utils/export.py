"""Webtoon export slicer (moved to package)."""
from __future__ import annotations
from typing import List
import os
from PIL import Image


def resize_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image
    ratio = width / image.width
    height = int(image.height * ratio)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def slice_vertical(image: Image.Image, max_height: int, overlap: int) -> List[Image.Image]:
    slices: List[Image.Image] = []
    y = 0
    while y < image.height:
        box_bottom = min(y + max_height, image.height)
        crop = image.crop((0, y, image.width, box_bottom))
        slices.append(crop)
        if box_bottom == image.height:
            break
        y = box_bottom - overlap
    return slices


def export_webtoon(
    chapter_image: Image.Image,
    out_dir: str,
    basename: str = "episode",
    width: int = 800,
    max_slice_height: int = 1280,
    overlap: int = 40,
    fmt: str = "png",
    quality: int = 95,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    img = resize_width(chapter_image, width)
    parts = slice_vertical(img, max_slice_height, overlap)
    paths: List[str] = []
    for i, part in enumerate(parts, 1):
        path = os.path.join(out_dir, f"{basename}_{i:02d}.{fmt}")
        if fmt.lower() == "jpg":
            part = part.convert("RGB")
        part.save(path, quality=quality)
        paths.append(path)
    return paths
