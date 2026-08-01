from __future__ import annotations

from PIL import Image, ImageOps


def resize_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(
        image,
        size,
        method=Image.Resampling.LANCZOS,
    )
