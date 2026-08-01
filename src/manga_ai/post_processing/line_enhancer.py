from __future__ import annotations

from PIL import Image, ImageFilter


def enhance_lines(
    image: Image.Image,
    *,
    radius: float,
    percent: int,
    threshold: int,
) -> Image.Image:
    return image.filter(
        ImageFilter.UnsharpMask(
            radius=radius,
            percent=percent,
            threshold=threshold,
        )
    )
