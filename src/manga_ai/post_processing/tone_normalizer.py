from __future__ import annotations

from PIL import Image, ImageEnhance, ImageOps


def autocontrast_luma(image: Image.Image, cutoff: float) -> Image.Image:
    try:
        return ImageOps.autocontrast(image, cutoff=(cutoff, cutoff), preserve_tone=True)
    except TypeError:
        return ImageOps.autocontrast(image, cutoff=(cutoff, cutoff))


def apply_gamma(image: Image.Image, gamma: float) -> Image.Image:
    lookup_table = [round(255 * ((value / 255) ** gamma)) for value in range(256)]
    return image.point(lookup_table)


def normalize_tone(
    image: Image.Image,
    *,
    gamma: float,
    contrast: float,
    autocontrast_cutoff: float = 1.0,
) -> Image.Image:
    result = image.convert("L")
    result = autocontrast_luma(result, autocontrast_cutoff)
    result = apply_gamma(result, gamma)
    return ImageEnhance.Contrast(result).enhance(contrast)
