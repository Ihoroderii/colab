from __future__ import annotations

from pathlib import Path

from PIL import Image


def load_image(path: Path | str, mode: str = "RGB") -> Image.Image:
    return Image.open(path).convert(mode)


def save_png(image: Image.Image, path: Path | str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, format="PNG")
