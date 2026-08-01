from __future__ import annotations

import argparse
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from PIL import Image, ImageDraw, ImageFont

from manga_ai.config import Config
from manga_ai.pipelines.blender_control import render_control_image


POSES = [
    "standing",
    "sitting",
    "walking",
    "running",
    "action",
    "leaning",
    "kneeling",
    "pointing",
    "reaching",
    "arms_crossed",
    "looking_down",
]


def _pose_panel(pose: str) -> dict:
    return {
        "context": f"Pose test: {pose}",
        "scene": ["classroom", "wide", "table", "window", f"{pose} pose"],
        "speaker": "Kai",
        "speech": f"Kai is {pose}",
        "force_scene": {
            "location": "classroom",
            "shot": "medium",
            "camera_angle": "eye_level",
            "characters": [
                {
                    "name": "Kai",
                    "position": (0.0, 0.0, 0.0),
                    "pose": pose,
                    "facing": "camera",
                }
            ],
            "props": [
                {"type": "table", "position": (0.0, 0.35, 0.0)},
                {"type": "window", "position": (1.6, 2.05, 1.35)},
            ],
        },
    }


def _make_sheet(paths: list[str], labels: list[str], output_path: str) -> None:
    thumbs = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((260, 340), Image.Resampling.LANCZOS)
        thumbs.append(img)

    gap = 24
    label_h = 36
    columns = min(4, len(thumbs))
    rows = (len(thumbs) + columns - 1) // columns
    width = columns * 260 + (columns + 1) * gap
    height = rows * (340 + label_h + gap) + gap
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, (img, label) in enumerate(zip(thumbs, labels)):
        col = index % columns
        row = index // columns
        x = gap + col * (260 + gap)
        y = gap + row * (340 + label_h + gap) + label_h
        sheet.paste(img, (x + (260 - img.width) // 2, y))
        draw.rectangle((x, y, x + 260, y + 340), outline="black", width=2)
        draw.text((x + 10, y - label_h + 8), label, fill="black", font=font)

    sheet.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a pose test sheet for Blender/PIL control poses.")
    parser.add_argument("--backend", choices=["pil", "blender"], default="pil")
    parser.add_argument("--output-dir", default="output/pose_test")
    args = parser.parse_args()

    config = Config.from_env()
    config.blender.enabled = True
    config.blender.fallback_to_pil = True
    config.blender.output_subdir = "renders"
    config.blender.render_width = 512
    config.blender.render_height = 768
    if args.backend == "pil":
        config.blender.executable = "__force_pil_pose_test__"

    os.makedirs(args.output_dir, exist_ok=True)
    paths = []
    records = []

    for index, pose in enumerate(POSES, 1):
        result = render_control_image(config, _pose_panel(pose), index, args.output_dir)
        if result is None:
            raise RuntimeError(f"No render produced for pose {pose}")
        out_path = os.path.join(args.output_dir, f"pose_{index:02d}_{pose}.png")
        result.image.save(out_path)
        paths.append(out_path)
        records.append(
            {
                "pose": pose,
                "path": out_path,
                "source": result.source,
                "fallback_reason": result.fallback_reason,
                "scene": result.scene,
            }
        )

    sheet_path = os.path.join(args.output_dir, "pose_sheet.png")
    _make_sheet(paths, POSES, sheet_path)
    with open(os.path.join(args.output_dir, "pose_sheet.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Pose sheet: {sheet_path}")


if __name__ == "__main__":
    main()
