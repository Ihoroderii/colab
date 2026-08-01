from __future__ import annotations

import importlib
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass
class BubbleLine:
    speaker: str | None
    text: str
    kind: str = "speech"
    position_hint: str | None = None


@dataclass
class BubbleCharacter:
    name: str
    bbox: tuple[int, int, int, int]
    head: tuple[int, int] | None = None


def _bubble_project_path(config: Any | None = None) -> str:
    configured = None
    if config is not None and getattr(config, "bubbles", None) is not None:
        configured = getattr(config.bubbles, "project_path", None)
    configured = configured or os.getenv("BUBBLE_PROJECT_PATH")
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    return os.path.abspath(os.path.join(os.getcwd(), "..", "bubble"))


def ensure_manhwa_bubbles_available(config: Any | None = None) -> None:
    project_path = _bubble_project_path(config)
    if project_path not in sys.path and os.path.isdir(project_path):
        sys.path.insert(0, project_path)
    importlib.import_module("manhwa_bubbles")


def manhwa_bubbles_available(config: Any | None = None) -> bool:
    try:
        ensure_manhwa_bubbles_available(config)
        return True
    except Exception:
        return False


def _line_to_emotion(line: BubbleLine) -> str:
    kind = (line.kind or "speech").lower()
    text = line.text or ""
    if kind in ("narration", "narrator", "caption"):
        return "narration"
    if kind in ("thought", "thinking"):
        return "thinking"
    if kind in ("shout", "shouting", "yelling") or "!" in text:
        return "shouting"
    if kind in ("whisper", "whispering"):
        return "whispering"
    if kind in ("sfx", "sound_effect"):
        return "sfx"
    return "normal"


def _image_to_temp_path(image: Image.Image) -> str:
    tmp = tempfile.NamedTemporaryFile(prefix="manga_bubble_input_", suffix=".png", delete=False)
    tmp_path = tmp.name
    tmp.close()
    image.convert("RGB").save(tmp_path)
    return tmp_path


def _default_output_path(input_path: str) -> str:
    root, _ext = os.path.splitext(input_path)
    return f"{root}_bubbles.png"


def apply_manhwa_bubbles(
    image: Image.Image,
    lines: list[BubbleLine],
    *,
    characters: list[BubbleCharacter] | None = None,
    config: Any | None = None,
    panel_id: int = 1,
    input_path: str | None = None,
    output_path: str | None = None,
    seed: int | None = None,
) -> Image.Image:
    """Apply bubbles using the sibling ``manhwa_bubbles`` project.

    The external project is path-based by default and is expected at
    ``../bubble`` from the manga-ai-bot repo root. Set ``BUBBLE_PROJECT_PATH``
    to override it.
    """
    ensure_manhwa_bubbles_available(config)
    from manhwa_bubbles.pipeline import process_panel
    from manhwa_bubbles.scenario_parser import CharacterPosition, DialogueEntry, PanelData

    if not lines:
        return image

    created_temp_input = False
    if input_path is None:
        input_path = _image_to_temp_path(image)
        created_temp_input = True
    else:
        image.convert("RGB").save(input_path)

    created_temp_output = output_path is None
    if output_path is None:
        output_path = _default_output_path(input_path)

    dialogue_entries = [
        DialogueEntry(
            character=line.speaker or "Narrator",
            text=line.text,
            emotion=_line_to_emotion(line),
            position_hint=line.position_hint,
            no_tail=(line.speaker is None or _line_to_emotion(line) == "narration"),
        )
        for line in lines
        if line.text
    ]
    character_entries = [
        CharacterPosition(name=ch.name, bbox=ch.bbox, head=ch.head)
        for ch in (characters or [])
    ]
    panel = PanelData(
        panel_id=panel_id,
        image=os.path.basename(input_path),
        dialogues=dialogue_entries,
        characters=character_entries,
    )

    use_yolo = bool(getattr(getattr(config, "bubbles", None), "use_yolo", False)) if config is not None else False
    prefer_cairo = bool(getattr(getattr(config, "bubbles", None), "prefer_cairo", True)) if config is not None else True
    process_panel(
        input_path,
        panel,
        output_path=output_path,
        use_yolo=use_yolo,
        prefer_cairo=prefer_cairo,
        seed=seed,
    )
    result = Image.open(output_path).convert("RGB")

    if created_temp_input:
        cleanup_paths = [input_path]
        if created_temp_output:
            cleanup_paths.append(output_path)
        for path in cleanup_paths:
            try:
                os.unlink(path)
            except OSError:
                pass
    return result
