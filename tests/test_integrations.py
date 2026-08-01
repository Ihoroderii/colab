from __future__ import annotations

import os
from types import SimpleNamespace

from PIL import Image

from manga_ai.config import Config
from manga_ai.integrations import bubbles
from manga_ai.integrations.bubbles import BubbleCharacter, BubbleLine, apply_manhwa_bubbles, manhwa_bubbles_available


def test_bubble_dataclasses_and_project_availability() -> None:
    config = Config()
    config.bubbles.project_path = "../bubble"

    line = BubbleLine(speaker="Kai", text="Hello", kind="speech", position_hint="left")
    character = BubbleCharacter(name="Kai", bbox=(10, 20, 80, 140), head=(45, 35))

    assert line.speaker == "Kai"
    assert character.head == (45, 35)
    assert manhwa_bubbles_available(config) is True


def test_line_to_emotion_mapping() -> None:
    assert bubbles._line_to_emotion(BubbleLine("Kai", "HEY!", "speech")) == "shouting"
    assert bubbles._line_to_emotion(BubbleLine(None, "Later...", "narration")) == "narration"
    assert bubbles._line_to_emotion(BubbleLine("Kai", "quiet", "whisper")) == "whispering"
    assert bubbles._line_to_emotion(BubbleLine("Kai", "thinking", "thought")) == "thinking"


def test_apply_manhwa_bubbles_uses_external_package(tmp_path) -> None:
    config = Config()
    config.bubbles.project_path = "../bubble"
    config.bubbles.use_yolo = False
    config.bubbles.prefer_cairo = False

    image = Image.new("RGB", (320, 480), "white")
    output_path = tmp_path / "panel_bubbles.png"
    result = apply_manhwa_bubbles(
        image,
        [BubbleLine("Kai", "We should go.", "speech", "left")],
        characters=[BubbleCharacter("Kai", (80, 180, 150, 420), (115, 205))],
        config=config,
        output_path=str(output_path),
        seed=1,
    )

    assert result.size == image.size
    assert output_path.exists()


def test_apply_manhwa_bubbles_cleans_temp_files(monkeypatch, tmp_path) -> None:
    created = {}

    def fake_process_panel(input_path, panel, output_path=None, **kwargs):
        created["input_path"] = input_path
        created["output_path"] = output_path
        Image.new("RGB", (64, 64), "white").save(output_path)
        return SimpleNamespace(output_path=output_path)

    monkeypatch.setattr(bubbles, "ensure_manhwa_bubbles_available", lambda config=None: None)
    monkeypatch.setitem(__import__("sys").modules, "manhwa_bubbles.pipeline", SimpleNamespace(process_panel=fake_process_panel))
    monkeypatch.setitem(
        __import__("sys").modules,
        "manhwa_bubbles.scenario_parser",
        SimpleNamespace(
            CharacterPosition=lambda name, bbox, head=None: SimpleNamespace(name=name, bbox=bbox, head=head),
            DialogueEntry=lambda **kwargs: SimpleNamespace(**kwargs),
            PanelData=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    )

    result = apply_manhwa_bubbles(
        Image.new("RGB", (64, 64), "white"),
        [BubbleLine("Kai", "Hello")],
        config=Config(),
    )

    assert result.size == (64, 64)
    assert created["input_path"]
    assert not os.path.exists(created["input_path"])
    assert not os.path.exists(created["output_path"])
