from __future__ import annotations

import pytest
from pydantic import ValidationError

from manga_ai.models import CameraPlan, CharacterState, DialogueLine, PanelPlan, ScenePlan


def test_scene_plan_serializes_nested_panel_data() -> None:
    scene = ScenePlan(
        scenario="Kai enters the classroom.",
        episode_title="Demo",
        panels=[
            PanelPlan(
                panel_number=1,
                title="Opening",
                action="Kai stands near the window.",
                camera=CameraPlan(shot="wide", angle_degrees=35, target="Kai"),
                characters=[CharacterState(name="Kai", pose="standing", facing="camera")],
                dialogue=[DialogueLine(speaker="Kai", text="I'm here.", kind="speech")],
                seed=123,
            )
        ],
    )

    data = scene.model_dump(mode="json")

    assert data["panels"][0]["characters"][0]["pose"] == "standing"
    assert data["panels"][0]["dialogue"][0]["text"] == "I'm here."


def test_character_state_rejects_unknown_pose() -> None:
    with pytest.raises(ValidationError):
        CharacterState(name="Kai", pose="flying")


def test_panel_plan_requires_positive_panel_number() -> None:
    with pytest.raises(ValidationError):
        PanelPlan(
            panel_number=0,
            title="Bad",
            action="Invalid panel number.",
            camera=CameraPlan(),
            seed=1,
        )


def test_dialogue_line_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        DialogueLine(speaker="Kai", text="Hello", kind="unknown")
