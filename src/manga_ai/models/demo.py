from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CharacterState(BaseModel):
    name: str
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pose: Literal[
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
    ] = "standing"
    facing: Literal["left", "right", "camera", "away"] = "camera"
    expression: str = "neutral"


class CameraPlan(BaseModel):
    shot: Literal["wide", "medium", "close_up", "over_shoulder"] = "medium"
    angle_degrees: float = 35.0
    target: str = "center"
    mood: str = "dramatic"


class DialogueLine(BaseModel):
    speaker: str | None = None
    text: str
    kind: Literal["speech", "thought", "shout", "whisper", "narration"] = "speech"
    position: tuple[int, int] = (24, 24)


class PanelPlan(BaseModel):
    panel_number: int = Field(ge=1)
    title: str
    location: Literal["classroom", "rooftop", "street", "room"] = "classroom"
    action: str
    camera: CameraPlan
    characters: list[CharacterState] = Field(default_factory=list)
    props: list[str] = Field(default_factory=list)
    dialogue: list[DialogueLine] = Field(default_factory=list)
    seed: int


class ScenePlan(BaseModel):
    scenario: str
    episode_title: str = "Demo Episode"
    panels: list[PanelPlan] = Field(default_factory=list)
