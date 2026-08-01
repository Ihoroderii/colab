from __future__ import annotations

from pathlib import Path

from PIL import Image

from manga_ai.config import Config
from manga_ai.models.demo import CharacterState
from manga_ai.pipelines.blender_control import build_control_scene, render_control_image


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


def _forced_pose_panel(pose: str) -> dict:
    return {
        "context": f"Pose test: {pose}",
        "scene": ["classroom", "wide", f"{pose} pose"],
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
            "props": [{"type": "window", "position": (1.6, 2.05, 1.35)}],
        },
    }


def test_character_state_accepts_supported_poses() -> None:
    for pose in POSES:
        state = CharacterState(name="Kai", pose=pose)
        assert state.pose == pose


def test_force_scene_preserves_each_pose() -> None:
    config = Config()
    for index, pose in enumerate(POSES, 1):
        scene = build_control_scene(config, _forced_pose_panel(pose), index)
        assert scene["characters"][0]["pose"] == pose
        assert scene["panel_index"] == index


def test_pil_fallback_renders_every_pose(tmp_path: Path) -> None:
    config = Config()
    config.blender.enabled = True
    config.blender.executable = "__missing_blender_for_test__"
    config.blender.fallback_to_pil = True
    config.blender.render_width = 320
    config.blender.render_height = 480

    for index, pose in enumerate(POSES, 1):
        result = render_control_image(config, _forced_pose_panel(pose), index, str(tmp_path))
        assert result is not None
        assert result.source == "pil_fallback"
        assert Path(result.image_path).exists()
        assert Image.open(result.image_path).size == (320, 480)
        assert result.scene["characters"][0]["pose"] == pose
