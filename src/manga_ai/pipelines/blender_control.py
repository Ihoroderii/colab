"""Blender-backed composition/control image generation.

Blender is used here as an external geometry system. The generated image is
fed into the existing img2img path as a layout reference, while diffusion still
does the final manga rendering.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from typing import Any

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


@dataclass
class ControlRenderResult:
    image: Image.Image
    image_path: str
    metadata_path: str
    source: str
    scene: dict[str, Any]
    fallback_reason: str | None = None


def _as_text(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(x) for x in value)
    return str(value or "")


def build_control_scene(config, panel: dict[str, Any], panel_index: int) -> dict[str, Any]:
    """Create a small, deterministic scene spec from a manga panel dict."""
    context = _as_text(panel.get("context"))
    scene_text = _as_text(panel.get("scene"))
    if isinstance(panel.get("force_scene"), dict):
        scene = dict(panel["force_scene"])
        scene.setdefault("panel_index", panel_index)
        scene.setdefault("context", context)
        scene.setdefault("scene_text", scene_text)
        scene.setdefault("location", "room")
        scene.setdefault("shot", "medium")
        scene.setdefault("camera_angle", "eye_level")
        scene.setdefault("characters", [])
        scene.setdefault("props", [])
        return scene

    speaker = _as_text(panel.get("speaker")) or "Narrator"
    speech = _as_text(panel.get("speech"))
    combined = " ".join([context, scene_text, speaker, speech]).lower()

    protagonist = getattr(config.scenario, "protagonist", "Kai") or "Kai"
    antagonist = getattr(config.scenario, "antagonist", "Maria") or "Maria"
    names = []
    for name in [protagonist, antagonist, speaker]:
        if name and name.lower() not in ("narrator", "narration") and name not in names:
            names.append(name)
    if not names:
        names = [protagonist]

    location = "room"
    if any(k in combined for k in ("classroom", "school", "desk", "blackboard")):
        location = "classroom"
    elif any(k in combined for k in ("street", "city", "alley", "road", "neon")):
        location = "street"
    elif any(k in combined for k in ("forest", "shrine", "mountain", "river")):
        location = "outdoor"
    elif any(k in combined for k in ("rooftop", "roof")):
        location = "rooftop"

    shot = "medium"
    if any(k in combined for k in ("close-up", "close up", "face", "expression")):
        shot = "close"
    elif any(k in combined for k in ("wide", "establishing", "room", "street")):
        shot = "wide"
    elif any(k in combined for k in ("over shoulder", "over-the-shoulder")):
        shot = "over_shoulder"

    camera_angle = "eye_level"
    if any(k in combined for k in ("low angle", "towering", "dominates")):
        camera_angle = "low"
    elif any(k in combined for k in ("high angle", "above", "looking down")):
        camera_angle = "high"
    elif any(k in combined for k in ("35 degrees", "diagonal", "dutch")):
        camera_angle = "three_quarter"

    base_positions = [(-1.35, 0.0, 0.0), (1.35, 0.25, 0.0), (0.0, 1.0, 0.0)]
    characters = []
    for i, name in enumerate(names[:3]):
        pose = "standing"
        if name.lower() in combined and "sitting" in combined:
            pose = "sitting"
        elif any(k in combined for k in ("kneel", "kneeling", "on one knee")):
            pose = "kneeling"
        elif any(k in combined for k in ("point", "pointing")):
            pose = "pointing"
        elif any(k in combined for k in ("reach", "reaching", "grab")):
            pose = "reaching"
        elif any(k in combined for k in ("arms crossed", "crossed arms")):
            pose = "arms_crossed"
        elif any(k in combined for k in ("looking down", "looks down", "head bowed")):
            pose = "looking_down"
        elif any(k in combined for k in ("run", "running", "chase", "sprint")):
            pose = "running"
        elif any(k in combined for k in ("walk", "walking")):
            pose = "walking"
        elif any(k in combined for k in ("fight", "attack", "strike")):
            pose = "action"
        characters.append(
            {
                "name": name,
                "position": base_positions[i],
                "pose": pose,
                "facing": "right" if i == 0 else "left",
            }
        )

    props = []
    if any(k in combined for k in ("table", "desk", "classroom")):
        props.append({"type": "table", "position": (0.0, 0.35, 0.0)})
    if any(k in combined for k in ("window", "rain", "classroom", "room")):
        props.append({"type": "window", "position": (1.6, 2.05, 1.35)})
    if any(k in combined for k in ("door", "exit")):
        props.append({"type": "door", "position": (-1.8, 2.06, 1.0)})

    return {
        "panel_index": panel_index,
        "context": context,
        "scene_text": scene_text,
        "location": location,
        "shot": shot,
        "camera_angle": camera_angle,
        "characters": characters,
        "props": props,
    }


def _blender_script() -> str:
    return r'''
import json
import math
import sys

import bpy
from mathutils import Vector

spec_path = sys.argv[sys.argv.index("--") + 1]
with open(spec_path, "r", encoding="utf-8") as f:
    payload = json.load(f)

scene_spec = payload["scene"]
output_path = payload["output_path"]
width = int(payload["width"])
height = int(payload["height"])

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

def material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    return mat

mat_floor = material("floor_light", (0.78, 0.78, 0.78, 1))
mat_wall = material("wall_light", (0.9, 0.9, 0.9, 1))
mat_prop = material("prop_mid", (0.48, 0.48, 0.48, 1))
mat_char = material("character_dark", (0.18, 0.18, 0.18, 1))
mat_head = material("character_head", (0.62, 0.62, 0.62, 1))
mat_window = material("window_pale", (0.72, 0.82, 0.9, 1))

def cube(name, loc, scale, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(mat)
    return obj

def cylinder_between(name, start, end, radius, mat):
    start_v = Vector(start)
    end_v = Vector(end)
    mid = (start_v + end_v) / 2.0
    length = (end_v - start_v).length
    bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=radius, depth=length, location=mid)
    obj = bpy.context.object
    obj.name = name
    direction = end_v - start_v
    obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    obj.data.materials.append(mat)
    return obj

def character(name, x, y, pose):
    pose = (pose or "standing").lower()
    hip_z = 0.75
    height = 1.35
    if pose == "sitting":
        hip_z = 0.45
        height = 1.0
    elif pose == "kneeling":
        hip_z = 0.52
        height = 1.08
    lean = -0.28 if pose == "leaning" else 0.0
    stride = 0.32 if pose in ("walking", "running") else 0.0
    body_top = (x + lean, y, hip_z + height * 0.46)
    cylinder_between(name + "_body", (x, y, hip_z), body_top, 0.13, mat_char)
    head_drop = -0.16 if pose == "looking_down" else 0.0
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.22, location=(x + lean * 1.25, y + head_drop * 0.25, hip_z + height * 0.72 + head_drop))
    head = bpy.context.object
    head.name = name + "_head"
    head.data.materials.append(mat_head)
    arm_y = y - 0.02
    if pose == "action":
        cylinder_between(name + "_arm_l", (x - 0.1, arm_y, hip_z + 0.55), (x - 0.65, arm_y - 0.2, hip_z + 0.95), 0.045, mat_char)
        cylinder_between(name + "_arm_r", (x + 0.1, arm_y, hip_z + 0.55), (x + 0.62, arm_y + 0.05, hip_z + 0.25), 0.045, mat_char)
    elif pose in ("walking", "running"):
        cylinder_between(name + "_arm_l", (x - 0.1, arm_y, hip_z + 0.55), (x - 0.48, arm_y - 0.08, hip_z + 0.18), 0.045, mat_char)
        cylinder_between(name + "_arm_r", (x + 0.1, arm_y, hip_z + 0.55), (x + 0.48, arm_y + 0.06, hip_z + 0.78), 0.045, mat_char)
    elif pose == "pointing":
        cylinder_between(name + "_arm_l", (x - 0.1, arm_y, hip_z + 0.55), (x - 0.42, arm_y, hip_z + 0.25), 0.045, mat_char)
        cylinder_between(name + "_arm_r", (x + 0.1, arm_y, hip_z + 0.58), (x + 0.82, arm_y - 0.05, hip_z + 0.62), 0.045, mat_char)
    elif pose == "reaching":
        cylinder_between(name + "_arm_l", (x - 0.1, arm_y, hip_z + 0.58), (x - 0.58, arm_y - 0.06, hip_z + 1.05), 0.045, mat_char)
        cylinder_between(name + "_arm_r", (x + 0.1, arm_y, hip_z + 0.58), (x + 0.58, arm_y - 0.06, hip_z + 1.05), 0.045, mat_char)
    elif pose == "arms_crossed":
        cylinder_between(name + "_arm_l", (x - 0.18, arm_y, hip_z + 0.58), (x + 0.28, arm_y - 0.02, hip_z + 0.46), 0.05, mat_char)
        cylinder_between(name + "_arm_r", (x + 0.18, arm_y, hip_z + 0.58), (x - 0.28, arm_y - 0.02, hip_z + 0.46), 0.05, mat_char)
    elif pose == "looking_down":
        cylinder_between(name + "_arm_l", (x - 0.1, arm_y, hip_z + 0.55), (x - 0.32, arm_y, hip_z + 0.18), 0.045, mat_char)
        cylinder_between(name + "_arm_r", (x + 0.1, arm_y, hip_z + 0.55), (x + 0.32, arm_y, hip_z + 0.18), 0.045, mat_char)
    elif pose == "leaning":
        cylinder_between(name + "_arm_l", (x - 0.1 + lean, arm_y, hip_z + 0.55), (x - 0.58 + lean, arm_y, hip_z + 0.18), 0.045, mat_char)
        cylinder_between(name + "_arm_r", (x + 0.1 + lean, arm_y, hip_z + 0.55), (x + 0.18, arm_y, hip_z + 0.18), 0.045, mat_char)
    else:
        cylinder_between(name + "_arm_l", (x - 0.1 + lean, arm_y, hip_z + 0.55), (x - 0.42 + lean, arm_y, hip_z + 0.25), 0.045, mat_char)
        cylinder_between(name + "_arm_r", (x + 0.1 + lean, arm_y, hip_z + 0.55), (x + 0.42 + lean, arm_y, hip_z + 0.25), 0.045, mat_char)
    if pose == "sitting":
        cylinder_between(name + "_leg_l", (x - 0.07, y, hip_z), (x - 0.4, y - 0.15, 0.18), 0.055, mat_char)
        cylinder_between(name + "_leg_r", (x + 0.07, y, hip_z), (x + 0.4, y - 0.15, 0.18), 0.055, mat_char)
    elif pose == "kneeling":
        cylinder_between(name + "_leg_l", (x - 0.07, y, hip_z), (x - 0.34, y - 0.05, 0.08), 0.06, mat_char)
        cylinder_between(name + "_leg_r", (x + 0.07, y, hip_z), (x + 0.35, y + 0.12, 0.42), 0.06, mat_char)
        cylinder_between(name + "_shin_r", (x + 0.35, y + 0.12, 0.42), (x + 0.58, y - 0.04, 0.08), 0.055, mat_char)
    elif pose in ("walking", "running"):
        cylinder_between(name + "_leg_l", (x - 0.07, y, hip_z), (x - 0.22 - stride, y, 0.08), 0.06, mat_char)
        cylinder_between(name + "_leg_r", (x + 0.07, y, hip_z), (x + 0.22 + stride, y, 0.08), 0.06, mat_char)
    elif pose == "leaning":
        cylinder_between(name + "_leg_l", (x - 0.07, y, hip_z), (x - 0.36, y, 0.08), 0.06, mat_char)
        cylinder_between(name + "_leg_r", (x + 0.07, y, hip_z), (x + 0.1, y, 0.08), 0.06, mat_char)
    else:
        cylinder_between(name + "_leg_l", (x - 0.07, y, hip_z), (x - 0.22, y, 0.08), 0.06, mat_char)
        cylinder_between(name + "_leg_r", (x + 0.07, y, hip_z), (x + 0.22, y, 0.08), 0.06, mat_char)

location = scene_spec.get("location", "room")
cube("floor", (0, 0.6, -0.04), (4.2, 3.2, 0.04), mat_floor)
if location in ("room", "classroom"):
    cube("back_wall", (0, 2.18, 1.2), (4.2, 0.05, 1.3), mat_wall)
    cube("left_wall", (-2.1, 0.6, 1.2), (0.05, 3.2, 1.3), mat_wall)
elif location == "street":
    for i, x in enumerate([-1.8, 1.8]):
        cube("building_%d" % i, (x, 2.0, 1.3), (0.55, 0.35, 1.3), mat_wall)
elif location == "rooftop":
    cube("roof_ledge", (0, 2.05, 0.4), (4.2, 0.12, 0.4), mat_prop)

for prop in scene_spec.get("props", []):
    ptype = prop.get("type")
    x, y, z = prop.get("position", (0, 0, 0))
    if ptype == "table":
        cube("table_top", (x, y, 0.55), (0.95, 0.45, 0.08), mat_prop)
        for dx in [-0.75, 0.75]:
            for dy in [-0.28, 0.28]:
                cube("table_leg", (x + dx, y + dy, 0.27), (0.04, 0.04, 0.27), mat_prop)
    elif ptype == "window":
        cube("window", (x, y, z), (0.62, 0.03, 0.42), mat_window)
    elif ptype == "door":
        cube("door", (x, y, z), (0.42, 0.04, 0.95), mat_prop)

for ch in scene_spec.get("characters", []):
    x, y, z = ch.get("position", (0, 0, 0))
    character(ch.get("name", "character"), x, y, ch.get("pose", "standing"))

bpy.ops.object.light_add(type="AREA", location=(0, -2.5, 4.0))
light = bpy.context.object
light.name = "softbox"
light.data.energy = 450
light.data.size = 4

bpy.ops.object.camera_add()
camera = bpy.context.object
shot = scene_spec.get("shot", "medium")
angle = scene_spec.get("camera_angle", "eye_level")
if shot == "close":
    camera.location = (0.0, -3.4, 1.35)
elif shot == "wide":
    camera.location = (3.4, -5.4, 2.8)
elif shot == "over_shoulder":
    camera.location = (-1.8, -3.0, 1.45)
else:
    camera.location = (2.5, -4.2, 2.0)
if angle == "low":
    camera.location.z = 0.9
elif angle == "high":
    camera.location.z = 4.0
target = Vector((0.0, 0.45, 0.95))
direction = target - Vector(camera.location)
camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
camera.data.lens = 35
bpy.context.scene.camera = camera

scene = bpy.context.scene
try:
    scene.render.engine = "BLENDER_EEVEE_NEXT"
except Exception:
    scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = width
scene.render.resolution_y = height
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "Medium High Contrast"
scene.world.color = (1, 1, 1)
scene.render.use_freestyle = True
scene.render.line_thickness = 1.6
scene.render.filepath = output_path
bpy.ops.render.render(write_still=True)
'''


def _render_with_blender(config, scene: dict[str, Any], image_path: str) -> tuple[bool, str | None]:
    executable = getattr(config.blender, "executable", "blender")
    if not shutil.which(executable) and not os.path.exists(executable):
        return False, f"Blender executable not found: {executable}"

    payload = {
        "scene": scene,
        "output_path": image_path,
        "width": int(getattr(config.blender, "render_width", 768)),
        "height": int(getattr(config.blender, "render_height", 1024)),
    }
    with tempfile.TemporaryDirectory(prefix="manga_blender_") as tmpdir:
        spec_path = os.path.join(tmpdir, "scene.json")
        script_path = os.path.join(tmpdir, "render_scene.py")
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(_blender_script())

        cmd = [executable, "--background", "--factory-startup", "--python", script_path, "--", spec_path]
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            reason = completed.stderr[-1000:] or completed.stdout[-1000:] or f"exit code {completed.returncode}"
            logger.warning("Blender control render failed: %s", reason)
            return False, reason
    if not os.path.exists(image_path):
        return False, f"Blender completed but did not create {image_path}"
    return True, None


def _render_pil_fallback(config, scene: dict[str, Any], image_path: str) -> None:
    width = int(getattr(config.blender, "render_width", 768))
    height = int(getattr(config.blender, "render_height", 1024))
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    horizon = int(height * 0.44)
    draw.rectangle([0, 0, width, horizon], fill=(238, 238, 238), outline=(20, 20, 20), width=3)
    draw.rectangle([0, horizon, width, height], fill=(218, 218, 218), outline=(20, 20, 20), width=3)
    vanishing = (width // 2, horizon)
    for x in range(-width, width * 2, max(80, width // 6)):
        draw.line([x, height, vanishing[0], vanishing[1]], fill=(90, 90, 90), width=2)

    for prop in scene.get("props", []):
        ptype = prop.get("type")
        if ptype == "table":
            y = int(height * 0.58)
            draw.rectangle([int(width * 0.25), y, int(width * 0.75), y + int(height * 0.08)], outline="black", width=5, fill=(170, 170, 170))
            draw.line([int(width * 0.32), y + 60, int(width * 0.25), y + 160], fill="black", width=5)
            draw.line([int(width * 0.68), y + 60, int(width * 0.75), y + 160], fill="black", width=5)
        elif ptype == "window":
            draw.rectangle([int(width * 0.62), int(height * 0.15), int(width * 0.88), int(height * 0.34)], outline="black", width=5, fill=(225, 235, 240))
            draw.line([int(width * 0.75), int(height * 0.15), int(width * 0.75), int(height * 0.34)], fill="black", width=3)

    characters = scene.get("characters", [])[:3]
    character_slots = [(0.5, 0.62)] if len(characters) == 1 else [(0.32, 0.62), (0.68, 0.62), (0.5, 0.55)]
    for index, ch in enumerate(characters):
        sx, sy = character_slots[index]
        cx = int(width * sx)
        foot_y = int(height * sy)
        scale = height // 8
        head_r = max(16, scale // 5)
        pose = ch.get("pose", "standing")
        lean = -scale // 3 if pose == "leaning" else 0
        head_x = cx + int(lean * 1.2)
        head_y = foot_y - scale * 2
        if pose == "sitting":
            foot_y += scale // 3
            head_y += scale // 2
        elif pose == "kneeling":
            foot_y += scale // 5
            head_y += scale // 3
        elif pose == "looking_down":
            head_y += scale // 5
        draw.ellipse([head_x - head_r, head_y, head_x + head_r, head_y + head_r * 2], outline="black", width=5, fill=(190, 190, 190))
        neck_y = head_y + head_r * 2
        hip_y = foot_y - scale
        draw.line([head_x, neck_y, cx, hip_y], fill="black", width=8)
        if pose == "action":
            draw.line([cx, neck_y + 20, cx - scale, neck_y - 20], fill="black", width=6)
            draw.line([cx, neck_y + 20, cx + scale, neck_y + 45], fill="black", width=6)
        elif pose in ("walking", "running"):
            arm_scale = scale if pose == "running" else scale // 2
            draw.line([cx, neck_y + 20, cx - arm_scale, neck_y + scale // 2], fill="black", width=6)
            draw.line([cx, neck_y + 20, cx + arm_scale, neck_y - scale // 4], fill="black", width=6)
        elif pose == "pointing":
            draw.line([cx, neck_y + 20, cx - scale // 2, neck_y + scale // 2], fill="black", width=6)
            draw.line([cx, neck_y + 20, cx + scale, neck_y + 10], fill="black", width=6)
        elif pose == "reaching":
            draw.line([cx, neck_y + 20, cx - scale // 2, neck_y - scale // 2], fill="black", width=6)
            draw.line([cx, neck_y + 20, cx + scale // 2, neck_y - scale // 2], fill="black", width=6)
        elif pose == "arms_crossed":
            draw.line([cx - scale // 2, neck_y + 25, cx + scale // 2, neck_y + scale // 3], fill="black", width=7)
            draw.line([cx + scale // 2, neck_y + 25, cx - scale // 2, neck_y + scale // 3], fill="black", width=7)
        elif pose == "looking_down":
            draw.line([cx, neck_y + 20, cx - scale // 3, neck_y + scale // 2], fill="black", width=6)
            draw.line([cx, neck_y + 20, cx + scale // 3, neck_y + scale // 2], fill="black", width=6)
        elif pose == "leaning":
            draw.line([head_x, neck_y + 20, head_x - scale // 2, neck_y + scale // 2], fill="black", width=6)
            draw.line([head_x, neck_y + 20, cx + scale // 3, neck_y + scale // 2], fill="black", width=6)
        else:
            draw.line([cx, neck_y + 20, cx - scale // 2, neck_y + scale // 2], fill="black", width=6)
            draw.line([cx, neck_y + 20, cx + scale // 2, neck_y + scale // 2], fill="black", width=6)
        if pose == "sitting":
            draw.line([cx, hip_y, cx - scale // 2, foot_y - scale // 8], fill="black", width=7)
            draw.line([cx, hip_y, cx + scale // 2, foot_y - scale // 8], fill="black", width=7)
            draw.line([cx - scale // 2, foot_y - scale // 8, cx - scale, foot_y], fill="black", width=7)
            draw.line([cx + scale // 2, foot_y - scale // 8, cx + scale, foot_y], fill="black", width=7)
        elif pose == "kneeling":
            draw.line([cx, hip_y, cx - scale // 2, foot_y], fill="black", width=7)
            draw.line([cx, hip_y, cx + scale // 2, hip_y + scale // 2], fill="black", width=7)
            draw.line([cx + scale // 2, hip_y + scale // 2, cx + scale, foot_y], fill="black", width=7)
        elif pose in ("walking", "running"):
            stride = scale if pose == "running" else scale * 2 // 3
            draw.line([cx, hip_y, cx - stride, foot_y], fill="black", width=7)
            draw.line([cx, hip_y, cx + stride, foot_y], fill="black", width=7)
        elif pose == "leaning":
            draw.line([cx, hip_y, cx - scale // 2, foot_y], fill="black", width=7)
            draw.line([cx, hip_y, cx + scale // 6, foot_y], fill="black", width=7)
        else:
            draw.line([cx, hip_y, cx - scale // 2, foot_y], fill="black", width=7)
            draw.line([cx, hip_y, cx + scale // 2, foot_y], fill="black", width=7)
        draw.text((max(10, cx - scale), min(height - 30, foot_y + 12)), pose, fill="black")

    img.save(image_path, quality=95)


def render_control_image(config, panel: dict[str, Any], panel_index: int, output_dir: str) -> ControlRenderResult | None:
    if not getattr(config.blender, "enabled", False):
        return None

    control_dir = os.path.join(output_dir, getattr(config.blender, "output_subdir", "control"))
    os.makedirs(control_dir, exist_ok=True)
    image_path = os.path.join(control_dir, f"panel_{panel_index:03d}_layout.png")
    metadata_path = os.path.join(control_dir, f"panel_{panel_index:03d}_scene.json")
    scene = build_control_scene(config, panel, panel_index)

    source = "blender"
    fallback_reason = None
    rendered, failure_reason = _render_with_blender(config, scene, image_path)
    if not rendered:
        if not getattr(config.blender, "fallback_to_pil", True):
            return None
        source = "pil_fallback"
        fallback_reason = failure_reason
        logger.warning("Using PIL control fallback for panel %s: %s", panel_index, fallback_reason)
        _render_pil_fallback(config, scene, image_path)

    metadata = {
        "source": source,
        "image_path": image_path,
        "scene": scene,
        "config": asdict(getattr(config, "blender")),
        "fallback_reason": fallback_reason,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return ControlRenderResult(
        image=Image.open(image_path).convert("RGB"),
        image_path=image_path,
        metadata_path=metadata_path,
        source=source,
        scene=scene,
        fallback_reason=fallback_reason,
    )
