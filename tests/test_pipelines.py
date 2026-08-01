from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from manga_ai.config import Config
from manga_ai.pipelines import diffusion, image_api, llm_api, scenario
from manga_ai.pipelines.assemble import ManhwaAssembler
from manga_ai.pipelines.blender_control import build_control_scene, render_control_image


def _png_b64(size: tuple[int, int] = (16, 16), color: str = "white") -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class FakeResponse:
    def __init__(self, payload=None, status_code=200, content=b"", headers=None, text=None):
        self._payload = payload
        self.status_code = status_code
        self.content = content
        self.headers = headers or {"content-type": "application/json"}
        self.text = text if text is not None else str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError("bad response", response=self)


def test_assemble_panels_vertical_page() -> None:
    page = ManhwaAssembler.assemble_panels(
        [Image.new("RGB", (80, 100), "white"), Image.new("RGB", (60, 120), "gray")],
        panel_gap=10,
        background_color="white",
    )

    assert page.size == (80, 230)


def test_blender_control_force_scene_and_pil_fallback(tmp_path) -> None:
    config = Config()
    config.blender.enabled = True
    config.blender.executable = "__missing_blender__"
    config.blender.fallback_to_pil = True
    config.blender.render_width = 160
    config.blender.render_height = 240

    panel = {
        "context": "pose",
        "scene": ["classroom"],
        "force_scene": {
            "location": "classroom",
            "shot": "medium",
            "camera_angle": "eye_level",
            "characters": [{"name": "Kai", "position": (0, 0, 0), "pose": "pointing"}],
            "props": [],
        },
    }
    scene = build_control_scene(config, panel, 1)
    result = render_control_image(config, panel, 1, str(tmp_path))

    assert scene["characters"][0]["pose"] == "pointing"
    assert result is not None
    assert result.source == "pil_fallback"
    assert result.image.size == (160, 240)


def test_image_api_helpers_decode_cloudflare_and_parse_size() -> None:
    response = FakeResponse({"success": True, "result": {"image": _png_b64()}})

    image = image_api._decode_cloudflare_response(response)

    assert image.size == (16, 16)
    assert image_api._parse_size("320x480") == (320, 480)
    assert image_api._parse_size("bad") == (1024, 1024)
    assert "mysterious rival" in image_api._cloudflare_safe_prompt("an assassin appears")


def test_cloudflare_image_generate_posts_json_payload(monkeypatch) -> None:
    config = Config()
    config.image_api.provider = "cloudflare"
    config.image_api.model = "@cf/runwayml/stable-diffusion-v1-5-img2img"
    config.image_api.cloudflare_account_id = "acct"
    config.image_api.cloudflare_api_token = "token"
    config.image_api.cloudflare_use_multipart = False
    config.image_api.size = "32x32"
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None, **kwargs):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse({"success": True, "result": {"image": _png_b64((32, 32))}})

    monkeypatch.setattr(image_api.requests, "post", fake_post)

    result = image_api.generate_image_with_api(
        config,
        "prompt",
        "bad",
        reference_image=Image.new("RGB", (32, 32), "white"),
        strength=0.25,
        seed=7,
        num_steps=12,
    )

    assert result.size == (32, 32)
    assert captured["json"]["strength"] == 0.25
    assert captured["json"]["seed"] == 7
    assert "image_b64" in captured["json"]


def test_llm_content_to_text_and_cloudflare_client(monkeypatch) -> None:
    assert llm_api._content_to_text([{"text": "A"}, "B"]) == "A\nB"

    def fake_post(url, headers=None, json=None, timeout=None):
        return FakeResponse({"success": True, "result": {"response": [{"text": "hello"}]}})

    monkeypatch.setattr(llm_api.requests, "post", fake_post)
    client = llm_api.CloudflareChatClient(account_id="acct", api_token="token")
    completion = client.chat.completions.create(model="@cf/model", messages=[{"role": "user", "content": "hi"}])

    assert completion.choices[0].message.content == "hello"


def test_create_llm_client_none_and_cloudflare() -> None:
    config = Config()
    config.model.llm_provider = "none"
    assert llm_api.create_llm_client(config) is None

    config.model.llm_provider = "cloudflare"
    config.image_api.cloudflare_account_id = "acct"
    config.image_api.cloudflare_api_token = "token"
    assert isinstance(llm_api.create_llm_client(config), llm_api.CloudflareChatClient)


def test_scenario_fallbacks_and_safe_json() -> None:
    config = Config()
    config.scenario.panels = 2
    config.scenario.randomize = False

    parsed = scenario.safe_parse_json('{"a": 1,}')
    story = scenario.generate_story(config, client=None, target_words=60)
    panels = scenario.panels_from_story(config, client=None, story_text=story)
    synthesized = scenario.synthesize_prose_story(config, panels)

    assert parsed == {"a": 1}
    assert len(story.split()) >= 50
    assert len(panels) == 2
    assert synthesized


def test_diffusion_device_selection_and_cache(monkeypatch) -> None:
    monkeypatch.setattr(diffusion.torch.cuda, "is_available", lambda: False)
    if hasattr(diffusion.torch.backends, "mps"):
        monkeypatch.setattr(diffusion.torch.backends.mps, "is_available", lambda: False)

    assert diffusion.select_device_and_dtype("cuda") == ("cpu", torch.float32)

    diffusion._DEVICE_DTYPE = None
    diffusion._CACHED_PIPELINE = None

    def fake_load(model_id, device, dtype, token, fallback_model):
        return {"model": model_id, "device": device, "dtype": dtype, "token": token, "fallback": fallback_model}

    monkeypatch.setattr(diffusion, "load_pipeline_with_fallback", fake_load)
    pipe, device_dtype = diffusion.get_cached_pipeline("model-a", "cpu", "token", "fallback")

    assert pipe["model"] == "model-a"
    assert device_dtype == ("cpu", torch.float32)
