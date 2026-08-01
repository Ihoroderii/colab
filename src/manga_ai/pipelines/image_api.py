"""Remote image API backend for manga panel generation."""
from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from PIL import Image
import requests

logger = logging.getLogger(__name__)


@dataclass
class ImageRequest:
    prompt: str
    negative_prompt: str | None = None
    width: int = 1024
    height: int = 1024
    reference_image: Image.Image | None = None
    strength: float | None = None
    guidance: float | None = None
    seed: int | None = None
    num_steps: int | None = None


class ImageProvider(ABC):
    @abstractmethod
    def generate(self, request: ImageRequest) -> Image.Image:
        raise NotImplementedError


def _response_value(item: Any, key: str):
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _decode_image_response(response) -> Image.Image:
    data = _response_value(response, "data")
    if not data:
        raise RuntimeError("Image API returned no image data")

    first = data[0]
    b64_json = _response_value(first, "b64_json")
    if b64_json:
        return Image.open(io.BytesIO(base64.b64decode(b64_json))).convert("RGB")

    url = _response_value(first, "url")
    if url:
        with urllib.request.urlopen(url, timeout=120) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")

    raise RuntimeError("Image API response did not include b64_json or url")


def _openai_client(config):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai package is required for --image-backend api") from exc

    api_key = getattr(config.image_api, "api_key", None)
    if not api_key:
        raise ValueError("Missing image API key. Set IMAGE_API_KEY, OPENAI_IMAGE_API_KEY, or OPENAI_API_KEY.")

    kwargs = {"api_key": api_key}
    base_url = getattr(config.image_api, "base_url", None)
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _parse_size(size: str) -> tuple[int, int]:
    try:
        width, height = str(size).lower().split("x", 1)
        return int(width), int(height)
    except Exception:
        return 1024, 1024


def _image_from_api_payload(payload: Any) -> Image.Image:
    if isinstance(payload, Image.Image):
        return payload.convert("RGB")
    if isinstance(payload, bytes):
        return Image.open(io.BytesIO(payload)).convert("RGB")
    if isinstance(payload, str):
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    if isinstance(payload, list):
        return Image.fromarray(payload).convert("RGB")
    raise RuntimeError(f"Unsupported image payload type: {type(payload).__name__}")


def _huggingface_client(config):
    try:
        from huggingface_hub import InferenceClient
    except ImportError as exc:
        raise ImportError("huggingface_hub is required for Hugging Face image API generation") from exc

    api_key = getattr(config.image_api, "api_key", None)
    if not api_key:
        raise ValueError("Missing Hugging Face API key. Set HF_TOKEN or IMAGE_API_KEY.")

    provider = getattr(config.image_api, "hf_provider", None)
    if provider:
        return InferenceClient(provider=provider, api_key=api_key)
    return InferenceClient(api_key=api_key)


class HuggingFaceImageProvider(ImageProvider):
    def __init__(self, config):
        self.config = config

    def generate(self, request: ImageRequest) -> Image.Image:
        return _huggingface_generate(
            self.config,
            request.prompt,
            request.negative_prompt,
            request.reference_image,
        )


def _huggingface_generate(config, prompt: str, negative_prompt: str | None, reference_image: Image.Image | None) -> Image.Image:
    client = _huggingface_client(config)
    model = getattr(config.image_api, "model", None)
    if not model:
        raise ValueError("Missing Hugging Face image model. Set IMAGE_API_MODEL or --image-api-model.")

    width, height = _parse_size(getattr(config.image_api, "size", "1024x1024"))
    kwargs = {
        "model": model,
        "guidance_scale": getattr(config.generation, "guidance_scale", 7.5),
        "num_inference_steps": getattr(config.generation, "num_inference_steps", 30),
    }
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt

    def _text_to_image(text_prompt: str) -> Image.Image:
        return client.text_to_image(
            text_prompt,
            width=width,
            height=height,
            **kwargs,
        )

    if reference_image is None:
        image = _text_to_image(prompt)
    else:
        with io.BytesIO() as buf:
            reference_image.convert("RGB").save(buf, format="PNG")
            input_image = buf.getvalue()
        try:
            image = client.image_to_image(
                input_image,
                prompt=prompt,
                target_size={"width": width, "height": height},
                **kwargs,
            )
        except Exception as exc:
            message = str(exc)
            unsupported_i2i = (
                "not supported for task image-to-image" in message
                or "Supported task: text-to-image" in message
            )
            if not unsupported_i2i:
                raise
            logger.warning(
                "Hugging Face model/provider does not support image-to-image; retrying as text-to-image: %s",
                message,
            )
            control_prompt = (
                f"{prompt}\n\n"
                "Composition guide from control render: preserve the described character positions, "
                "camera angle, room geometry, foreground/background spacing, and prop layout."
            )
            image = _text_to_image(control_prompt)

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, bytes):
        return Image.open(io.BytesIO(image)).convert("RGB")
    return Image.open(image).convert("RGB")


class CloudflareImageProvider(ImageProvider):
    def __init__(self, config):
        self.config = config

    def generate(self, request: ImageRequest) -> Image.Image:
        return _cloudflare_generate(
            self.config,
            request.prompt,
            request.negative_prompt,
            request.reference_image,
            width=request.width,
            height=request.height,
            strength=request.strength,
            guidance=request.guidance,
            seed=request.seed,
            num_steps=request.num_steps,
        )


def _decode_cloudflare_response(response: requests.Response) -> Image.Image:
    content_type = response.headers.get("content-type", "").lower()
    if content_type.startswith("image/"):
        return Image.open(io.BytesIO(response.content)).convert("RGB")

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Cloudflare returned non-image, non-JSON response: {response.text[:500]}") from exc

    if not payload.get("success", True):
        raise RuntimeError(f"Cloudflare image request failed: {payload}")

    result = payload.get("result", payload)
    if isinstance(result, dict):
        for key in ("image", "image_b64", "b64_json", "data"):
            value = result.get(key)
            if value:
                return _image_from_api_payload(value)
        if "images" in result and result["images"]:
            return _image_from_api_payload(result["images"][0])
    if isinstance(result, list) and result:
        return _image_from_api_payload(result[0])
    if isinstance(result, str):
        return _image_from_api_payload(result)

    raise RuntimeError(f"Cloudflare response did not include image data: {payload}")


def _prepare_cloudflare_reference(reference_image: Image.Image) -> bytes:
    img = reference_image.convert("RGB")
    img.thumbnail((1024, 1024))
    with io.BytesIO() as buf:
        img.save(buf, format="PNG")
        return buf.getvalue()


def _cloudflare_safe_prompt(prompt: str, negative_prompt: str | None = None) -> str:
    # Keep the image request about cinematic composition and avoid words that
    # often trip provider moderation in action/noir stories.
    text = prompt or "cinematic manga panel"
    replacements = {
        "assassin": "mysterious rival",
        "deadly": "intense",
        "kill": "defeat",
        "killing": "confrontation",
        "take them down": "confront them",
        "terrorizing": "challenging",
        "underworld": "hidden city network",
        "unscathed": "unchanged",
        "weapon": "tool",
        "blood": "shadow",
    }
    for old, new in replacements.items():
        text = text.replace(old, new).replace(old.title(), new.title())
    safe = (
        f"{text}\n\n"
        "Safe visual direction: non-graphic cinematic webtoon panel, no injury, no gore, no weapons, "
        "dramatic but non-violent tension, expressive characters, urban rooftop setting, professional manga illustration."
    )
    if negative_prompt:
        safe += f"\nAvoid: {negative_prompt}, gore, blood, wounds, weapons, explicit violence"
    return safe


def _cloudflare_generate(
    config,
    prompt: str,
    negative_prompt: str | None,
    reference_image: Image.Image | None,
    *,
    width: int | None = None,
    height: int | None = None,
    strength: float | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    num_steps: int | None = None,
) -> Image.Image:
    account_id = getattr(config.image_api, "cloudflare_account_id", None)
    token = getattr(config.image_api, "cloudflare_api_token", None) or getattr(config.image_api, "api_key", None)
    model = getattr(config.image_api, "model", None)
    if not account_id:
        raise ValueError("Missing Cloudflare account ID. Set CLOUDFLARE_ACCOUNT_ID or --cloudflare-account-id.")
    if not token:
        raise ValueError("Missing Cloudflare API token. Set CLOUDFLARE_API_TOKEN or --cloudflare-api-token.")
    if not model:
        raise ValueError("Missing Cloudflare image model. Set IMAGE_API_MODEL or --image-api-model.")

    parsed_width, parsed_height = _parse_size(getattr(config.image_api, "size", "1024x1024"))
    width = int(width or parsed_width)
    height = int(height or parsed_height)
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    full_prompt = prompt
    guidance_value = float(guidance if guidance is not None else getattr(config.generation, "guidance_scale", 7.5))
    steps_value = int(num_steps if num_steps is not None else getattr(config.generation, "num_inference_steps", 30))
    use_multipart = bool(getattr(config.image_api, "cloudflare_use_multipart", True))

    def _post_cloudflare(request_prompt: str, request_reference: Image.Image | None) -> requests.Response:
        mode = "image-to-image" if request_reference is not None else "text-to-image"
        transport = "multipart" if use_multipart else "json"
        logger.info(
            "Calling Cloudflare image API: model=%s mode=%s transport=%s size=%sx%s strength=%s guidance=%s steps=%s seed=%s",
            model,
            mode,
            transport,
            width,
            height,
            strength,
            guidance_value,
            steps_value,
            seed,
        )
        if not use_multipart:
            payload = {
                "prompt": request_prompt,
                "width": width,
                "height": height,
                "guidance": guidance_value,
                "num_steps": steps_value,
            }
            if negative_prompt:
                payload["negative_prompt"] = negative_prompt
            if strength is not None:
                payload["strength"] = float(strength)
            if seed is not None:
                payload["seed"] = int(seed)
            if request_reference is not None:
                payload["image_b64"] = base64.b64encode(_prepare_cloudflare_reference(request_reference)).decode("ascii")
            return requests.post(
                url,
                headers={**headers, "content-type": "application/json"},
                json=payload,
                timeout=180,
            )

        data = {
            "prompt": request_prompt,
            "width": str(width),
            "height": str(height),
            "guidance": str(guidance_value),
            "num_steps": str(steps_value),
        }
        if negative_prompt:
            data["negative_prompt"] = negative_prompt
        if strength is not None:
            data["strength"] = str(float(strength))
        if seed is not None:
            data["seed"] = str(int(seed))
        files = None
        if request_reference is not None:
            data["prompt"] = (
                f"{request_prompt}\n\n"
                "Use input_image_0 as a composition/control reference. Preserve the layout, "
                "camera angle, character positions, depth, and major object placement while rendering manga style."
            )
            files = {
                "input_image_0": ("control.png", _prepare_cloudflare_reference(request_reference), "image/png")
            }
        return requests.post(url, headers=headers, data=data, files=files, timeout=180)

    response = _post_cloudflare(full_prompt, reference_image)

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        if response.status_code == 401:
            raise RuntimeError(
                "Cloudflare authentication failed (401). Check CLOUDFLARE_API_TOKEN, rotate it if it was exposed, "
                "and create it from Workers AI > Use REST API or grant account-scoped Workers AI Read and Workers AI Edit permissions. "
                "Also confirm CLOUDFLARE_ACCOUNT_ID belongs to the same account as the token. "
                f"Cloudflare response: {response.text[:1000]}"
            ) from exc
        flagged = response.status_code == 400 and (
            "flagged" in response.text.lower()
            or '"code":3030' in response.text
            or "code\":3030" in response.text
        )
        if flagged:
            logger.warning("Cloudflare flagged the prompt/reference; retrying with safer text-only prompt")
            retry_response = _post_cloudflare(_cloudflare_safe_prompt(prompt, negative_prompt), None)
            try:
                retry_response.raise_for_status()
            except requests.HTTPError as retry_exc:
                raise RuntimeError(
                    f"Cloudflare image request failed after safety retry: {retry_response.status_code} {retry_response.text[:1000]}"
                ) from retry_exc
            return _decode_cloudflare_response(retry_response)
        raise RuntimeError(f"Cloudflare image request failed: {response.status_code} {response.text[:1000]}") from exc
    logger.info("Cloudflare image API completed: model=%s", model)
    return _decode_cloudflare_response(response)


class OpenAIImageProvider(ImageProvider):
    def __init__(self, config):
        self.config = config

    def generate(self, request: ImageRequest) -> Image.Image:
        return _openai_generate(
            self.config,
            request.prompt,
            request.negative_prompt,
            request.reference_image,
        )


def generate_image_with_api(
    config,
    prompt: str,
    negative_prompt: str | None = None,
    reference_image: Image.Image | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
    strength: float | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    num_steps: int | None = None,
) -> Image.Image:
    """Generate a panel remotely.

    For OpenAI, text-only uses image generation. A reference/control image uses
    image edit so the API receives the Blender/PIL layout as conditioning.
    """
    provider = getattr(config.image_api, "provider", "openai").lower()
    if provider == "huggingface":
        return HuggingFaceImageProvider(config).generate(ImageRequest(prompt, negative_prompt, reference_image=reference_image))
    if provider == "cloudflare":
        parsed_width, parsed_height = _parse_size(getattr(config.image_api, "size", "1024x1024"))
        return CloudflareImageProvider(config).generate(
            ImageRequest(
                prompt,
                negative_prompt,
                width=width or parsed_width,
                height=height or parsed_height,
                reference_image=reference_image,
                strength=strength,
                guidance=guidance,
                seed=seed,
                num_steps=num_steps,
            )
        )
    if provider != "openai":
        raise ValueError(f"Unsupported image API provider: {provider}")

    return OpenAIImageProvider(config).generate(ImageRequest(prompt, negative_prompt, reference_image=reference_image))


def _openai_generate(config, prompt: str, negative_prompt: str | None, reference_image: Image.Image | None) -> Image.Image:
    client = _openai_client(config)
    model = getattr(config.image_api, "model", "gpt-image-1")
    size = getattr(config.image_api, "size", "1024x1024")
    quality = getattr(config.image_api, "quality", "medium")
    output_format = getattr(config.image_api, "output_format", "png")

    full_prompt = prompt
    if negative_prompt:
        full_prompt = f"{prompt}\n\nAvoid: {negative_prompt}"

    common = {
        "model": model,
        "prompt": full_prompt,
        "size": size,
        "quality": quality,
    }
    if output_format:
        common["output_format"] = output_format

    if reference_image is None:
        response = client.images.generate(**common)
        return _decode_image_response(response)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = tmp.name
        reference_image.convert("RGB").save(tmp, format="PNG")

    try:
        with open(temp_path, "rb") as image_file:
            response = client.images.edit(
                **common,
                image=image_file,
            )
        return _decode_image_response(response)
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
