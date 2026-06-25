"""Diffusion pipeline loading, device/dtype selection, and caching."""
from __future__ import annotations
from typing import Optional
import torch
from diffusers import AutoPipelineForImage2Image, StableDiffusion3Pipeline, StableDiffusionPipeline
from huggingface_hub.errors import GatedRepoError
from requests.exceptions import HTTPError


def select_device_and_dtype(preferred: str) -> tuple[str, torch.dtype]:
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda", torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def load_pipeline_with_fallback(
    model_id: str,
    device: str,
    dtype: torch.dtype,
    token: Optional[str],
    fallback_model: str = "runwayml/stable-diffusion-v1-5",
):
    def _load(model: str):
        print(f"Loading model {model} on {device} with dtype {dtype}")
        if "stable-diffusion-3" in model:
            return StableDiffusion3Pipeline.from_pretrained(model, torch_dtype=dtype, token=token).to(device)
        else:
            return StableDiffusionPipeline.from_pretrained(model, torch_dtype=dtype, token=token, safety_checker=None).to(device)

    try:
        return _load(model_id)
    except GatedRepoError as e:
        return StableDiffusionPipeline.from_pretrained(fallback_model, torch_dtype=dtype, safety_checker=None).to(device)
    except HTTPError:
        return StableDiffusionPipeline.from_pretrained(fallback_model, torch_dtype=dtype, safety_checker=None).to(device)
    except Exception:
        return StableDiffusionPipeline.from_pretrained(fallback_model, torch_dtype=dtype, safety_checker=None).to(device)


def load_img2img_pipeline_with_fallback(
    model_id: str,
    device: str,
    dtype: torch.dtype,
    token: Optional[str],
    fallback_model: str = "runwayml/stable-diffusion-v1-5",
):
    def _load(model: str):
        print(f"Loading img2img model {model} on {device} with dtype {dtype}")
        kwargs = {"torch_dtype": dtype, "token": token}
        if "stable-diffusion-3" not in model:
            kwargs["safety_checker"] = None
        return AutoPipelineForImage2Image.from_pretrained(model, **kwargs).to(device)

    try:
        return _load(model_id)
    except GatedRepoError:
        return AutoPipelineForImage2Image.from_pretrained(
            fallback_model,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(device)
    except HTTPError:
        return AutoPipelineForImage2Image.from_pretrained(
            fallback_model,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(device)
    except Exception:
        return AutoPipelineForImage2Image.from_pretrained(
            fallback_model,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(device)


# Simple module-level cache
_DEVICE_DTYPE: Optional[tuple[str, torch.dtype]] = None
_CACHED_PIPELINE = None
_CACHED_IMG2IMG_PIPELINE = None


def get_cached_pipeline(
    model_id: str,
    preferred_device: str,
    token: Optional[str],
    fallback_model: str = "runwayml/stable-diffusion-v1-5",
):
    global _DEVICE_DTYPE, _CACHED_PIPELINE
    if _DEVICE_DTYPE is None:
        _DEVICE_DTYPE = select_device_and_dtype(preferred_device)
    if _CACHED_PIPELINE is None:
        device, dtype = _DEVICE_DTYPE
        _CACHED_PIPELINE = load_pipeline_with_fallback(model_id, device, dtype, token, fallback_model)
    return _CACHED_PIPELINE, _DEVICE_DTYPE


def get_cached_img2img_pipeline(
    model_id: str,
    preferred_device: str,
    token: Optional[str],
    fallback_model: str = "runwayml/stable-diffusion-v1-5",
):
    global _DEVICE_DTYPE, _CACHED_IMG2IMG_PIPELINE
    if _DEVICE_DTYPE is None:
        _DEVICE_DTYPE = select_device_and_dtype(preferred_device)
    if _CACHED_IMG2IMG_PIPELINE is None:
        device, dtype = _DEVICE_DTYPE
        _CACHED_IMG2IMG_PIPELINE = load_img2img_pipeline_with_fallback(model_id, device, dtype, token, fallback_model)
    return _CACHED_IMG2IMG_PIPELINE, _DEVICE_DTYPE
