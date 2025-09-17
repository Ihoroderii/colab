"""Diffusion pipeline loading, device/dtype selection, and caching."""
from __future__ import annotations
from typing import Tuple, Optional
import torch
from diffusers import StableDiffusion3Pipeline, StableDiffusionPipeline
from huggingface_hub.errors import GatedRepoError
from requests.exceptions import HTTPError


def select_device_and_dtype(preferred: str) -> tuple[str, torch.dtype]:
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda", torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def load_pipeline_with_fallback(model_id: str, device: str, dtype: torch.dtype, token: Optional[str]):
    def _load(model: str):
        if "stable-diffusion-3" in model:
            return StableDiffusion3Pipeline.from_pretrained(model, torch_dtype=dtype, token=token).to(device)
        else:
            return StableDiffusionPipeline.from_pretrained(model, torch_dtype=dtype, token=token, safety_checker=None).to(device)

    try:
        return _load(model_id)
    except GatedRepoError as e:
        # Fallback to an open model
        fallback_model = "stabilityai/stable-diffusion-2-1"
        return StableDiffusionPipeline.from_pretrained(fallback_model, torch_dtype=dtype, safety_checker=None).to(device)
    except HTTPError:
        fallback_model = "stabilityai/stable-diffusion-2-1"
        return StableDiffusionPipeline.from_pretrained(fallback_model, torch_dtype=dtype, safety_checker=None).to(device)
    except Exception:
        fallback_model = "stabilityai/stable-diffusion-2-1"
        return StableDiffusionPipeline.from_pretrained(fallback_model, torch_dtype=dtype, safety_checker=None).to(device)


# Simple module-level cache
_DEVICE_DTYPE: Optional[tuple[str, torch.dtype]] = None
_CACHED_PIPELINE = None


def get_cached_pipeline(model_id: str, preferred_device: str, token: Optional[str]):
    global _DEVICE_DTYPE, _CACHED_PIPELINE
    if _DEVICE_DTYPE is None:
        _DEVICE_DTYPE = select_device_and_dtype(preferred_device)
    if _CACHED_PIPELINE is None:
        device, dtype = _DEVICE_DTYPE
        _CACHED_PIPELINE = load_pipeline_with_fallback(model_id, device, dtype, token)
    return _CACHED_PIPELINE, _DEVICE_DTYPE
