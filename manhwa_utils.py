"""Shim to preserve old imports while code lives in src/manga_ai/*"""
from manga_ai.effects.style import ManhwaStyler  # type: ignore
from manga_ai.pipelines.assemble import ManhwaAssembler  # type: ignore
from manga_ai.utils.prompts import get_manhwa_prompts  # type: ignore

__all__ = [
    "ManhwaStyler",
    "ManhwaAssembler",
    "get_manhwa_prompts",
]