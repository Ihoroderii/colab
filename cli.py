"""Shim CLI to keep old imports working while using src package."""
from src.manga_ai.cli import parse_args, update_config_from_args  # type: ignore