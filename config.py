"""Shim to keep imports stable while migrating to src package."""
from manga_ai.config import (  # type: ignore
    Config,
    ModelConfig,
    GenerationConfig,
    StyleConfig,
    OutputConfig,
    ScenarioConfig,
    ExportConfig,
    ValidationConfig,
)

# Maintain same API
config = Config.from_env()