from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MangaStyleConfig:
    style_id: str = "manga_bw_v1"
    width: int = 768
    height: int = 1024
    grayscale: bool = True

    gamma: float = 0.96
    contrast: float = 1.08
    autocontrast_cutoff: float = 1.0

    sharpen_radius: float = 1.2
    sharpen_percent: int = 110
    sharpen_threshold: int = 3

    black_threshold: int = 50
    dark_tone_threshold: int = 110
    light_tone_threshold: int = 175

    grain_strength: float = 0.02
    grain_seed: int = 12345

    screentone_style: str = "dots_medium"
    dark_tone_step: int = 8
    light_tone_step: int = 13
    dark_tone_radius: int = 2
    light_tone_radius: int = 1

    page_autocontrast_cutoff: float = 0.3
    page_contrast: float = 1.02

    @property
    def target_size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def to_dict(self) -> dict:
        return asdict(self)


StyleNormalizationSettings = MangaStyleConfig
