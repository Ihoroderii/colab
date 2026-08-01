from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AIRestyleRequest:
    image: bytes
    reference_style: bytes | None = None
    prompt: str = (
        "convert to the established black-and-white manga style, clean consistent ink lines, "
        "matching line weight, matching screentones and contrast, preserve composition, "
        "preserve character identity, preserve background geometry"
    )
    strength: float = 0.2


class AIRestyler:
    """Boundary for optional low-strength AI restyling.

    Deterministic post-processing should handle normal tone and line consistency.
    This class exists for future provider-specific implementations when a panel is
    stylistically too far away, such as painterly output among manga ink panels.
    """

    async def restyle(self, request: AIRestyleRequest) -> bytes:
        raise NotImplementedError("AI restyling provider is not configured.")
