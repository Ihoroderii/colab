"""Optional validator using BLIP (captioning) and CLIP (similarity)."""
from __future__ import annotations
import logging
from typing import List, Optional, Tuple

try:
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration  # type: ignore
except Exception:
    BlipProcessor = None  # type: ignore
    BlipForConditionalGeneration = None  # type: ignore

try:
    import clip  # type: ignore
except Exception:
    clip = None  # type: ignore

from PIL import Image

logger = logging.getLogger(__name__)


class PanelValidator:
    def __init__(
        self,
        device: str = "cpu",
        blip_model_id: str = "Salesforce/blip-image-captioning-large",
        clip_model_id: str = "ViT-B/32",
        threshold: float = 0.7,
    ) -> None:
        self.device = device
        self.threshold = threshold
        self._blip_processor = None
        self._blip_model = None
        self._clip_model = None
        self._clip_preprocess = None
        self._blip_model_id = blip_model_id
        self._clip_model_id = clip_model_id

    def available(self) -> bool:
        return (BlipProcessor is not None and BlipForConditionalGeneration is not None and clip is not None)

    def load(self) -> None:
        if not self.available():
            logger.warning("Validation dependencies not available; skipping.")
            return
        self._blip_processor = BlipProcessor.from_pretrained(self._blip_model_id)
        self._blip_model = BlipForConditionalGeneration.from_pretrained(self._blip_model_id).to(self.device)
        self._clip_model, self._clip_preprocess = clip.load(self._clip_model_id, device=self.device)

    def caption(self, image: Image.Image) -> Optional[str]:
        if self._blip_processor is None or self._blip_model is None:
            return None
        inputs = self._blip_processor(image, return_tensors="pt").to(self.device)
        out = self._blip_model.generate(**inputs)
        caption = self._blip_processor.decode(out[0], skip_special_tokens=True)
        return caption

    def similarity(self, img_a: Image.Image, img_b: Image.Image) -> Optional[float]:
        if self._clip_model is None or self._clip_preprocess is None:
            return None
        a = self._clip_preprocess(img_a).unsqueeze(0).to(self.device)
        b = self._clip_preprocess(img_b).unsqueeze(0).to(self.device)
        e1, e2 = self._clip_model.encode_image(a), self._clip_model.encode_image(b)
        import torch
        sim = torch.cosine_similarity(e1 / e1.norm(), e2 / e2.norm()).item()
        return float(sim)

    def validate(
        self,
        image: Image.Image,
        reference: Optional[Image.Image] = None,
        expected_traits: Optional[List[str]] = None,
    ) -> Tuple[bool, dict]:
        details = {}
        caption = self.caption(image)
        if caption is not None:
            details["caption"] = caption
            if expected_traits:
                missing = [t for t in expected_traits if t.lower() not in caption.lower()]
                details["missing_traits"] = missing
        sim = None
        if reference is not None:
            sim = self.similarity(image, reference)
            if sim is not None:
                details["similarity"] = sim
        passed = True
        if details.get("missing_traits"):
            passed = False
        if sim is not None and sim < self.threshold:
            passed = False
        return passed, details
