from typing import Optional, Tuple
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

class ManhwaStyler:
    """Handle manhwa-specific styling and formatting"""

    def __init__(self, font_path: Optional[str] = None):
        self.font_path = font_path
        self._init_fonts()

    def _init_fonts(self, size: int = 24, bold_size: int = 20):
        try:
            if self.font_path:
                self.font = ImageFont.truetype(self.font_path, size)
                self.bold_font = ImageFont.truetype(self.font_path, bold_size)
            else:
                font_paths = [
                    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                    "/usr/local/share/fonts/NanumGothic.ttf",
                    "./fonts/NanumGothic.ttf",
                ]
                for path in font_paths:
                    if os.path.exists(path):
                        self.font = ImageFont.truetype(path, size)
                        self.bold_font = ImageFont.truetype(path, bold_size)
                        return
                self.font = ImageFont.load_default()
                self.bold_font = ImageFont.load_default()
        except Exception:
            self.font = ImageFont.load_default()
            self.bold_font = self.font

    def adjust_panel(self, image: Image.Image, target_width: int = 800) -> Image.Image:
        width, height = image.size
        aspect_ratio = width / height
        if aspect_ratio < 2.0:
            new_height = int(target_width / 2.5)
            image = image.resize((target_width, new_height), Image.Resampling.LANCZOS)
        return image

    def add_text(
        self,
        img: Image.Image,
        text: str,
        speaker: Optional[str] = None,
        is_thought: bool = False,
        position: Tuple[int, int] = (20, 20),
        bubble_type: Optional[str] = None,
    ) -> Image.Image:
        """Draw a speech/FX bubble with text.

        bubble_type: 'speech' | 'thought' | 'shout' | 'whisper' | 'narration' | None
        If None, falls back to is_thought or defaults to 'speech'.
        """
        # Normalize and choose type
        btype = (bubble_type or ("thought" if is_thought else "speech")).lower()

        # Work on RGBA to support semi-transparent fills if needed
        base = img.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        padding = 16
        x, y = position
        # Measure text size
        tw, th = draw.textbbox((0, 0), text, font=self.font)[2:]
        bw, bh = tw + 2 * padding, th + 2 * padding

        rect = (x, y, x + bw, y + bh)

        def rounded(draw_obj, box, radius, fill, outline, width=2):
            draw_obj.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

        def burst(draw_obj, box, spikes=14, amplitude=12, fill=(255,255,255,255), outline=(0,0,0,255), width=2):
            # Starburst polygon around the rectangle center
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            rx = (box[2] - box[0]) / 2 + 6
            ry = (box[3] - box[1]) / 2 + 6
            pts = []
            for i in range(spikes * 2):
                ang = i * np.pi / spikes
                rfx = rx + (amplitude if i % 2 == 0 else 0)
                rfy = ry + (amplitude if i % 2 == 0 else 0)
                px = cx + rfx * np.cos(ang)
                py = cy + rfy * np.sin(ang)
                pts.append((px, py))
            draw_obj.polygon(pts, fill=fill, outline=outline)
            # inner white area for text
            inner = (box[0]+6, box[1]+6, box[2]-6, box[3]-6)
            rounded(draw_obj, inner, 12, fill=fill, outline=outline, width=2)

        def dashed_rounded(draw_obj, box, radius, fill, outline, dash_len=6, gap=6):
            # Draw fill first
            rounded(draw_obj, box, radius, fill=fill, outline=None, width=0)
            # Approximate dashed outline by drawing short lines along rectangle edges
            x1, y1, x2, y2 = box
            # Horizontal top/bottom
            cur = x1 + radius
            while cur < x2 - radius:
                draw_obj.line([(cur, y1), (min(cur + dash_len, x2 - radius), y1)], fill=outline, width=2)
                draw_obj.line([(cur, y2), (min(cur + dash_len, x2 - radius), y2)], fill=outline, width=2)
                cur += dash_len + gap
            # Vertical left/right
            cur = y1 + radius
            while cur < y2 - radius:
                draw_obj.line([(x1, cur), (x1, min(cur + dash_len, y2 - radius))], fill=outline, width=2)
                draw_obj.line([(x2, cur), (x2, min(cur + dash_len, y2 - radius))], fill=outline, width=2)
                cur += dash_len + gap

        # Draw bubble based on type
        if btype == "thought":
            draw.ellipse(rect, fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))
            # thought tail (bubbles)
            for i in range(3):
                r = 10 - 3 * i
                offset = 18 * (i + 1)
                draw.ellipse((x - offset, y + bh + 4 + offset, x - offset + r, y + bh + 4 + offset + r), fill=(255,255,255,255), outline=(0,0,0,255))
            text_fill = (0, 0, 0, 255)
        elif btype == "shout":
            burst(draw, rect, spikes=16, amplitude=14, fill=(255,255,255,255), outline=(0,0,0,255), width=3)
            text_fill = (0, 0, 0, 255)
        elif btype == "whisper":
            dashed_rounded(draw, rect, radius=16, fill=(255,255,255,200), outline=(80,80,80,255), dash_len=6, gap=6)
            text_fill = (20, 20, 20, 255)
        elif btype == "narration":
            # narration box (caption)
            rounded(draw, rect, radius=6, fill=(255, 247, 204, 255), outline=(0, 0, 0, 255), width=2)
            text_fill = (0, 0, 0, 255)
        else:  # 'speech'
            rounded(draw, rect, radius=16, fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=2)
            text_fill = (0, 0, 0, 255)

        # Draw text and optional speaker label
        draw.text((x + padding, y + padding), text, font=self.font, fill=text_fill)
        if speaker and btype != "narration":
            draw.text((x, y - 25), speaker, font=self.bold_font, fill=(0, 0, 0, 255))

        # Composite overlay onto base
        return Image.alpha_composite(base, overlay).convert("RGB")

    def apply_style(self, image: Image.Image) -> Image.Image:
        img_array = np.array(image)
        contrast = 1.2
        brightness = 1.1
        img_array = cv2.convertScaleAbs(img_array, alpha=contrast, beta=brightness)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * 1.2
        img_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return Image.fromarray(img_array.clip(0, 255).astype("uint8"))
