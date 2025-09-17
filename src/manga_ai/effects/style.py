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

    @staticmethod
    def add_border(img: Image.Image, border_width: int = 0, border_color: str = "#000000") -> Image.Image:
        if border_width <= 0:
            return img
        w, h = img.size
        out = Image.new("RGB", (w + 2 * border_width, h + 2 * border_width), border_color)
        out.paste(img, (border_width, border_width))
        return out

    @staticmethod
    def pad_to_square(img: Image.Image, size: int, fill_color: str = "#ffffff") -> Image.Image:
        w, h = img.size
        if w == h == size:
            return img
        out = Image.new("RGB", (size, size), fill_color)
        # Fit the image inside the square without scaling (only pad). If larger, center-crop.
        x = max(0, (size - w) // 2)
        y = max(0, (size - h) // 2)
        if w <= size and h <= size:
            out.paste(img, (x, y))
            return out
        # If either dimension exceeds size, center-crop the image to size
        left = max(0, (w - size) // 2)
        top = max(0, (h - size) // 2)
        crop = img.crop((left, top, left + size, top + size))
        return crop

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
        # Choose font (whisper uses slightly smaller text)
        use_font = self.font
        if btype == "whisper":
            try:
                size = max(14, int(getattr(self.font, "size", 24) * 0.85))
                # Prefer explicit font_path if provided
                if self.font_path and os.path.exists(self.font_path):
                    use_font = ImageFont.truetype(self.font_path, size)
            except Exception:
                use_font = self.font

        # Measure text size with the selected font
        tb = draw.textbbox((0, 0), text, font=use_font)
        tw = (tb[2] - tb[0])
        th = (tb[3] - tb[1])
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

        def cloud_bubble(draw_obj, box, fill=(255,255,255,255), outline=(0,0,0,255)):
            """Draw a cloud-like perimeter by tiling soft circles around the text bounds."""
            x1, y1, x2, y2 = box
            w = max(30, x2 - x1)
            h = max(24, y2 - y1)
            # Inflate box slightly so circles don't overlap text too tightly
            pad = 10
            x1 -= pad; y1 -= pad; x2 += pad; y2 += pad
            w = x2 - x1; h = y2 - y1
            # Circle count along perimeter based on size
            horiz = max(6, w // 24)
            vert = max(4, h // 24)
            r = max(8, min(w, h) // 8)  # radius of small circles
            # Top edge
            for i in range(int(horiz) + 1):
                cx = int(x1 + i * (w / max(1, horiz)))
                cy = y1
                draw_obj.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=outline)
            # Bottom edge
            for i in range(int(horiz) + 1):
                cx = int(x1 + i * (w / max(1, horiz)))
                cy = y2
                draw_obj.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=outline)
            # Left edge
            for j in range(int(vert) + 1):
                cx = x1
                cy = int(y1 + j * (h / max(1, vert)))
                draw_obj.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=outline)
            # Right edge
            for j in range(int(vert) + 1):
                cx = x2
                cy = int(y1 + j * (h / max(1, vert)))
                draw_obj.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=outline)
            # Fill center softly by a rounded rectangle to avoid gaps
            inner = (x1 + r//2, y1 + r//2, x2 - r//2, y2 - r//2)
            rounded(draw_obj, inner, radius=min(20, r), fill=fill, outline=None, width=0)

        # Draw bubble based on type
        if btype == "sfx":
            # Sound effect: no bubble, big bold outlined text, optional slight rotation
            sfx_img = base.copy()
            draw_sfx = ImageDraw.Draw(sfx_img)
            try:
                # Derive a larger temporary font size based on text length
                approx = max(28, min(72, 60 - len(text) // 2))
                font = self.bold_font
                # Prefer explicit font_path if provided
                if self.font_path and os.path.exists(self.font_path):
                    font = ImageFont.truetype(self.font_path, approx)
            except Exception:
                font = self.bold_font

            # Outline effect by drawing offsets
            outline_color = (0, 0, 0, 255)
            fill_color = (255, 255, 255, 255)
            offsets = [(-2,0),(2,0),(0,-2),(0,2),(-2,-2),(2,2),(-2,2),(2,-2)]
            for dx, dy in offsets:
                draw_sfx.text((x + dx, y + dy), text, font=font, fill=outline_color)
            draw_sfx.text((x, y), text, font=font, fill=fill_color)
            return sfx_img.convert("RGB")
        elif btype == "thought":
            # Cloud-like perimeter around the text area
            cloud_bubble(draw, rect, fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))
            # Thought tail (series of small bubbles)
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
            # Narration / caption: simple rectangular box, no tail
            draw.rectangle(rect, fill=(255, 247, 204, 255), outline=(0, 0, 0, 255), width=2)
            text_fill = (0, 0, 0, 255)
        else:  # 'speech' normal dialogue: smooth oval/ellipse
            draw.ellipse(rect, fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))
            text_fill = (0, 0, 0, 255)

        # Draw text and optional speaker label
        draw.text((x + padding, y + padding), text, font=use_font, fill=text_fill)
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
