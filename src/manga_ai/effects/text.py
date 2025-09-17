"""Manhwa-specific visual text/effects (moved from root)."""
from typing import Tuple, Optional
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageFont

class ManhwaEffects:
    @staticmethod
    def add_screentone(image: Image.Image, pattern_type: str = 'dots', opacity: float = 0.3, scale: float = 1.0) -> Image.Image:
        width, height = image.size
        pattern = Image.new('L', (width, height), 255)
        draw = ImageDraw.Draw(pattern)
        if pattern_type == 'dots':
            spacing = int(10 * scale)
            for x in range(0, width, spacing):
                for y in range(0, height, spacing):
                    draw.ellipse([x, y, x+3, y+3], fill=0)
        elif pattern_type == 'lines':
            spacing = int(8 * scale)
            for y in range(0, height, spacing):
                draw.line([(0, y), (width, y)], fill=0, width=1)
        elif pattern_type == 'crosshatch':
            spacing = int(15 * scale)
            for offset in range(-height, width + height, spacing):
                draw.line([(offset, 0), (offset + height, height)], fill=0, width=1)
                draw.line([(offset, height), (offset + height, 0)], fill=0, width=1)
        pattern = pattern.convert('RGBA')
        pattern.putalpha(int(255 * opacity))
        return Image.alpha_composite(image.convert('RGBA'), pattern)

    @staticmethod
    def add_speed_lines(image: Image.Image, direction: str = 'horizontal', intensity: float = 0.5, focus_point: Optional[Tuple[int, int]] = None) -> Image.Image:
        width, height = image.size
        lines = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(lines)
        if direction == 'radial':
            if focus_point is None:
                focus_point = (width // 2, height // 2)
            num_lines = int(50 * intensity)
            for _ in range(num_lines):
                angle = random.uniform(0, 2 * np.pi)
                length = random.uniform(0.3, 1.0) * max(width, height)
                end_x = focus_point[0] + length * np.cos(angle)
                end_y = focus_point[1] + length * np.sin(angle)
                draw.line([focus_point, (end_x, end_y)], fill=(255, 255, 255, int(100 * intensity)), width=2)
        else:
            spacing = int(20 * (1 - intensity))
            if direction == 'horizontal':
                for y in range(0, height, spacing):
                    draw.line([(0, y), (width, y)], fill=(255, 255, 255, int(100 * intensity)), width=2)
            else:
                for x in range(0, width, spacing):
                    draw.line([(x, 0), (x, height)], fill=(255, 255, 255, int(100 * intensity)), width=2)
        lines = lines.filter(ImageFilter.GaussianBlur(radius=2))
        return Image.alpha_composite(image.convert('RGBA'), lines)

    @staticmethod
    def add_dramatic_lighting(image: Image.Image, style: str = 'dramatic', intensity: float = 0.5) -> Image.Image:
        import numpy as np
        import cv2
        img_array = np.array(image)
        if style == 'dramatic':
            from PIL import ImageEnhance
            contrast = ImageEnhance.Contrast(image)
            image = contrast.enhance(1.0 + intensity)
            rows, cols = img_array.shape[:2]
            kernel_x = cv2.getGaussianKernel(cols, cols/4)
            kernel_y = cv2.getGaussianKernel(rows, rows/4)
            kernel = kernel_y * kernel_x.T
            mask = 255 * kernel / np.linalg.norm(kernel)
            vignette = np.copy(img_array)
            for i in range(3):
                vignette[:, :, i] = vignette[:, :, i] * mask
            img_array = cv2.addWeighted(img_array, 1 - intensity/2, vignette, intensity/2, 0)
            return Image.fromarray(img_array)
        elif style == 'soft':
            import cv2
            blur = cv2.GaussianBlur(img_array, (0, 0), 10)
            img_array = cv2.addWeighted(img_array, 1 - intensity/3, blur, intensity/3, 0)
            return Image.fromarray(img_array)
        elif style == 'contrast':
            from PIL import ImageEnhance
            contrast = ImageEnhance.Contrast(image)
            image = contrast.enhance(1.0 + intensity)
            brightness = ImageEnhance.Brightness(image)
            image = brightness.enhance(1.0 + intensity/2)
            return image
        return image

    @staticmethod
    def add_text_effects(image: Image.Image, text: str, position: Tuple[int, int], effect_type: str = 'emphasis', size: int = 30) -> Image.Image:
        result = image.copy()
        draw = ImageDraw.Draw(result)
        if effect_type == 'emphasis':
            x, y = position
            for offset_x, offset_y in [(1,1), (-1,-1), (1,-1), (-1,1)]:
                draw.text((x + offset_x, y + offset_y), text, fill='black', font=ImageFont.load_default())
            draw.text(position, text, fill='white', font=ImageFont.load_default())
        elif effect_type == 'sound':
            x, y = position
            for i, char in enumerate(text):
                draw.text((x + i*15, y), char, fill='white', font=ImageFont.load_default())
        elif effect_type == 'thought':
            x, y = position
            bubble_padding = 10
            text_bbox = draw.textbbox((x, y), text, font=ImageFont.load_default())
            draw.ellipse([text_bbox[0] - bubble_padding, text_bbox[1] - bubble_padding, text_bbox[2] + bubble_padding, text_bbox[3] + bubble_padding], fill='white', outline='black')
            draw.text(position, text, fill='black', font=ImageFont.load_default())
            for i in range(3):
                size = 6 - i*2
                draw.ellipse([x - 20 - i*10, y + 20 + i*10, x - 20 - i*10 + size, y + 20 + i*10 + size], fill='white', outline='black')
        return result
