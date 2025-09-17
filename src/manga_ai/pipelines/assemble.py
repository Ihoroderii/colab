from typing import List, Optional
from PIL import Image

class ManhwaAssembler:
    """Handle panel assembly and chapter composition for manhwa format"""

    @staticmethod
    def assemble_panels(
        panels: List[Image.Image],
        output_path: Optional[str] = None,
        panel_gap: int = 40,
        background_color: tuple[int, int, int] | str = (255, 255, 255),
    ) -> Image.Image:
        if not panels:
            raise ValueError("No panels provided for assembly")
        width = max(p.size[0] for p in panels)
        total_height = sum(p.size[1] for p in panels) + (len(panels) - 1) * panel_gap
        final_img = Image.new("RGB", (width, total_height), background_color)
        y_offset = 0
        for panel in panels:
            x_offset = (width - panel.size[0]) // 2
            final_img.paste(panel, (x_offset, y_offset))
            y_offset += panel.size[1] + panel_gap
        if output_path:
            final_img.save(output_path, quality=95)
        return final_img
