from typing import Tuple

def get_manhwa_prompts(context: str, scene_prompt: str) -> Tuple[str, str]:
    prompt = (
        f"{context}. {scene_prompt}, Korean webtoon manhwa illustration, detailed coloring, dramatic lighting, "
        f"highly detailed faces, modern fashion, webtoon format, digital art, clean lineart, "
        f"vibrant colors, professional coloring, high contrast, cinematic composition, "
        f"non-graphic dramatic tension, no injury, no gore, no weapons"
    )
    negative_prompt = (
        "blurry, deformed face, extra eyes, extra limbs, bad anatomy, ugly, distorted, "
        "chibi style, super deformed, sketch, watercolor, gore, blood, wounds, weapons, explicit violence"
    )
    return prompt, negative_prompt
