from typing import Tuple

def get_manhwa_prompts(context: str, scene_prompt: str) -> Tuple[str, str]:
    prompt = (
        f"{context}. {scene_prompt}, Korean manhwa style, detailed coloring, dramatic lighting, "
        f"highly detailed faces, modern fashion, webtoon format, digital art, clean lineart, "
        f"vibrant colors, professional coloring, high contrast, cinematic composition"
    )
    negative_prompt = (
        "blurry, deformed face, extra eyes, extra limbs, bad anatomy, ugly, distorted, "
        "chibi style, super deformed, anime style, manga style, sketch, watercolor"
    )
    return prompt, negative_prompt
