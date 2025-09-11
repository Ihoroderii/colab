import os, json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
from openai import OpenAI
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from IPython.display import display


os.environ["HF_TOKEN"] = "REMOVED_TOKENWCiVcWEHseVwmKPnxYYtWOoLhoktEROfhH"
# -------------------# Hugging Face Router Setup (LLM for scenario generation)
# -------------------
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],  # set in Colab: os.environ["HF_TOKEN"] = "your_token"
)

# -------------------
# Scenario generator (LLM)
# -------------------
def generate_scenario():
    completion = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct:novita",  # requires gated access
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a manga scriptwriter. "
                    "Return ONLY valid JSON. "
                    "Each panel must have: context, scene, speaker, speech."
                )
            },
            {
                "role": "user",
                "content": (
                    "Write 2 short manga panels in JSON format. Example:\n"
                    '[{"context": "Rainy night in Tokyo", '
                    '"scene": "Akira sits at a cafe", '
                    '"speaker": "Akira", '
                    '"speech": "I just can\'t believe she\'s gone..."}]'
                )
            },
        ],
        temperature=0.8,
        max_tokens=500,
    )

    raw = completion.choices[0].message.content
    print("Raw LLM Output:", raw)

    # --- Cleaning step ---
    cleaned = raw.strip()
    if cleaned.count("[") > 1:
        parts = cleaned.split("\n")
        merged = []
        for p in parts:
            try:
                data = json.loads(p)
                if isinstance(data, list):
                    merged.extend(data)
            except Exception:
                continue
        return merged

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print("⚠️ Could not parse JSON, returning empty list.")
        return []


# -------------------
# Placeholder: Text-to-Pose
# -------------------
def text_to_pose(scene_text):
    # TODO: Replace with a real text-to-pose model
    return [(250,100),(250,200),(220,300),(280,300),(210,400),(290,400)]

# -------------------
# Draw skeleton pose
# -------------------
def draw_pose(keypoints, size=512):
    skeleton = np.ones((size, size, 3), dtype=np.uint8) * 255
    connections = [(0,1),(1,2),(1,3),(2,4),(3,5)]
    for s, e in connections:
        if s < len(keypoints) and e < len(keypoints):
            cv2.line(skeleton, keypoints[s], keypoints[e], (0,0,0), 4)
    for p in keypoints:
        cv2.circle(skeleton, p, 6, (0,0,255), -1)
    return Image.fromarray(cv2.cvtColor(skeleton, cv2.COLOR_BGR2RGB))

# -------------------
# Face detection
# -------------------
def detect_face(img_pil):
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
        return (x, y, w, h)
    return None

# -------------------
# Add speech bubble
# -------------------
def add_speech_bubble(img_pil, text):
    face = detect_face(img_pil)
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    if face:
        x, y, w, h = face
        bubble_x, bubble_y = x + w + 20, y
    else:
        bubble_x, bubble_y = 50, 50

    text_box = draw.textbbox((bubble_x, bubble_y), text, font=font)
    draw.rectangle(
        (text_box[0]-10, text_box[1]-10, text_box[2]+10, text_box[3]+10),
        fill=(255,255,255)
    )
    draw.text((bubble_x, bubble_y), text, font=font, fill=(0,0,0))
    return img_pil

# -------------------
# Add narration box
# -------------------
def add_caption_box(img_pil, text):
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    x, y = 20, 20
    text_box = draw.textbbox((x, y), text, font=font)
    draw.rectangle(
        (text_box[0]-10, text_box[1]-10, text_box[2]+10, text_box[3]+10),
        fill=(255, 255, 200)
    )
    draw.text((x, y), text, font=font, fill=(0, 0, 0))
    return img_pil

# -------------------
# Generate manga panel with SD3.5 + ControlNet SD3.5
# -------------------
def generate_manga_panel(context, scene_prompt, speaker, speech_text):
    keypoints = text_to_pose(scene_prompt)
    pose_image = draw_pose(keypoints)

    controlnet = ControlNetModel.from_pretrained(
        "stabilityai/stable-diffusion-3.5-controlnets",  # ✅ ControlNet for SD3.5
        torch_dtype=torch.float16
    )
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",  # ✅ Base model SD3.5
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

    negative_prompt = (
        "blurry, deformed face, extra eyes, extra limbs, bad anatomy, "
        "ugly, poorly drawn, distorted"
    )

    prompt = (
        f"{context}. {scene_prompt}, "
        "anime character, manga art style, detailed lines, full body"
    )

    result_img = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=pose_image,
        num_inference_steps=30
    ).images[0]

    # Add narration + speech bubble
    caption_text = f"{context}. {scene_prompt}"
    result_img = add_caption_box(result_img, caption_text)

    bubble_text = f"{speaker}: {speech_text}"
    result_img = add_speech_bubble(result_img, bubble_text)

    return result_img

# -------------------
# MAIN
# -------------------
if __name__ == "__main__":
    scenes = generate_scenario()

    for i, sc in enumerate(scenes):
        context = sc.get("context", "")
        scene = sc.get("scene", "A person is standing")
        speaker = sc.get("speaker", "Character")
        speech = sc.get("speech", "...")

        print(f"[Panel {i+1}] {context} | {scene} → {speaker}: {speech}")
        panel = generate_manga_panel(context, scene, speaker, speech)
        panel.save(f"panel_{i+1}.png")
        display(panel)
