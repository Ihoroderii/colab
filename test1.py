import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

# --- STEP 1: Input Text Prompt ---
scene_prompt = input("Describe the scene (e.g., 'girl jumping arms up'): ")

# --- STEP 2: Text-to-Pose (Mock Library) ---
pose_library = {
    "standing": [(250, 100), (250, 200), (220, 300), (280, 300), (210, 400), (290, 400)],
    "jumping": [(250, 80), (250, 180), (200, 260), (300, 260), (180, 360), (320, 360)],
    "sitting": [(250, 150), (250, 250), (220, 300), (280, 300), (230, 350), (270, 350)]
}

if "jump" in scene_prompt.lower():
    pose = pose_library["jumping"]
elif "sit" in scene_prompt.lower():
    pose = pose_library["sitting"]
else:
    pose = pose_library["standing"]

# --- STEP 3: Draw Pose as Skeleton with OpenCV ---
skeleton = np.ones((512, 512, 3), dtype=np.uint8) * 255
connections = [(0,1), (1,2), (1,3), (2,4), (3,5)]

for start, end in connections:
    cv2.line(skeleton, pose[start], pose[end], (0,0,0), 4)

for point in pose:
    cv2.circle(skeleton, point, 6, (0,0,255), -1)

pose_path = "generated_pose.jpg"
cv2.imwrite(pose_path, skeleton)

plt.imshow(cv2.cvtColor(skeleton, cv2.COLOR_BGR2RGB))
plt.title("Generated Pose (OpenCV)")
plt.axis('off')
plt.show()

# --- STEP 4: Generate Image Using ControlNet ---
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu")
pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

pose_image = Image.open(pose_path).resize((512, 512))
prompt = "anime character, manga art style, detailed lines, full body"
output = pipeline(prompt=prompt, image=pose_image, num_inference_steps=30)
result_img = output.images[0]
result_img.save("generated_manga.jpg")

# --- STEP 5: Add Speech Bubble ---
speech_text = "Let's start the adventure!"
bubble_img = result_img.copy()
draw = ImageDraw.Draw(bubble_img)
font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20)

x, y = 50, 50
text_size = draw.textbbox((x, y), speech_text, font=font)
bg_box = (x - 10, y - 10, text_size[2] + 10, text_size[3] + 10)
draw.rectangle(bg_box, fill=(255, 255, 255))
draw.text((x, y), speech_text, font=font, fill=(0, 0, 0))

bubble_img.save("final_manga_panel.jpg")
plt.imshow(bubble_img)
plt.title("Final Manga Panel")
plt.axis('off')
plt.show()