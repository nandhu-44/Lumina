import os
import torch
from diffusers import  StableDiffusion3Pipeline

large_model = "stabilityai/stable-diffusion-3.5-large"

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN") or ""

pipe = StableDiffusion3Pipeline.from_pretrained(large_model, torch_dtype=torch.bfloat16)
pipe.enable_attention_slicing()
pipe = pipe.to("cuda")

prompt = "a programmer touching grass"

results = pipe(
    prompt,
    num_inference_steps=20,
    guidance_scale=3.5,
    height=512,
    width=512
)

images = results.images
results_folder = "generations-3_5"
if not os.path.exists(results_folder):
    os.makedirs(results_folder, exist_ok=True)

for i, img in enumerate(images):
    img.save(f"{results_folder}/image_{i}.png")