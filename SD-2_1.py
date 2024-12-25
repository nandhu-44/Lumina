import os
import torch

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("Using CPU")

small_model = "stabilityai/stable-diffusion-2-1"

pipe = StableDiffusionPipeline.from_pretrained(small_model, torch_dtype=torch.bfloat16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()
pipe = pipe.to(device)

prompts = ["a programmer touching grass"]

results = pipe(
    prompts, num_inference_steps=50, guidance_scale=3.5, height=512, width=512
)


images = results.images
results_folder = "generations-2_1"
if not os.path.exists(results_folder):
    os.makedirs(results_folder, exist_ok=True)

for i, img in enumerate(images):
    img.save(f"{results_folder}/image_{i}.png")
