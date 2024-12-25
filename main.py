from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from utils.prompt_validator import PromptValidator
from utils.image_validator import ImageValidator
from utils.websocket_manager import WebSocketManager
from utils.pipeline_callback import GenerationProgress
import base64
from io import BytesIO
import uvicorn
import os
from dotenv import load_dotenv
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

load_dotenv()

required_env_vars = ["OPENAI_API_KEY", "GEMINI_API_KEY", "HF_TOKEN"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise RuntimeError(
        f"Missing required environment variables: {', '.join(missing_vars)}"
    )

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


class PromptRequest(BaseModel):
    prompt: str


class ImageResponse(BaseModel):
    success: bool
    message: str
    image: str | None = None
    enhanced_prompt: str | None = None


class GenerationConfig(BaseModel):
    prompt: str
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512

    @classmethod
    def get_default_config(cls):
        return {
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "height": 512,
            "width": 512,
        }

    @classmethod
    def get_valid_ranges(cls):
        return {
            "num_inference_steps": {"min": 20, "max": 50},
            "guidance_scale": {"min": 1.0, "max": 10.0},
            "height": {"min": 384, "max": 768, "step": 128},
            "width": {"min": 384, "max": 768, "step": 128},
        }

    def validate_and_normalize(self) -> Dict[str, Any]:
        ranges = self.get_valid_ranges()
        defaults = self.get_default_config()
        validated = {}

        for key, value in self.dict().items():
            if key == "prompt":
                validated[key] = value
                continue

            valid_range = ranges.get(key, {})
            if not valid_range:
                validated[key] = defaults[key]
                continue

            min_val = valid_range.get("min")
            max_val = valid_range.get("max")
            step = valid_range.get("step")

            if value is None:
                validated[key] = defaults[key]
            elif step:
                value = round(value / step) * step
                validated[key] = max(min_val, min(max_val, value))
            else:
                validated[key] = max(min_val, min(max_val, value))

        return validated


app = FastAPI(title="Lumina AI Art Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipe = None
prompt_validator = None
image_validator = None
websocket_manager = WebSocketManager()


@app.on_event("startup")
async def load_models():
    global pipe, prompt_validator, image_validator
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Use "stabilityai/stable-diffusion-3.5-large" for better quality but needs higher resources
        pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1", torch_dtype=torch.bfloat16
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_attention_slicing()
        pipe = pipe.to(device)

        prompt_validator = PromptValidator()
        image_validator = ImageValidator()

        print("All models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise RuntimeError(f"Failed to load models: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": pipe is not None}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        websocket_manager.disconnect(websocket)


@app.post("/generate", response_model=ImageResponse)
async def generate_image(config: GenerationConfig):
    if not pipe or not prompt_validator or not image_validator:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        validated_config = config.validate_and_normalize()
        prompt = validated_config.pop("prompt")

        await websocket_manager.broadcast(
            {"type": "status", "data": "Validating prompt..."}
        )

        validation_result = await prompt_validator.is_safe_prompt(prompt)
        if not validation_result.is_safe:
            await websocket_manager.broadcast(
                {"type": "error", "data": validation_result.message}
            )
            return ImageResponse(success=False, message=validation_result.message)

        await websocket_manager.broadcast(
            {"type": "status", "data": "Enhancing prompt..."}
        )

        try:
            enhancement_result = await prompt_validator.enhance_prompt(prompt)
            await websocket_manager.broadcast(
                {"type": "prompt", "data": enhancement_result.prompt}
            )
            enhanced_prompt = enhancement_result.prompt
        except Exception as e:
            enhanced_prompt = prompt
            error_msg = f"Prompt enhancement failed: {str(e)}"
            await websocket_manager.broadcast({"type": "error", "data": error_msg})
            print(error_msg)

        await websocket_manager.broadcast(
            {"type": "status", "data": "Generating image..."}
        )

        progress = GenerationProgress(websocket_manager, config.num_inference_steps)

        results = pipe(
            [enhanced_prompt],
            **validated_config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            callback=progress.callback,
            callback_steps=1,
        )

        is_safe, message = image_validator.is_safe_image(results.images[0])
        if not is_safe:
            await websocket_manager.broadcast({"type": "error", "data": message})
            return ImageResponse(success=False, message=message)

        buffered = BytesIO()
        results.images[0].save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        await websocket_manager.broadcast(
            {"type": "status", "data": "Generation complete!"}
        )

        return ImageResponse(
            success=True,
            message="Image generated successfully",
            image=img_str,
            enhanced_prompt=enhanced_prompt,
        )

    except Exception as e:
        error_msg = str(e)
        await websocket_manager.broadcast({"type": "error", "data": error_msg})
        raise HTTPException(status_code=500, detail=error_msg)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
