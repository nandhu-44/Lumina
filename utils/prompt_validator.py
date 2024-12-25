import os
from typing import Tuple, Optional
import logging
from dataclasses import dataclass
import google.generativeai as genai
from transformers import pipeline
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    is_safe: bool
    message: str
    score: float


@dataclass
class EnhancementResult:
    success: bool
    prompt: str
    error: Optional[str] = None


class PromptValidator:
    def __init__(self, enhancer: str = "openai", max_retries: int = 3):
        """
        Initialize the PromptValidator with configurable settings.

        Args:
            enhancer: Which service to use for prompt enhancement ("openai" or "gemini")
            max_retries: Maximum number of retries for API calls
        """
        self.max_retries = max_retries
        self.enhancer = enhancer
        self._initialize_apis()
        self._setup_classifiers()

    def _initialize_apis(self) -> None:
        """Initialize API clients with error handling."""
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")

        if not self.openai_key or not self.gemini_key:
            missing_keys = []
            if not self.openai_key:
                missing_keys.append("OPENAI_API_KEY")
            if not self.gemini_key:
                missing_keys.append("GEMINI_API_KEY")
            raise RuntimeError(f"Missing required API keys: {', '.join(missing_keys)}")

        try:
            genai.configure(api_key=self.gemini_key)
            self.gemini = genai.GenerativeModel("gemini-pro")
            self.openai_client = AsyncOpenAI(api_key=self.openai_key)
        except Exception as e:
            logger.error(f"Failed to initialize API clients: {e}")
            raise

    def _setup_classifiers(self) -> None:
        """Set up the content classification pipeline."""
        try:
            self.classifier = pipeline(
                "text-classification",
                model="michellejieli/inappropriate_text_classifier",
                device="cuda" if os.getenv("USE_GPU", "0") == "1" else "cpu",
            )
        except Exception as e:
            logger.error(f"Failed to initialize content classifier: {e}")
            raise

    async def is_safe_prompt(self, prompt: str) -> ValidationResult:
        """
        Check if a prompt is safe to use.

        Args:
            prompt: The prompt to validate

        Returns:
            ValidationResult containing safety status and details
        """
        if not prompt or not prompt.strip():
            return ValidationResult(
                is_safe=False, message="Prompt cannot be empty", score=0.0
            )

        try:
            result = self.classifier(prompt)
            logger.info(f"Prompt classification result: {result}")

            score = result[0]["score"]
            is_inappropriate = result[0]["label"] == "inappropriate" and score > 0.85

            return ValidationResult(
                is_safe=not is_inappropriate,
                message=(
                    ""
                    if not is_inappropriate
                    else "This type of content is not supported."
                ),
                score=score,
            )

        except Exception as e:
            logger.error(f"Prompt classification failed: {e}")
            return ValidationResult(
                is_safe=False, message=f"Failed to validate prompt: {str(e)}", score=0.0
            )

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def enhance_prompt_openai(self, prompt: str) -> EnhancementResult:
        """
        Enhance a prompt using OpenAI's GPT-4.

        Args:
            prompt: Original prompt to enhance

        Returns:
            EnhancementResult containing the enhanced prompt or error details
        """
        prompt_template = f"""
        You are a professional artist advisor helping to create image generation prompts.
        Enhance this prompt by adding artistic details and style elements.
        Keep it concise (under 20 words), high quality, and family-friendly.
        Original prompt: '{prompt}'
        Return only the enhanced prompt without any explanation or quotes.
        Focus on key visual elements and photography/art terminology.
        """

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an artistic prompt enhancer.",
                    },
                    {"role": "user", "content": prompt_template},
                ],
                max_tokens=100,
                temperature=0.7,
            )

            enhanced = response.choices[0].message.content.strip().strip("\"'")
            logger.info(f"OpenAI enhanced prompt: {enhanced}")

            return EnhancementResult(success=True, prompt=enhanced)

        except Exception as e:
            logger.error(f"OpenAI prompt enhancement failed: {e}")
            return EnhancementResult(
                success=False,
                prompt=prompt,
                error=f"OpenAI enhancement failed: {str(e)}",
            )

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def enhance_prompt_gemini(self, prompt: str) -> EnhancementResult:
        """
        Enhance a prompt using Google's Gemini model.

        Args:
            prompt: Original prompt to enhance

        Returns:
            EnhancementResult containing the enhanced prompt or error details
        """
        prompt_template = f"""
        Enhance this prompt for an AI image generator by adding artistic details and style elements.
        Keep it concise (under 50 words), high quality, and family-friendly.
        Original prompt: '{prompt}'
        Return only the enhanced prompt without explanation or quotes.
        Focus on key visual elements and avoid lengthy descriptions.
        """

        try:
            response = self.gemini.generate_content(prompt_template)
            enhanced = response.text.strip().strip("\"'")

            words = enhanced.split()
            if len(words) > 50:
                enhanced = " ".join(words[:50])

            logger.info(f"Gemini enhanced prompt: {enhanced}")
            return EnhancementResult(success=True, prompt=enhanced)

        except Exception as e:
            logger.error(f"Gemini prompt enhancement failed: {e}")
            return EnhancementResult(
                success=False,
                prompt=prompt,
                error=f"Gemini enhancement failed: {str(e)}",
            )

    async def enhance_prompt(self, prompt: str) -> EnhancementResult:
        """
        Enhance a prompt using the configured enhancement service.

        Args:
            prompt: Original prompt to enhance

        Returns:
            EnhancementResult containing the enhanced prompt or error details
        """
        try:
            if self.enhancer == "openai":
                return await self.enhance_prompt_openai(prompt)
            elif self.enhancer == "gemini":
                return await self.enhance_prompt_gemini(prompt)
            else:
                return EnhancementResult(
                    success=False,
                    prompt=prompt,
                    error=f"Unknown enhancer type: {self.enhancer}",
                )
        except Exception as e:
            logger.error(f"Prompt enhancement failed: {e}")
            return EnhancementResult(
                success=False, prompt=prompt, error=f"Enhancement failed: {str(e)}"
            )
