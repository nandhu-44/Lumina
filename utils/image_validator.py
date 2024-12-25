from transformers import pipeline
import numpy as np


class ImageValidator:
    def __init__(self):
        self.nsfw_detector = pipeline(
            "image-classification", model="Falconsai/nsfw_image_detection"
        )

    def is_safe_image(self, image) -> tuple[bool, str]:
        try:
            # Convert PIL image to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            result = self.nsfw_detector(image)
            print(f"Image classification result: {result}")  # Debug logging

            # Get both safe and unsafe scores
            safe_score = next((x["score"] for x in result if x["label"] == "safe"), 0)
            unsafe_score = next((x["score"] for x in result if x["label"] == "nsfw"), 0)

            print(
                f"Safe score: {safe_score}, Unsafe score: {unsafe_score}"
            )  # Debug logging

            # More lenient threshold - only block if very confident about NSFW
            is_safe = safe_score > 0.3 or unsafe_score < 0.8

            if not is_safe:
                return False, "Image validation failed - generating new image"
            return True, ""

        except Exception as e:
            print(f"Warning: Image validation failed: {e}")
            return True, ""
