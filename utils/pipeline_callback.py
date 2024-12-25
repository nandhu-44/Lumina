import time
from typing import Any, Dict, Union


class GenerationProgress:
    def __init__(self, websocket_manager, total_steps):
        self.websocket_manager = websocket_manager
        self.total_steps = total_steps
        self.start_time = time.time()
        self.last_step = 0

    async def callback(self, step: int, timestep: int, latents: Any) -> None:
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        if step > self.last_step:
            steps_per_second = step / elapsed_time if elapsed_time > 0 else 0
            remaining_steps = self.total_steps - step
            estimated_time = (
                remaining_steps / steps_per_second if steps_per_second > 0 else 0
            )

            progress_data = {
                "type": "progress",
                "data": {
                    "step": step,
                    "total_steps": self.total_steps,
                    "progress": (step / self.total_steps) * 100,
                    "estimated_time": round(estimated_time, 1),
                    "elapsed_time": round(elapsed_time, 1),
                },
            }

            await self.websocket_manager.broadcast(progress_data)
            self.last_step = step
