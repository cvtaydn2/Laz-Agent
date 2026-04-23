from __future__ import annotations

from typing import Iterable

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agent_core.config import Settings
from agent_core.models import ChatMessage, ModelResponse


class NvidiaClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(httpx.HTTPError),
    )
    def chat(self, messages: Iterable[ChatMessage]) -> ModelResponse:
        if not self.settings.nvidia_api_key:
            raise ValueError("NVIDIA_API_KEY is not configured.")

        payload = {
            "model": self.settings.nvidia_model,
            "temperature": self.settings.temperature,
            "messages": [message.model_dump() for message in messages],
        }

        headers = {
            "Authorization": f"Bearer {self.settings.nvidia_api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(
            base_url=str(self.settings.nvidia_base_url),
            timeout=self.settings.timeout_seconds,
        ) as client:
            response = client.post("/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        choices = data.get("choices", [])
        if not choices:
            raise ValueError("Model response did not include choices.")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Model response did not include text content.")

        usage = data.get("usage", {})
        return ModelResponse(content=content, usage=usage)
