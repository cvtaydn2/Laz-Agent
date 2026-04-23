from __future__ import annotations

import time
from typing import Iterable

import httpx

from agent_core.config import Settings
from agent_core.logger import configure_logger
from agent_core.models import ChatMessage, ModelResponse


class NvidiaInferenceError(RuntimeError):
    def __init__(self, user_message: str, *, error_type: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.error_type = error_type


class NvidiaTimeoutError(NvidiaInferenceError):
    def __init__(self) -> None:
        super().__init__(
            "The model backend timed out while generating a response. Please retry with a smaller request or a narrower workspace context.",
            error_type="timeout",
        )


class NvidiaBackendError(NvidiaInferenceError):
    def __init__(self) -> None:
        super().__init__(
            "The model backend is temporarily unavailable. Please retry in a moment or reduce the request scope.",
            error_type="backend_unavailable",
        )


class NvidiaClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = configure_logger(settings.logs_dir / "editor-agent.log")
        self.max_attempts = 3
        self.max_backoff_seconds = 3.0

    def chat(
        self,
        messages: Iterable[ChatMessage],
        temperature_override: float | str | None = None,
        max_tokens_override: int | None = None,
    ) -> ModelResponse:
        if not self.settings.nvidia_api_key:
            raise ValueError("NVIDIA_API_KEY is not configured.")

        # Models to try in order
        models_to_try = [self.settings.nvidia_model]
        if self.settings.nvidia_fallback_model and self.settings.nvidia_fallback_model != self.settings.nvidia_model:
            models_to_try.append(self.settings.nvidia_fallback_model)

        last_exception: Exception | None = None
        started_at = time.monotonic()

        for model_name in models_to_try:
            payload = {
                "model": model_name,
                "temperature": self._resolve_temperature(temperature_override),
                "max_tokens": self._resolve_max_tokens(max_tokens_override),
                "messages": [message.model_dump() for message in messages],
            }
            timeout = httpx.Timeout(
                connect=min(5.0, self.settings.timeout_seconds),
                read=self.settings.timeout_seconds,
                write=min(10.0, self.settings.timeout_seconds),
                pool=min(5.0, self.settings.timeout_seconds),
            )
            headers = {
                "Authorization": f"Bearer {self.settings.nvidia_api_key}",
                "Content-Type": "application/json",
            }

            self.logger.info(
                "NVIDIA request starting: model=%s timeout_seconds=%.1f max_attempts=%s",
                model_name,
                self.settings.timeout_seconds,
                self.max_attempts,
            )

            for attempt in range(1, self.max_attempts + 1):
                try:
                    self.logger.info("Attempt %d/%d for model %s", attempt, self.max_attempts, model_name)
                    with httpx.Client(
                        base_url=str(self.settings.nvidia_base_url),
                        timeout=timeout,
                    ) as client:
                        response = client.post("/chat/completions", headers=headers, json=payload)
                        response.raise_for_status()
                        data = response.json()
                    
                    model_response = self._parse_response(data)
                    self.logger.info(
                        "NVIDIA request succeeded: model=%s attempt=%s elapsed_seconds=%.2f",
                        model_name,
                        attempt,
                        time.monotonic() - started_at,
                    )
                    return model_response
                except httpx.HTTPStatusError as exc:
                    last_exception = exc
                    error_detail = ""
                    try:
                        error_detail = exc.response.text
                    except Exception:
                        pass
                    self.logger.error(
                        "Attempt %d failed with status %d for model %s. Detail: %s",
                        attempt, exc.response.status_code, model_name, error_detail
                    )
                    if attempt < self.max_attempts:
                        time.sleep(min(float(2 ** (attempt - 1)), self.max_backoff_seconds))
                    continue
                except (httpx.TimeoutException, httpx.HTTPError) as exc:
                    last_exception = exc
                    self.logger.warning("Attempt %d failed for model %s: %s", attempt, model_name, str(exc))
                    if attempt < self.max_attempts:
                        time.sleep(min(float(2 ** (attempt - 1)), self.max_backoff_seconds))
                    continue

            self.logger.warning("Model %s failed all attempts. Trying fallback if available.", model_name)

        if isinstance(last_exception, httpx.TimeoutException):
            raise NvidiaTimeoutError() from last_exception
        raise NvidiaBackendError() from last_exception

    def _parse_response(self, data: dict) -> ModelResponse:
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("Model response did not include choices.")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Model response did not include text content.")

        usage = data.get("usage", {})
        return ModelResponse(content=content, usage=usage, status="ok")

    def _resolve_temperature(self, temperature_override: float | str | None) -> float:
        value = temperature_override
        if value is None:
            value = self.settings.temperature
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Temperature override must be numeric.") from exc
        return min(max(numeric_value, 0.0), 0.2)

    def _resolve_max_tokens(self, max_tokens_override: int | None) -> int:
        if max_tokens_override is None:
            return self.settings.max_completion_tokens
        return max(128, min(max_tokens_override, self.settings.max_completion_tokens))
