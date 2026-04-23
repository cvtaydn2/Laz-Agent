from __future__ import annotations

import json
import time
import asyncio
from typing import Any, AsyncIterable, Iterable

import httpx

from agent_core.config import Settings
from agent_core.logger import configure_logger
from agent_core.models import ChatMessage, ModelResponse
from agent_core.llm.provider import LLMProvider
from agent_core.server.metrics import LLM_LATENCY_SECONDS


class NvidiaInferenceError(RuntimeError):
    def __init__(self, user_message: str, *, error_type: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.error_type = error_type


class NvidiaTimeoutError(NvidiaInferenceError):
    def __init__(self) -> None:
        super().__init__(
            "The model backend timed out while generating a response. "
            "Please retry with a smaller request or a narrower workspace context.",
            error_type="timeout",
        )


class NvidiaBackendError(NvidiaInferenceError):
    def __init__(self) -> None:
        super().__init__(
            "The model backend is temporarily unavailable. "
            "Please retry in a moment or reduce the request scope.",
            error_type="backend_unavailable",
        )


class NvidiaProvider(LLMProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = configure_logger(settings.logs_dir / "editor-agent.log")
        self.max_attempts = 3
        self.max_backoff_seconds = 3.0

    # ------------------------------------------------------------------
    # Non-streaming chat
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: Iterable[ChatMessage],
        temperature_override: float | str | None = None,
        max_tokens_override: int | None = None,
        model_override: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ModelResponse:
        if not self.settings.nvidia_api_key:
            raise ValueError("NVIDIA_API_KEY is not configured.")

        if model_override:
            models_to_try = [model_override]
        else:
            models_to_try = [self.settings.nvidia_model]
            if (
                self.settings.nvidia_fallback_model
                and self.settings.nvidia_fallback_model != self.settings.nvidia_model
            ):
                models_to_try.append(self.settings.nvidia_fallback_model)

        last_exception: Exception | None = None
        started_at = time.monotonic()

        for model_name in models_to_try:
            payload: dict[str, Any] = {
                "model": model_name,
                "temperature": self._resolve_temperature(temperature_override),
                "max_tokens": self._resolve_max_tokens(max_tokens_override),
                "messages": [message.model_dump() for message in messages],
                "stream": False,
            }
            if tools:
                payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

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
                "NVIDIA chat starting: model=%s timeout_seconds=%.1f",
                model_name,
                self.settings.timeout_seconds,
            )

            for attempt in range(1, self.max_attempts + 1):
                try:
                    async with httpx.AsyncClient(
                        base_url=str(self.settings.nvidia_base_url),
                        timeout=timeout,
                    ) as client:
                        response = await client.post(
                            "/chat/completions", headers=headers, json=payload
                        )
                        response.raise_for_status()
                        data = response.json()

                    model_response = self._parse_response(data)
                    elapsed = time.monotonic() - started_at
                    LLM_LATENCY_SECONDS.labels(
                        provider="nvidia", model=model_name
                    ).observe(elapsed)
                    self.logger.info(
                        "NVIDIA chat succeeded: model=%s attempt=%s elapsed_seconds=%.2f",
                        model_name,
                        attempt,
                        elapsed,
                    )
                    return model_response
                except httpx.HTTPStatusError as exc:
                    last_exception = exc
                    self.logger.error(
                        "NVIDIA status error %d: %s",
                        exc.response.status_code,
                        exc.response.text,
                    )
                    if attempt < self.max_attempts:
                        await asyncio.sleep(
                            min(float(2 ** (attempt - 1)), self.max_backoff_seconds)
                        )
                    continue
                except (httpx.TimeoutException, httpx.HTTPError) as exc:
                    last_exception = exc
                    if attempt < self.max_attempts:
                        await asyncio.sleep(
                            min(float(2 ** (attempt - 1)), self.max_backoff_seconds)
                        )
                    continue

        if isinstance(last_exception, httpx.TimeoutException):
            raise NvidiaTimeoutError() from last_exception
        raise NvidiaBackendError() from last_exception

    # ------------------------------------------------------------------
    # Streaming chat — yields structured dicts, not raw strings
    # ------------------------------------------------------------------

    async def chat_stream(
        self,
        messages: Iterable[ChatMessage],
        temperature_override: float | str | None = None,
        max_tokens_override: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> AsyncIterable[dict[str, Any]]:
        """Yield structured chunk dicts:

        {
            "content": str | None,
            "tool_calls": list[dict],   # delta tool_calls from the provider
            "finish_reason": str | None,
            "raw": dict,                # full raw chunk for debugging
        }
        """
        if not self.settings.nvidia_api_key:
            raise ValueError("NVIDIA_API_KEY is not configured.")

        model_name = self.settings.nvidia_model
        payload: dict[str, Any] = {
            "model": model_name,
            "temperature": self._resolve_temperature(temperature_override),
            "max_tokens": self._resolve_max_tokens(max_tokens_override),
            "messages": [message.model_dump() for message in messages],
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        headers = {
            "Authorization": f"Bearer {self.settings.nvidia_api_key}",
            "Content-Type": "application/json",
        }

        self.logger.info("NVIDIA stream starting: model=%s", model_name)
        started_at = time.monotonic()

        async with httpx.AsyncClient(
            base_url=str(self.settings.nvidia_base_url),
            timeout=self.settings.timeout_seconds,
        ) as client:
            async with client.stream(
                "POST", "/chat/completions", headers=headers, json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        choice = choices[0]
                        delta = choice.get("delta", {})
                        finish_reason = choice.get("finish_reason")
                        yield {
                            "content": delta.get("content"),
                            "tool_calls": delta.get("tool_calls") or [],
                            "finish_reason": finish_reason,
                            "raw": chunk,
                        }
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

                LLM_LATENCY_SECONDS.labels(
                    provider="nvidia", model=model_name
                ).observe(time.monotonic() - started_at)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_response(self, data: dict[str, Any]) -> ModelResponse:
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("Model response did not include choices.")
        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content") or ""
        usage = data.get("usage", {})
        tool_calls = message.get("tool_calls") or []
        return ModelResponse(
            content=content,
            usage=usage,
            status="ok",
            finish_reason=choice.get("finish_reason"),
            tool_calls=tool_calls,
            raw_response=data,
        )

    def _resolve_temperature(self, temperature_override: float | str | None) -> float:
        try:
            val = (
                float(temperature_override)
                if temperature_override is not None
                else self.settings.temperature
            )
            # Honour the caller's intent; only clamp to the valid API range
            return min(max(val, 0.0), 2.0)
        except (TypeError, ValueError):
            return self.settings.temperature

    def _resolve_max_tokens(self, max_tokens_override: int | None) -> int:
        val = (
            max_tokens_override
            if max_tokens_override is not None
            else self.settings.max_completion_tokens
        )
        return max(128, min(val, 4096))
