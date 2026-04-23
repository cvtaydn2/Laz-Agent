from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterable, Iterable

from agent_core.models import ChatMessage, ModelResponse


class LLMProvider(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: Iterable[ChatMessage],
        temperature_override: float | str | None = None,
        max_tokens_override: int | None = None,
        model_override: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ModelResponse:
        """Execute a non-streaming chat completion."""
        ...

    @abstractmethod
    async def chat_stream(
        self,
        messages: Iterable[ChatMessage],
        temperature_override: float | str | None = None,
        max_tokens_override: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> AsyncIterable[dict[str, Any]]:
        """Execute a streaming chat completion.

        Yields structured dicts:
            {
                "content": str | None,
                "tool_calls": list[dict],
                "finish_reason": str | None,
                "raw": dict,
            }
        """
        ...
