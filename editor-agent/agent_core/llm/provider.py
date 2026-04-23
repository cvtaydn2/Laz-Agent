from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, AsyncIterable

from agent_core.models import ChatMessage, ModelResponse


class LLMProvider(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: Iterable[ChatMessage],
        temperature_override: float | str | None = None,
        max_tokens_override: int | None = None,
    ) -> ModelResponse:
        """Execute a non-streaming chat completion."""
        pass

    @abstractmethod
    async def chat_stream(
        self,
        messages: Iterable[ChatMessage],
        temperature_override: float | str | None = None,
        max_tokens_override: int | None = None,
    ) -> AsyncIterable[str]:
        """Execute a streaming chat completion, yielding text chunks."""
        pass
