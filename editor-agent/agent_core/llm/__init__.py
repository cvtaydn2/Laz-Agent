from __future__ import annotations

from agent_core.config import Settings
from agent_core.llm.provider import LLMProvider
from agent_core.llm.nvidia import NvidiaProvider


def get_llm_provider(settings: Settings) -> LLMProvider:
    """
    Factory function to return the configured LLM provider.
    Currently defaults to NVIDIA, but can be extended to support other backends.
    """
    return NvidiaProvider(settings)
