from __future__ import annotations

import os
from agent_core.config import Settings
from agent_core.llm.provider import LLMProvider
from agent_core.llm.nvidia import NvidiaProvider

# Future implementations
# from agent_core.llm.openai import OpenAIProvider
# from agent_core.llm.ollama import OllamaProvider

def get_llm_provider(settings: Settings) -> LLMProvider:
    """
    Factory to return the configured LLM provider.
    Defaults to NVIDIA, but ready for OpenAI or Ollama based on LLM_PROVIDER env var.
    """
    provider_type = os.getenv("LLM_PROVIDER", "nvidia").lower()
    
    if provider_type == "nvidia":
        return NvidiaProvider(settings)
    
    # Placeholder for future expansion
    # if provider_type == "openai":
    #     return OpenAIProvider(settings)
    # if provider_type == "ollama":
    #     return OllamaProvider(settings)
        
    # Fallback to NVIDIA
    return NvidiaProvider(settings)
