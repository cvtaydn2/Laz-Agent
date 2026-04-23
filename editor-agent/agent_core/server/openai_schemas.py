from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class OpenAIMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = None


class OpenAIChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = "laz-agent"
    messages: list[OpenAIMessage] = Field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    extra_body: dict[str, Any] | None = Field(
        default=None,
        validation_alias=AliasChoices("extra_body", "extraBody"),
    )
    metadata: dict[str, Any] | None = None
    workspace: str | None = None
    mode: str | None = None
    changed_files: list[str] | None = None
    diff: str | None = None


class OpenAIModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "laz-agent"


class OpenAIModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[OpenAIModelObject]


class OpenAIResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class OpenAIChoice(BaseModel):
    index: int = 0
    message: OpenAIResponseMessage
    finish_reason: str = "stop"


class OpenAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChoice]
    usage: OpenAIUsage = Field(default_factory=OpenAIUsage)
