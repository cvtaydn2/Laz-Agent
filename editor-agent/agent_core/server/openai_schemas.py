from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Message content parts (multimodal)
# ---------------------------------------------------------------------------

class OpenAIContentPartText(BaseModel):
    type: Literal["text"] = "text"
    text: str


class OpenAIContentPartImage(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: dict[str, Any]


# ---------------------------------------------------------------------------
# Tool / function calling schemas
# ---------------------------------------------------------------------------

class OpenAIFunctionDefinition(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class OpenAITool(BaseModel):
    type: Literal["function"] = "function"
    function: OpenAIFunctionDefinition


class OpenAIToolChoice(BaseModel):
    """Structured tool_choice object (e.g. {"type": "function", "function": {"name": "..."}})."""
    type: str
    function: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Message schemas
# ---------------------------------------------------------------------------

class OpenAIToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: dict[str, Any]


class OpenAIMessage(BaseModel):
    role: str
    # content can be a plain string, a list of content parts, or None (for
    # assistant messages that only contain tool_calls)
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None
    tool_call_id: str | None = None


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class OpenAIChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = "laz-agent"
    messages: list[OpenAIMessage] = Field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False

    # Tool / function calling (accepted but not forwarded to the local agent)
    tools: list[OpenAITool] | None = None
    tool_choice: str | OpenAIToolChoice | None = None

    # Response format (e.g. {"type": "json_object"})
    response_format: dict[str, Any] | None = None

    # Laz-agent extensions
    extra_body: dict[str, Any] | None = Field(
        default=None,
        validation_alias=AliasChoices("extra_body", "extraBody"),
    )
    metadata: dict[str, Any] | None = None
    workspace: str | None = None
    mode: str | None = None
    changed_files: list[str] | None = None
    diff: str | None = None


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

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
    # tool_calls omitted — local agent does not emit native tool calls yet


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
