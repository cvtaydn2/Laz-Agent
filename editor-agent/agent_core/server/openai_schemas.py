from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Message content parts (multimodal)
# ---------------------------------------------------------------------------

class OpenAIContentPartText(BaseModel):
    type: Literal["text"] = "text"
    text: str


class OpenAIContentPartImageUrl(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: dict[str, Any]


OpenAIContentPart = Union[OpenAIContentPartText, OpenAIContentPartImageUrl]


# ---------------------------------------------------------------------------
# Tool / function calling schemas
# ---------------------------------------------------------------------------

class OpenAIFunctionSpec(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class OpenAITool(BaseModel):
    type: Literal["function"] = "function"
    function: OpenAIFunctionSpec


class OpenAIToolChoiceFunction(BaseModel):
    type: Literal["function"] = "function"
    function: dict[str, str]


# tool_choice can be "none" | "auto" | "required" or a structured object
OpenAIToolChoice = Union[Literal["none", "auto", "required"], OpenAIToolChoiceFunction]


class OpenAIResponseFormat(BaseModel):
    type: str  # e.g. "text" | "json_object" | "json_schema"
    json_schema: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Tool call (in assistant messages)
# ---------------------------------------------------------------------------

class OpenAIToolCallFunction(BaseModel):
    name: str
    arguments: str


class OpenAIToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: OpenAIToolCallFunction


# ---------------------------------------------------------------------------
# Message schemas
# ---------------------------------------------------------------------------

class OpenAIMessage(BaseModel):
    role: str
    # content can be a plain string, a list of content parts, or None
    # (assistant messages that only contain tool_calls have content=None)
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

    # Tool / function calling — accepted and passed through to NVIDIA NIM
    tools: list[OpenAITool] | None = None
    tool_choice: OpenAIToolChoice | None = None

    # Response format (e.g. {"type": "json_object"})
    response_format: OpenAIResponseFormat | None = None

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
    content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None


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
