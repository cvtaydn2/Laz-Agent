from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agent_core.models import AgentMode, SessionRecord
from agent_core.server.openai_schemas import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIChoice,
    OpenAIMessage,
    OpenAIModelObject,
    OpenAIModelsResponse,
    OpenAIResponseMessage,
    OpenAIUsage,
)

OPENAI_COMPATIBLE_MODEL_ID = "laz-agent"
OPENAI_ALLOWED_MODES = {
    AgentMode.ASK.value,
    AgentMode.ANALYZE.value,
    AgentMode.SUGGEST.value,
    AgentMode.PATCH_PREVIEW.value,
}


def build_openai_models_response() -> OpenAIModelsResponse:
    created = int(datetime.now(timezone.utc).timestamp())
    return OpenAIModelsResponse(
        data=[
            OpenAIModelObject(
                id=OPENAI_COMPATIBLE_MODEL_ID,
                created=created,
            )
        ]
    )


def extract_user_message(messages: list[OpenAIMessage]) -> str | None:
    for message in reversed(messages):
        if message.role != "user":
            continue
        text = _content_to_text(message.content)
        if text:
            return text
    return None


def extract_request_workspace(request: OpenAIChatCompletionRequest) -> str | None:
    for container in (request.extra_body, request.metadata):
        if isinstance(container, dict):
            workspace = container.get("workspace")
            if isinstance(workspace, str) and workspace.strip():
                return workspace.strip()

    if request.workspace and request.workspace.strip():
        return request.workspace.strip()
    return None


def extract_request_mode(request: OpenAIChatCompletionRequest) -> str:
    for container in (request.extra_body, request.metadata):
        if isinstance(container, dict):
            mode = container.get("mode")
            if isinstance(mode, str) and mode.strip():
                return mode.strip().lower()

    if request.mode and request.mode.strip():
        return request.mode.strip().lower()
    return AgentMode.ASK.value


def validate_openai_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in OPENAI_ALLOWED_MODES:
        raise ValueError(
            "Unsupported mode for OpenAI-compatible endpoint. "
            "Allowed values: ask, analyze, suggest, patch-preview."
        )
    return normalized.replace("-", "_")


def build_openai_response(
    session: SessionRecord,
    requested_model: str,
) -> OpenAIChatCompletionResponse:
    content = session_to_plain_text(session)
    created = int(session.created_at.timestamp())
    return OpenAIChatCompletionResponse(
        id=f"chatcmpl-{session.session_id}",
        created=created,
        model=requested_model or OPENAI_COMPATIBLE_MODEL_ID,
        choices=[
            OpenAIChoice(
                message=OpenAIResponseMessage(content=content),
                finish_reason="stop",
            )
        ],
        usage=OpenAIUsage(),
    )


def build_openai_error(
    message: str,
    error_type: str = "invalid_request_error",
    code: str | None = None,
) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": code,
        }
    }


def session_to_plain_text(session: SessionRecord) -> str:
    parsed = session.parsed_response
    sections: list[str] = []

    if parsed.summary.strip():
        sections.append(parsed.summary.strip())

    _append_section(sections, "Findings", parsed.findings)
    _append_section(sections, "Suggestions", parsed.suggestions)
    _append_section(sections, "Commands To Consider", parsed.commands_to_consider)
    _append_section(sections, "Risks", parsed.risks)
    _append_section(sections, "Affected Files", parsed.affected_files)
    _append_section(sections, "Proposed Changes", parsed.proposed_changes)
    _append_section(sections, "Next Steps", parsed.next_steps)

    if sections:
        return "\n\n".join(sections)

    if parsed.raw_text.strip():
        return parsed.raw_text.strip()

    if session.raw_response.strip():
        return session.raw_response.strip()

    return "The agent completed the request but did not return a formatted response."


def _append_section(target: list[str], title: str, items: list[str]) -> None:
    normalized_items = [item.strip() for item in items if item and item.strip()]
    if not normalized_items:
        return
    body = "\n".join(f"- {item}" for item in normalized_items)
    target.append(f"{title}:\n{body}")


def _content_to_text(content: str | list[dict[str, Any]] | None) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                text = item["text"].strip()
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()
    return ""
