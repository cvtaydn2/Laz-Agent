from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agent_core.models import AgentMode, SessionRecord
from agent_core.config import Settings
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
import json

OPENAI_COMPATIBLE_MODEL_ID = "laz-agent"
OPENAI_ALLOWED_MODES = {
    AgentMode.ASK.value,
    AgentMode.ANALYZE.value,
    AgentMode.SUGGEST.value,
    AgentMode.PATCH_PREVIEW.value,
    AgentMode.REVIEW.value,
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
    # Combine last few messages to capture full context if selection is separated
    parts: list[str] = []
    for message in messages[-3:]: # Look at last 3 messages for context
        text = _content_to_text(message.content)
        if text:
            parts.append(text)
    return "\n\n".join(parts) if parts else None


def extract_request_workspace(request: OpenAIChatCompletionRequest) -> str | None:
    for container in (request.extra_body, request.metadata):
        if isinstance(container, dict):
            workspace = container.get("workspace")
            if isinstance(workspace, str) and workspace.strip():
                return workspace.strip()

    if request.workspace and request.workspace.strip():
        return request.workspace.strip()
    settings = Settings.load()
    if settings.default_workspace.strip():
        return settings.default_workspace.strip()
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
            "Allowed values: ask, analyze, suggest, patch-preview, review."
        )
    return normalized.replace("-", "_")


def extract_changed_files(request: OpenAIChatCompletionRequest) -> list[str]:
    for container in (request.extra_body, request.metadata):
        if isinstance(container, dict):
            value = container.get("changed_files")
            if isinstance(value, list):
                return [str(item).replace("\\", "/").strip() for item in value if str(item).strip()]

    if isinstance(request.changed_files, list):
        return [str(item).replace("\\", "/").strip() for item in request.changed_files if str(item).strip()]
    return []


def extract_diff(request: OpenAIChatCompletionRequest) -> str | None:
    for container in (request.extra_body, request.metadata):
        if isinstance(container, dict):
            value = container.get("diff")
            if isinstance(value, str) and value.strip():
                return value.strip()

    if isinstance(request.diff, str) and request.diff.strip():
        return request.diff.strip()
    return None


def extract_preferred_files(request: OpenAIChatCompletionRequest) -> list[str]:
    preferred: set[str] = set()
    
    # 1. Check explicit metadata
    for container in (request.extra_body, request.metadata):
        if isinstance(container, dict):
            val = container.get("preferred_files") or container.get("active_files")
            if isinstance(val, list):
                for f in val:
                    preferred.add(str(f).replace("\\", "/").strip())

    # 2. Heuristic: Scan message history for potential file paths
    import re
    # Match strings that look like paths: word/word.ext or word.ext
    path_pattern = re.compile(r'([a-zA-Z0-9_\-\./]+\.[a-zA-Z0-9]{1,5})')
    
    for msg in request.messages[-5:]: # Scan last 5 messages
        text = _content_to_text(msg.content)
        for match in path_pattern.findall(text):
            # Basic validation to avoid matching plain words
            if "." in match and "/" in match or len(match.split(".")[-1]) in {2, 3, 4}:
                preferred.add(match.replace("\\", "/").strip())

    return sorted(list(preferred))


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
    if session.mode == AgentMode.REVIEW:
        return json_review_text(session)
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


def json_review_text(session: SessionRecord) -> str:
    parsed = session.parsed_response
    findings = [
        {
            "title": finding.title,
            "severity": finding.severity,
            "file": finding.file,
            "evidence": finding.evidence,
            "issue": finding.issue,
            "suggested_fix": finding.suggested_fix,
        }
        for finding in parsed.review_findings
    ]
    payload = {
        "summary": parsed.summary or "Review completed.",
        "findings": findings,
        "risks": parsed.risks_text or "\n".join(parsed.risks).strip() or "No major risks identified.",
        "next_steps": parsed.next_steps_text or "\n".join(parsed.next_steps).strip() or "Review the findings and apply the highest-confidence fixes first.",
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def build_openai_fallback_response(model: str, content: str) -> OpenAIChatCompletionResponse:
    created = int(datetime.now(timezone.utc).timestamp())
    return OpenAIChatCompletionResponse(
        id=f"chatcmpl-fallback-{created}",
        created=created,
        model=model or OPENAI_COMPATIBLE_MODEL_ID,
        choices=[
            OpenAIChoice(
                message=OpenAIResponseMessage(content=content),
                finish_reason="stop",
            )
        ],
        usage=OpenAIUsage(),
    )


def build_openai_streaming_chunks(
    *,
    model: str,
    content: str,
    completion_id: str,
    created: int,
) -> list[str]:
    chunks = [
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model or OPENAI_COMPATIBLE_MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model or OPENAI_COMPATIBLE_MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        },
    ]
    return [f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n" for chunk in chunks] + ["data: [DONE]\n\n"]


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
