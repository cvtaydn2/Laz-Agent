from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agent_core.models import AgentMode, ChatMessage, SessionRecord
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

# Extensions recognised as code/config files for heuristic path detection
_ALLOWED_EXTS = {
    "py", "ts", "tsx", "js", "jsx", "json", "md", "yml", "yaml",
    "toml", "ini", "cfg", "sh", "go", "rs", "java", "kt", "cs",
    "cpp", "c", "h", "hpp", "sql", "html", "css", "txt",
}


# ---------------------------------------------------------------------------
# Content helpers
# ---------------------------------------------------------------------------

def _content_to_text(content: str | list[dict[str, Any]] | None) -> str:
    """Flatten any OpenAI content shape to a plain string."""
    if content is None:
        return ""
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


# ---------------------------------------------------------------------------
# Message normalisation
# ---------------------------------------------------------------------------

def normalize_openai_messages(messages: list[OpenAIMessage]) -> list[ChatMessage]:
    """Convert OpenAI message list to internal ChatMessage list, preserving roles.

    - Skips messages with no usable text content.
    - Maps "developer" → "system" (OpenAI o-series convention).
    - Handles tool messages with a synthetic text fallback.
    """
    normalized: list[ChatMessage] = []
    for message in messages:
        role = (message.role or "").strip().lower()
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            continue

        text = _content_to_text(message.content)

        if role == "developer":
            role = "system"

        if role == "tool":
            # Tool result messages: use text if present, else synthetic placeholder
            text = text or f"[tool result id={message.tool_call_id or 'unknown'}]"

        if role == "assistant" and not text and message.tool_calls:
            # Assistant message that only contains tool_calls — skip for now;
            # the local agent does not execute tool calls.
            continue

        if text:
            normalized.append(ChatMessage(role=role, content=text))

    return normalized


def extract_last_user_text(messages: list[OpenAIMessage]) -> str | None:
    """Return the text of the last user-role message, or the last non-empty message."""
    for message in reversed(messages):
        if (message.role or "").strip().lower() == "user":
            text = _content_to_text(message.content)
            if text:
                return text
    # Fallback: last non-empty message of any role
    for message in reversed(messages):
        text = _content_to_text(message.content)
        if text:
            return text
    return None


# Keep the old name as an alias so existing call-sites don't break
def extract_user_message(messages: list[OpenAIMessage]) -> str | None:
    return extract_last_user_text(messages)


# ---------------------------------------------------------------------------
# Request field extractors
# ---------------------------------------------------------------------------

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


def extract_request_workspace(request: OpenAIChatCompletionRequest) -> str | None:
    for container in (request.extra_body, request.metadata):
        if isinstance(container, dict):
            for key in (
                "workspace", "working_directory", "workingDirectory",
                "cwd", "root", "rootPath", "projectPath", "baseDir",
            ):
                val = container.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()

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

    # 1. Explicit metadata fields
    for container in (request.extra_body, request.metadata):
        if isinstance(container, dict):
            val = container.get("preferred_files") or container.get("active_files")
            if isinstance(val, list):
                for f in val:
                    preferred.add(str(f).replace("\\", "/").strip())

    # 2. Heuristic: scan last 5 messages for path-like strings.
    #    A valid path must contain at least one "/" separator AND end with a
    #    recognised extension.  Plain dotted words (e.g. "e.g.", "i.e.") are
    #    excluded because they lack a directory component.
    import re
    path_pattern = re.compile(r'([a-zA-Z0-9_\-]+(?:/[a-zA-Z0-9_\-\.]+)+)')

    for msg in request.messages[-5:]:
        text = _content_to_text(msg.content)
        for match in path_pattern.findall(text):
            normalized = match.replace("\\", "/").strip()
            ext = normalized.rsplit(".", 1)[-1].lower() if "." in normalized else ""
            if "/" in normalized and ext in _ALLOWED_EXTS and " " not in normalized:
                preferred.add(normalized)

    return sorted(list(preferred))


# ---------------------------------------------------------------------------
# Tool payload helpers
# ---------------------------------------------------------------------------

def extract_tools_payload(request: OpenAIChatCompletionRequest) -> list[dict[str, Any]] | None:
    """Serialise request.tools to plain dicts for the NVIDIA provider."""
    if not request.tools:
        return None
    return [tool.model_dump(exclude_none=True) for tool in request.tools]


def extract_tool_choice_payload(request: OpenAIChatCompletionRequest) -> str | dict[str, Any] | None:
    """Serialise request.tool_choice to a provider-compatible value."""
    tc = request.tool_choice
    if tc is None:
        return None
    if isinstance(tc, str):
        return tc
    if hasattr(tc, "model_dump"):
        return tc.model_dump(exclude_none=True)
    return tc


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------

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
        "next_steps": (
            parsed.next_steps_text
            or "\n".join(parsed.next_steps).strip()
            or "Review the findings and apply the highest-confidence fixes first."
        ),
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


# ---------------------------------------------------------------------------
# SSE chunk formatter
# ---------------------------------------------------------------------------

def format_openai_stream_chunk(
    *,
    model: str,
    completion_id: str,
    created: int,
    delta_content: str | None = None,
    delta_tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
    # Legacy positional-keyword alias kept for call-sites that still use `content=`
    content: str | None = None,
) -> str:
    # Support old callers that pass `content=` instead of `delta_content=`
    if delta_content is None and content is not None:
        delta_content = content

    delta: dict[str, Any] = {}
    if delta_content is not None:
        delta["content"] = delta_content
    if delta_tool_calls:
        delta["tool_calls"] = delta_tool_calls

    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model or OPENAI_COMPATIBLE_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _append_section(target: list[str], title: str, items: list[str]) -> None:
    normalized_items = [item.strip() for item in items if item and item.strip()]
    if not normalized_items:
        return
    body = "\n".join(f"- {item}" for item in normalized_items)
    target.append(f"{title}:\n{body}")
