from __future__ import annotations

from pathlib import Path
from typing import AsyncIterable

from fastapi import HTTPException

from agent_core.agent.orchestrator import AgentOrchestrator
from agent_core.config import Settings
from agent_core.models import AgentMode, HealthStatus, ParsedAnswer, SessionRecord, WorkspaceSummary, build_session_id, utc_now
from agent_core.llm.nvidia import NvidiaBackendError, NvidiaTimeoutError
from agent_core.tools.file_tools import resolve_workspace

# Module-level singleton — created once, reused across all requests
_orchestrator: AgentOrchestrator | None = None


def get_orchestrator() -> AgentOrchestrator:
    """Return the shared orchestrator instance, creating it on first call."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator(settings=Settings.load())
    return _orchestrator


def build_orchestrator() -> AgentOrchestrator:
    """Alias kept for backward compatibility."""
    return get_orchestrator()


def resolve_workspace_or_400(workspace: str) -> Path:
    try:
        return resolve_workspace(workspace)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def build_safe_fallback_session(
    *,
    mode: AgentMode,
    workspace_path: Path,
    user_input: str | None,
    message: str,
    note: str,
) -> SessionRecord:
    return SessionRecord(
        session_id=build_session_id(mode),
        created_at=utc_now(),
        mode=mode,
        workspace_path=str(workspace_path),
        prompt=user_input or "",
        user_input=user_input,
        workspace_summary=WorkspaceSummary(
            root_path=str(workspace_path),
            total_files_scanned=0,
            included_files=0,
            skipped_files=0,
            top_extensions={},
            sampled_files=[],
            notes=[note],
        ),
        ranked_files=[],
        selected_context=[],
        raw_response=message,
        parsed_response=ParsedAnswer(
            summary=message,
            raw_text=message,
            parse_strategy="raw_text",
            risks_text=note if mode == AgentMode.REVIEW else "",
            next_steps_text=(
                "Retry with a smaller diff, fewer changed files, or a narrower workspace context."
                if mode == AgentMode.REVIEW
                else ""
            ),
        ),
    )


def build_health_status() -> HealthStatus:
    settings = Settings.load()
    return HealthStatus(
        ok=bool(settings.nvidia_api_key),
        base_url=str(settings.nvidia_base_url),
        model=settings.nvidia_model,
        api_key_configured=bool(settings.nvidia_api_key),
        session_dir=str(settings.session_dir),
        timeout_seconds=settings.timeout_seconds,
    )


async def run_agent(
    mode: AgentMode,
    workspace: str,
    user_input: str | None,
    *,
    temperature_override: float | None = None,
    max_tokens_override: int | None = None,
    changed_files: list[str] | None = None,
    diff_text: str | None = None,
    preferred_files: list[str] | None = None,
) -> SessionRecord:
    orchestrator = get_orchestrator()
    logger = orchestrator.logger
    workspace_path = resolve_workspace_or_400(workspace)
    try:
        logger.info("Starting orchestrator.run for mode=%s", mode.value)
        return await orchestrator.run(
            mode=mode,
            workspace_path=workspace_path,
            user_input=user_input,
            temperature_override=temperature_override,
            max_tokens_override=max_tokens_override,
            changed_files=changed_files,
            diff_text=diff_text,
            preferred_files=preferred_files,
        )
    except NvidiaTimeoutError as exc:
        logger.warning(
            "Backend timeout while running agent mode=%s workspace=%s",
            mode.value,
            workspace,
        )
        return build_safe_fallback_session(
            mode=mode,
            workspace_path=workspace_path,
            user_input=user_input,
            message=exc.user_message,
            note="Backend inference timed out.",
        )
    except NvidiaBackendError as exc:
        logger.warning(
            "Backend inference error while running agent mode=%s workspace=%s",
            mode.value,
            workspace,
        )
        return build_safe_fallback_session(
            mode=mode,
            workspace_path=workspace_path,
            user_input=user_input,
            message=exc.user_message,
            note="Backend inference failed.",
        )
    except ValueError as exc:
        logger.exception("Validation error while running agent mode=%s workspace=%s", mode.value, workspace)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unhandled server error while running agent mode=%s workspace=%s", mode.value, workspace)
        return build_safe_fallback_session(
            mode=mode,
            workspace_path=workspace_path,
            user_input=user_input,
            message="The local agent hit an internal processing error before it could return a stable result. Please retry with a smaller request or a narrower workspace context.",
            note="Internal error fallback response generated.",
        )
async def stream_agent(
    mode: AgentMode,
    workspace: str,
    user_input: str | None,
    *,
    temperature_override: float | None = None,
    max_tokens_override: int | None = None,
    changed_files: list[str] | None = None,
    diff_text: str | None = None,
    preferred_files: list[str] | None = None,
) -> AsyncIterable[str]:
    orchestrator = build_orchestrator()
    workspace_path = resolve_workspace_or_400(workspace)
    async for chunk in orchestrator.stream_run(
        mode=mode,
        workspace_path=workspace_path,
        user_input=user_input,
        temperature_override=temperature_override,
        max_tokens_override=max_tokens_override,
        changed_files=changed_files,
        diff_text=diff_text,
        preferred_files=preferred_files,
    ):
        yield chunk
