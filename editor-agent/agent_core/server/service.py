from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from agent_core.agent.orchestrator import AgentOrchestrator
from agent_core.config import Settings
from agent_core.logger import configure_logger
from agent_core.models import AgentMode, HealthStatus, SessionRecord


def resolve_workspace_or_400(workspace: str) -> Path:
    path = Path(workspace).expanduser().resolve()
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Workspace does not exist: {path}")
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Workspace must be a directory: {path}")
    return path


def build_orchestrator() -> AgentOrchestrator:
    settings = Settings.load()
    return AgentOrchestrator(settings=settings)


def build_server_logger():
    settings = Settings.load()
    return configure_logger(settings.logs_dir / "editor-agent.log")


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


def run_agent(
    mode: AgentMode,
    workspace: str,
    user_input: str | None,
    temperature_override: float | None = None,
    max_tokens_override: int | None = None,
) -> SessionRecord:
    logger = build_server_logger()
    orchestrator = build_orchestrator()
    workspace_path = resolve_workspace_or_400(workspace)
    try:
        return orchestrator.run(
            mode=mode,
            workspace_path=workspace_path,
            user_input=user_input,
            temperature_override=temperature_override,
            max_tokens_override=max_tokens_override,
        )
    except ValueError as exc:
        logger.exception("Validation error while running agent mode=%s workspace=%s", mode.value, workspace)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unhandled server error while running agent mode=%s workspace=%s", mode.value, workspace)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
