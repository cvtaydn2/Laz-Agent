from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from agent_core.agent.orchestrator import AgentOrchestrator
from agent_core.config import Settings
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


def run_agent(mode: AgentMode, workspace: str, user_input: str | None) -> SessionRecord:
    orchestrator = build_orchestrator()
    workspace_path = resolve_workspace_or_400(workspace)
    try:
        return orchestrator.run(
            mode=mode,
            workspace_path=workspace_path,
            user_input=user_input,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
