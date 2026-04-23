from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from agent_core.agent.orchestrator import AgentOrchestrator
from agent_core.config import Settings
from agent_core.models import AgentMode, HealthStatus
from agent_core.output.formatter import render_health, render_response

app = typer.Typer(add_completion=False, help="CLI-first local coding agent.")
console = Console()


def _resolve_workspace(workspace: str) -> Path:
    path = Path(workspace).expanduser().resolve()
    if not path.exists():
        raise typer.BadParameter(f"Workspace does not exist: {path}")
    if not path.is_dir():
        raise typer.BadParameter(f"Workspace must be a directory: {path}")
    return path


def _build_orchestrator() -> AgentOrchestrator:
    settings = Settings.load()
    return AgentOrchestrator(settings=settings)


@app.command()
def health() -> None:
    """Validate local configuration and endpoint readiness."""
    settings = Settings.load()
    status = HealthStatus(
        ok=bool(settings.nvidia_api_key),
        base_url=str(settings.nvidia_base_url),
        model=settings.nvidia_model,
        api_key_configured=bool(settings.nvidia_api_key),
        session_dir=str(settings.session_dir),
        timeout_seconds=settings.timeout_seconds,
    )
    render_health(console, status)
    if not status.ok:
        raise typer.Exit(code=1)


@app.command()
def analyze(workspace: str) -> None:
    """Analyze a project workspace."""
    orchestrator = _build_orchestrator()
    response = orchestrator.run(
        mode=AgentMode.ANALYZE,
        workspace_path=_resolve_workspace(workspace),
        user_input=None,
    )
    render_response(console, response)


@app.command()
def ask(workspace: str, question: str) -> None:
    """Ask a question about a project workspace."""
    orchestrator = _build_orchestrator()
    response = orchestrator.run(
        mode=AgentMode.ASK,
        workspace_path=_resolve_workspace(workspace),
        user_input=question,
    )
    render_response(console, response)


@app.command()
def suggest(workspace: str, request: str) -> None:
    """Get safe suggestions for a project workspace."""
    orchestrator = _build_orchestrator()
    response = orchestrator.run(
        mode=AgentMode.SUGGEST,
        workspace_path=_resolve_workspace(workspace),
        user_input=request,
    )
    render_response(console, response)


@app.command("patch-preview")
def patch_preview(workspace: str, request: str) -> None:
    """Generate a patch proposal preview without modifying files."""
    orchestrator = _build_orchestrator()
    response = orchestrator.run(
        mode=AgentMode.PATCH_PREVIEW,
        workspace_path=_resolve_workspace(workspace),
        user_input=request,
    )
    render_response(console, response)


def main(argv: Optional[list[str]] = None) -> int:
    app(args=argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
