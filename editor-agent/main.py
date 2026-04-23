from __future__ import annotations

import asyncio
from typing import Optional

import typer
from rich.console import Console

from agent_core.agent.orchestrator import AgentOrchestrator
from agent_core.config import Settings
from agent_core.models import AgentMode, HealthStatus
from agent_core.output.formatter import render_health, render_response
from agent_core.tools.file_tools import resolve_workspace as _resolve_workspace_util

app = typer.Typer(add_completion=False, help="CLI-first local coding agent.")
console = Console()


def _resolve_workspace(workspace: str) -> Path:
    try:
        return _resolve_workspace_util(workspace)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _build_orchestrator() -> AgentOrchestrator:
    settings = Settings.load()
    return AgentOrchestrator(settings=settings)


@app.command(help="Validate local configuration and endpoint readiness.")
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


@app.command(help="Analyze a project workspace and provide an overview of the architecture and potential issues.")
def analyze(
    workspace: str = typer.Argument(..., help="Path to the workspace directory to scan.")
) -> None:
    """Analyze a project workspace."""
    orchestrator = _build_orchestrator()
    response = asyncio.run(orchestrator.run(
        mode=AgentMode.ANALYZE,
        workspace_path=_resolve_workspace(workspace),
        user_input=None,
    ))
    render_response(console, response)


@app.command(help="Ask a natural language question about the project workspace using the LLM's context.")
def ask(
    workspace: str = typer.Argument(..., help="Path to the workspace directory."),
    question: str = typer.Argument(..., help="The natural language question to ask.")
) -> None:
    """Ask a question about a project workspace."""
    orchestrator = _build_orchestrator()
    response = asyncio.run(orchestrator.run(
        mode=AgentMode.ASK,
        workspace_path=_resolve_workspace(workspace),
        user_input=question,
    ))
    render_response(console, response)


@app.command(help="Get safe suggestions for improvements or refactors for a project workspace.")
def suggest(
    workspace: str = typer.Argument(..., help="Path to the workspace directory."),
    request: str = typer.Argument(..., help="The specific request for suggestions.")
) -> None:
    """Get safe suggestions for a project workspace."""
    orchestrator = _build_orchestrator()
    response = asyncio.run(orchestrator.run(
        mode=AgentMode.SUGGEST,
        workspace_path=_resolve_workspace(workspace),
        user_input=request,
    ))
    render_response(console, response)


@app.command("patch-preview", help="Generate a patch proposal preview (dry-run) without modifying any files.")
def patch_preview(
    workspace: str = typer.Argument(..., help="Path to the workspace directory."),
    request: str = typer.Argument(..., help="The code change request to preview.")
) -> None:
    """Generate a patch proposal preview without modifying files."""
    orchestrator = _build_orchestrator()
    response = asyncio.run(orchestrator.run(
        mode=AgentMode.PATCH_PREVIEW,
        workspace_path=_resolve_workspace(workspace),
        user_input=request,
    ))
    render_response(console, response)


@app.command(help="Generate and optionally apply a proposed patch with automated backups and rollback safety.")
def apply(
    workspace: str = typer.Argument(..., help="Path to the workspace directory."),
    request: str = typer.Argument(..., help="The specific code change request."),
    confirm: bool = typer.Option(False, "--confirm", help="Automatically apply changes without manual confirmation.")
) -> None:
    """Generate and optionally apply a proposed patch with backups and rollback."""
    orchestrator = _build_orchestrator()
    response = asyncio.run(orchestrator.run(
        mode=AgentMode.APPLY,
        workspace_path=_resolve_workspace(workspace),
        user_input=request,
        confirm=confirm,
    ))
    render_response(console, response)


@app.command(help="Revert the changes made in a specific session using its session ID.")
def rollback(
    session_id: str = typer.Argument(..., help="The session ID to rollback (e.g., apply-20240101-120000).")
) -> None:
    """Revert the changes made in a specific session."""
    orchestrator = _build_orchestrator()
    success = asyncio.run(orchestrator.rollback(session_id))
    if success:
        console.print(f"[bold green]Successfully rolled back session: {session_id}[/bold green]")
    else:
        console.print(f"[bold red]Failed to rollback session: {session_id}[/bold red]")
        raise typer.Exit(code=1)


@app.command(help="Compare primary and fallback models for the best architectural answer.")
def compare(
    workspace: str = typer.Argument(..., help="Path to the workspace directory."),
    request: str = typer.Argument(..., help="The request to compare model outputs for.")
) -> None:
    """Compare primary and secondary models."""
    orchestrator = _build_orchestrator()
    response = asyncio.run(orchestrator.run(
        mode=AgentMode.COMPARE,
        workspace_path=_resolve_workspace(workspace),
        user_input=request,
    ))
    render_response(console, response)


@app.command("bug-hunt", help="Deep dive into the workspace to find hidden bugs, race conditions, or performance leaks.")
def bug_hunt(
    workspace: str = typer.Argument(..., help="Path to the workspace directory."),
    target: Optional[str] = typer.Option(None, "--target", help="Specific area or issue to focus on.")
) -> None:
    """Deep hunt for bugs in the workspace."""
    orchestrator = _build_orchestrator()
    response = asyncio.run(orchestrator.run(
        mode=AgentMode.BUG_HUNT,
        workspace_path=_resolve_workspace(workspace),
        user_input=target,
    ))
    render_response(console, response)


@app.command(help="Focus strictly on fixing a specific bug or issue with high precision.")
def fix(
    workspace: str = typer.Argument(..., help="Path to the workspace directory."),
    issue: str = typer.Argument(..., help="The specific bug or issue to fix."),
    confirm: bool = typer.Option(False, "--confirm", help="Automatically apply the fix.")
) -> None:
    """Fix a specific bug with precision."""
    orchestrator = _build_orchestrator()
    response = asyncio.run(orchestrator.run(
        mode=AgentMode.FIX,
        workspace_path=_resolve_workspace(workspace),
        user_input=issue,
        confirm=confirm,
    ))
    render_response(console, response)


@app.command(help="Perform a formal code review of a file or the entire workspace.")
def review(
    workspace: str = typer.Argument(..., help="Path to the workspace directory."),
    target: Optional[str] = typer.Option(None, "--target", help="Specific file or directory to review.")
) -> None:
    """Perform a code review."""
    orchestrator = _build_orchestrator()
    response = asyncio.run(orchestrator.run(
        mode=AgentMode.REVIEW,
        workspace_path=_resolve_workspace(workspace),
        user_input=target,
    ))
    render_response(console, response)


def main(argv: Optional[list[str]] = None) -> int:
    app(args=argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
