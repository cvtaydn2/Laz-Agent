from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent_core.models import HealthStatus, SessionRecord


def render_health(console: Console, status: HealthStatus) -> None:
    table = Table(title="Editor Agent Health")
    table.add_column("Check")
    table.add_column("Value")
    table.add_row("Base URL", status.base_url)
    table.add_row("Model", status.model)
    table.add_row("API key configured", "yes" if status.api_key_configured else "no")
    table.add_row("Session dir", status.session_dir)
    table.add_row("Timeout seconds", str(status.timeout_seconds))
    console.print(table)
    if status.ok:
        console.print(Panel("Health check passed.", title="Status"))
    else:
        console.print(Panel("Health check failed: NVIDIA_API_KEY is missing.", title="Status"))


def render_response(console: Console, session: SessionRecord) -> None:
    parsed = session.parsed_response

    if parsed.thought:
        _safe_panel(console, parsed.thought, title="Reasoning & Planning", style="dim cyan")

    _safe_panel(console, parsed.summary, title=f"{session.mode.value.title()} Summary")

    table = Table(title="Workspace Snapshot")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Workspace", session.workspace_path)
    table.add_row("Included files", str(session.workspace_summary.included_files))
    table.add_row("Skipped files", str(session.workspace_summary.skipped_files))
    table.add_row("Session", session.session_id)
    console.print(table)

    ranked_table = Table(title="Top Ranked Files")
    ranked_table.add_column("File")
    ranked_table.add_column("Score")
    ranked_table.add_column("Reason")
    for item in session.ranked_files:
        ranked_table.add_row(item.relative_path, f"{item.score:.1f}", item.reason)
    console.print(ranked_table)

    _render_list(console, "Findings", parsed.findings)
    _render_list(console, "Suggestions", parsed.suggestions)
    _render_list(console, "Commands To Consider", parsed.commands_to_consider)
    _render_list(console, "Risks", parsed.risks)
    _render_list(console, "Affected Files", parsed.affected_files)
    _render_list(console, "Proposed Changes", parsed.proposed_changes)
    _render_list(console, "Next Steps", parsed.next_steps)
    if parsed.file_operations:
        operation_lines = [f"- {item.action}: {item.path}" for item in parsed.file_operations]
        _safe_panel(console, "\n".join(operation_lines), title="File Operations")

    console.print(f"Session saved to state/sessions/{session.session_id}.json")
    if session.patch_proposal_path:
        console.print(f"Patch proposal saved to {session.patch_proposal_path}")
    if session.mode.value == "apply" and not session.confirmed:
        console.print("Apply confirmation not provided. No files were changed.")
    if session.apply_log_path:
        console.print(f"Apply log saved to {session.apply_log_path}")


def _safe_panel(console: Console, text: str, title: str, style: str = "") -> None:
    # Replace problematic arrows and other non-ASCII characters for Windows CMD/PowerShell
    safe_text = text.replace("\u2192", "->").replace("\u21d2", "=>")
    # Generic encoding fallback for other weird chars
    safe_text = safe_text.encode("ascii", "replace").decode("ascii")
    console.print(Panel(safe_text, title=title, style=style))


def _render_list(console: Console, title: str, items: list[str]) -> None:
    if not items:
        return
    rendered = "\n".join(f"- {item}" for item in items)
    _safe_panel(console, rendered, title=title)
