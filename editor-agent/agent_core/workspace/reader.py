from __future__ import annotations

from pathlib import Path

from agent_core.config import Settings
from agent_core.models import FileContext, RankedFile
from agent_core.workspace.filters import is_probably_binary


class WorkspaceReader:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def read_ranked_files(
        self,
        workspace_path: Path,
        ranked_files: list[RankedFile],
    ) -> tuple[list[FileContext], list[str]]:
        selected_context: list[FileContext] = []
        notes: list[str] = []
        remaining_chars = self.settings.max_context_chars

        for ranked in ranked_files:
            file_path = workspace_path / ranked.relative_path
            try:
                raw_bytes = file_path.read_bytes()
            except OSError as exc:
                notes.append(f"Skipped unreadable file {ranked.relative_path}: {exc}")
                continue

            if len(raw_bytes) > self.settings.max_file_bytes:
                notes.append(
                    f"Skipped large file {ranked.relative_path}: {len(raw_bytes)} bytes exceeds limit."
                )
                continue

            if is_probably_binary(raw_bytes):
                notes.append(f"Skipped binary-looking file {ranked.relative_path}.")
                continue

            try:
                content = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                content = raw_bytes.decode("utf-8", errors="replace")

            normalized_content = content.strip()
            if not normalized_content:
                notes.append(f"Skipped empty file {ranked.relative_path}.")
                continue

            budgeted_content = normalized_content[:remaining_chars]
            if not budgeted_content:
                break

            selected_context.append(
                FileContext(
                    path=ranked.path,
                    relative_path=ranked.relative_path,
                    content=budgeted_content,
                    size_bytes=ranked.size_bytes,
                    score=ranked.score,
                    reason=ranked.reason,
                )
            )
            remaining_chars -= len(budgeted_content)
            if remaining_chars <= 0:
                notes.append("Stopped adding context after reaching max context char budget.")
                break

        return selected_context, notes
