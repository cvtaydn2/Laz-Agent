from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from agent_core.config import Settings
from agent_core.models import ApplyLogRecord, AppliedFileRecord, ProposedFileOperation


class ApplyError(Exception):
    pass


class ApplyEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def apply(
        self,
        workspace_path: Path,
        session_id: str,
        created_at: datetime,
        request: str | None,
        operations: list[ProposedFileOperation],
    ) -> ApplyLogRecord:
        if not operations:
            return ApplyLogRecord(
                session_id=session_id,
                created_at=created_at,
                workspace_path=str(workspace_path.resolve()),
                confirmed=True,
                success=False,
                request=request,
                error="No valid file operations were available to apply.",
            )

        preflight_error = self._preflight_validate(workspace_path, operations)
        if preflight_error:
            return ApplyLogRecord(
                session_id=session_id,
                created_at=created_at,
                workspace_path=str(workspace_path.resolve()),
                confirmed=True,
                success=False,
                request=request,
                error=preflight_error,
            )

        written_records: list[AppliedFileRecord] = []

        try:
            for operation in operations:
                target_path = self._resolve_target_path(workspace_path, operation.path)
                backup_path = self._backup_file(workspace_path, session_id, target_path)
                self._write_file(target_path, operation.content)
                written_records.append(
                    AppliedFileRecord(
                        path=operation.path,
                        action="update",
                        backup_path=str(backup_path) if backup_path.exists() else None,
                        existed_before=backup_path.exists(),
                    )
                )
        except Exception as exc:
            for record in reversed(written_records):
                self._rollback(workspace_path, [record])
            return ApplyLogRecord(
                session_id=session_id,
                created_at=created_at,
                workspace_path=str(workspace_path.resolve()),
                confirmed=True,
                success=False,
                rollback_performed=True,
                request=request,
                files_written=written_records,
                error=str(exc),
            )

        return ApplyLogRecord(
            session_id=session_id,
            created_at=created_at,
            workspace_path=str(workspace_path.resolve()),
            confirmed=True,
            success=True,
            request=request,
            files_written=written_records,
        )

    def rollback(self, workspace_path: Path, log: ApplyLogRecord) -> bool:
        """
        Manually rollback a previous apply operation using its log record.
        """
        if not log.files_written:
            return False
        
        try:
            self._rollback(workspace_path, log.files_written)
            return True
        except Exception:
            return False

    def _resolve_target_path(self, workspace_path: Path, relative_path: str) -> Path:
        candidate = (workspace_path / relative_path).resolve()
        workspace_root = workspace_path.resolve()
        try:
            candidate.relative_to(workspace_root)
        except ValueError as exc:
            raise ApplyError(f"Refusing to write outside workspace: {relative_path}") from exc
        return candidate

    ALLOWED_EXTENSIONS = {
        ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".md",
        ".txt", ".yml", ".yaml", ".html", ".css", ".env.example"
    }

    def _preflight_validate(
        self,
        workspace_path: Path,
        operations: list[ProposedFileOperation],
    ) -> str | None:
        for operation in operations:
            if operation.action not in {"update", "create", "delete"}:
                return f"Unsupported apply action: {operation.action}"
            
            target_path = self._resolve_target_path(workspace_path, operation.path)
            
            # Security: Check file extension
            if target_path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
                return f"Refusing to modify/create file with restricted extension: {operation.path}"

            # If it's an update, the file MUST exist. 
            # If it's a create, it can be new.
            if operation.action == "update" and not target_path.exists():
                return (
                    "Update operation requested for a non-existent file. "
                    f"Target: {operation.path}"
                )
        return None

    def _backup_file(self, workspace_path: Path, session_id: str, target_path: Path) -> Path:
        backup_root = self.settings.backups_dir / session_id
        relative = target_path.resolve().relative_to(workspace_path.resolve())
        backup_path = backup_root / relative
        if target_path.exists():
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target_path, backup_path)
        return backup_path

    def _write_file(self, target_path: Path, content: str) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content, encoding="utf-8")

    def _rollback(self, workspace_path: Path, written_records: list[AppliedFileRecord]) -> None:
        for record in reversed(written_records):
            target_path = self._resolve_target_path(workspace_path, record.path)
            if record.backup_path and record.backup_path != "None":
                backup_path = Path(record.backup_path)
                if backup_path.exists():
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_path, target_path)
            elif not record.existed_before:
                if target_path.exists():
                    target_path.unlink()
