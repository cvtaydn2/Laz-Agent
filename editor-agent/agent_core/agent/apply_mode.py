from __future__ import annotations

from agent_core.models import ParsedAnswer, ProposedFileOperation


class ApplyModePolicy:
    def apply(self, parsed: ParsedAnswer) -> ParsedAnswer:
        updated = parsed.model_copy(deep=True)
        allowed_operations: list[ProposedFileOperation] = []
        seen_paths: set[str] = set()

        for operation in updated.file_operations:
            normalized_path = operation.path.strip().strip("`")
            normalized_action = operation.action.strip().lower()
            if not normalized_path:
                continue
            if normalized_action not in {"update"}:
                continue
            if normalized_path in seen_paths:
                continue
            seen_paths.add(normalized_path)
            allowed_operations.append(
                ProposedFileOperation(
                    path=normalized_path,
                    action="update",
                    content=operation.content,
                )
            )

        updated.file_operations = allowed_operations
        updated.affected_files = [item.path for item in allowed_operations] or updated.affected_files

        if not updated.proposed_changes and allowed_operations:
            updated.proposed_changes = [
                f"{item.path}: update this file with the proposed content."
                for item in allowed_operations
            ]

        if not updated.next_steps:
            updated.next_steps.append(
                "Run apply again with --confirm only after reviewing the proposed file operations."
            )

        if not updated.risks:
            updated.risks.append(
                "Generated file contents should be reviewed manually before confirmed apply."
            )

        return updated
