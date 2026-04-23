from __future__ import annotations

from agent_core.models import ParsedAnswer


class PatchPreviewPolicy:
    def apply(self, parsed: ParsedAnswer) -> ParsedAnswer:
        updated = parsed.model_copy(deep=True)

        cleaned_files: list[str] = []
        for file_path in updated.affected_files:
            normalized = file_path.strip().strip("`")
            if normalized and normalized not in cleaned_files:
                cleaned_files.append(normalized)

        updated.affected_files = cleaned_files

        if not updated.proposed_changes:
            updated.proposed_changes.append(
                "No concrete file-by-file changes were produced from the current context."
            )

        if not updated.next_steps:
            updated.next_steps.append(
                "Review the proposed changes manually before implementing any file edits."
            )

        return updated
