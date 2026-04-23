from __future__ import annotations

from agent_core.models import ParsedAnswer


class SuggestionPolicy:
    def apply(self, parsed: ParsedAnswer) -> ParsedAnswer:
        updated = parsed.model_copy(deep=True)
        if not updated.suggestions:
            updated.suggestions.append("No concrete suggestions were produced from the current context.")

        safe_commands: list[str] = []
        for command in updated.commands_to_consider:
            normalized = command.strip()
            lowered = normalized.lower()
            if any(token in lowered for token in ("rm ", "del ", "rmdir", "format ", "mkfs", "shutdown")):
                continue
            safe_commands.append(normalized)

        updated.commands_to_consider = safe_commands
        return updated
