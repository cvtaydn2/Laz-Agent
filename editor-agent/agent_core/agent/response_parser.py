from __future__ import annotations

import re

from agent_core.models import ParsedAnswer, ProposedFileOperation


class ResponseParser:
    HEADINGS = {
        "SUMMARY:": "summary",
        "FINDINGS:": "findings",
        "SUGGESTIONS:": "suggestions",
        "COMMANDS_TO_CONSIDER:": "commands_to_consider",
        "RISKS:": "risks",
        "AFFECTED_FILES:": "affected_files",
        "PROPOSED_CHANGES:": "proposed_changes",
        "NEXT_STEPS:": "next_steps",
    }

    def parse(self, text: str) -> ParsedAnswer:
        sections: dict[str, list[str]] = {value: [] for value in self.HEADINGS.values()}
        current_key = "summary"

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            matched_heading = next((key for key in self.HEADINGS if line.upper() == key), None)
            if matched_heading:
                current_key = self.HEADINGS[matched_heading]
                continue

            normalized = line.removeprefix("- ").removeprefix("* ").strip()
            sections[current_key].append(normalized)

        summary = " ".join(sections["summary"]).strip() or text.strip()
        return ParsedAnswer(
            summary=summary,
            findings=sections["findings"],
            suggestions=sections["suggestions"],
            commands_to_consider=sections["commands_to_consider"],
            risks=sections["risks"],
            affected_files=sections["affected_files"],
            proposed_changes=sections["proposed_changes"],
            next_steps=sections["next_steps"],
            file_operations=self._parse_file_operations(text),
        )

    def _parse_file_operations(self, text: str) -> list[ProposedFileOperation]:
        pattern = re.compile(
            r"BEGIN_FILE\s+PATH:\s*(?P<path>[^\r\n]+)\s+ACTION:\s*(?P<action>[^\r\n]+)\s+CONTENT:\s*(?P<content>.*?)\s+END_FILE",
            re.DOTALL,
        )
        operations: list[ProposedFileOperation] = []
        for match in pattern.finditer(text):
            path = match.group("path").strip().strip("`")
            action = match.group("action").strip().lower()
            content = match.group("content")
            operations.append(
                ProposedFileOperation(
                    path=path,
                    action=action,
                    content=content.rstrip("\r\n"),
                )
            )
        return operations
