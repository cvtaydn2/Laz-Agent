from __future__ import annotations

import json
import re

from agent_core.models import ParsedAnswer, ProposedFileOperation, ReviewFinding

# Pre-compiled regex patterns (module-level for performance)
_STRICT_FILE_PATTERN = re.compile(
    r"BEGIN_FILE\s+PATH:\s*(?P<path>[^\r\n]+)\s+ACTION:\s*(?P<action>[^\r\n]+)\s+CONTENT:\s*(?P<content>.*?)\s+END_FILE",
    re.DOTALL,
)
_MD_BLOCK_PATTERN = re.compile(r"```[a-zA-Z]*\s+(?P<content>.*?)```", re.DOTALL)
_MD_PATH_HINT_PATTERN = re.compile(
    r"(?:<!--|#|//|Path:)\s*(?:File:)?\s*([a-zA-Z0-9_\-\./\\]+\.[a-zA-Z0-9]{1,5})",
    re.IGNORECASE,
)
_COMMAND_PATTERN = re.compile(
    r"BEGIN_COMMAND\s+COMMAND:\s*(?P<command>[^\r\n]+)\s+RATIONALE:\s*(?P<rationale>.*?)\s+END_COMMAND",
    re.DOTALL,
)
_JSON_FENCED_PATTERN = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_JSON_INLINE_PATTERN = re.compile(r"(\{[\s\S]*\"summary\"[\s\S]*\})")


class ResponseParser:
    HEADINGS = {
        "thought": "thought",
        "reasoning": "thought",
        "rationale": "thought",
        "summary": "summary",
        "findings": "findings",
        "suggestions": "suggestions",
        "commands to consider": "commands_to_consider",
        "commands_to_consider": "commands_to_consider",
        "risks": "risks",
        "affected files": "affected_files",
        "affected_files": "affected_files",
        "proposed changes": "proposed_changes",
        "proposed_changes": "proposed_changes",
        "next steps": "next_steps",
        "next_steps": "next_steps",
        "file operations": "file_operations",
        "file_operations": "file_operations",
        "command_operations": "command_operations",
        "commands": "command_operations",
    }

    def parse(self, text: str) -> ParsedAnswer:
        json_result = self._parse_json(text)
        if json_result is not None:
            return json_result

        return self._parse_text(text)

    def _parse_text(self, text: str) -> ParsedAnswer:
        sections: dict[str, list[str]] = {value: [] for value in self.HEADINGS.values()}
        current_key = "summary"

        try:
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue

                matched_heading = next((key for key in self.HEADINGS if line.lower().rstrip(":") == key), None)
                if matched_heading:
                    current_key = self.HEADINGS[matched_heading]
                    continue

                normalized = line.removeprefix("- ").removeprefix("* ").strip()
                sections[current_key].append(normalized)
        except Exception:
            return ParsedAnswer(
                summary=text.strip() or "No model content returned.",
                raw_text=text,
                parse_strategy="raw_text",
            )

        summary = " ".join(sections["summary"]).strip() or text.strip() or "No model content returned."
        thought = "\n".join(sections["thought"]).strip()
        return ParsedAnswer(
            thought=thought,
            summary=summary,
            findings=sections["findings"],
            suggestions=sections["suggestions"],
            commands_to_consider=sections["commands_to_consider"],
            risks=sections["risks"],
            affected_files=sections["affected_files"],
            proposed_changes=sections["proposed_changes"],
            next_steps=sections["next_steps"],
            file_operations=self._parse_file_operations(text),
            command_operations=self._parse_command_operations(text),
            raw_text=text,
            parse_strategy="text",
        )

    def _parse_json(self, text: str) -> ParsedAnswer | None:
        json_candidate = self._extract_json_candidate(text)
        if json_candidate is None:
            return None

        try:
            payload = json.loads(json_candidate)
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict):
            return None

        return ParsedAnswer(
            thought=self._string_or_default(payload.get("thought") or payload.get("reasoning") or payload.get("rationale"), ""),
            summary=self._string_or_default(payload.get("summary"), text),
            findings=self._to_string_list(payload.get("findings")),
            suggestions=self._to_string_list(payload.get("suggestions")),
            commands_to_consider=self._to_string_list(payload.get("commands_to_consider")),
            risks=self._to_string_list(payload.get("risks")),
            affected_files=self._to_string_list(payload.get("affected_files")),
            proposed_changes=self._to_string_list(payload.get("proposed_changes")),
            next_steps=self._to_string_list(payload.get("next_steps")),
            file_operations=self._parse_json_file_operations(payload.get("file_operations")),
            command_operations=self._parse_json_command_operations(payload.get("command_operations")),
            raw_text=text,
            parse_strategy="json",
            review_findings=self._parse_review_findings(payload.get("findings")),
            risks_text=self._string_or_default(payload.get("risks"), ""),
            next_steps_text=self._string_or_default(payload.get("next_steps"), ""),
        )

    def _parse_file_operations(self, text: str) -> list[ProposedFileOperation]:
        operations: list[ProposedFileOperation] = []

        # 1. BEGIN_FILE / END_FILE strict format
        for match in _STRICT_FILE_PATTERN.finditer(text):
            path = match.group("path").strip().strip("`").strip("'")
            action = match.group("action").strip().lower()
            content = match.group("content")
            operations.append(ProposedFileOperation(path=path, action=action, content=content.rstrip("\r\n")))

        # 2. Markdown code block with path hint comment
        for match in _MD_BLOCK_PATTERN.finditer(text):
            content = match.group("content")
            path_hint = None
            for line in content.splitlines()[:3]:
                path_match = _MD_PATH_HINT_PATTERN.search(line)
                if path_match:
                    path_hint = path_match.group(1).strip()
                    break

            if path_hint and not any(op.path == path_hint for op in operations):
                operations.append(
                    ProposedFileOperation(
                        path=path_hint,
                        action="update",
                        content=content.strip("\r\n"),
                    )
                )

        return operations

    def _parse_command_operations(self, text: str) -> list[ProposedCommandOperation]:
        from agent_core.models import ProposedCommandOperation
        commands: list[ProposedCommandOperation] = []
        for match in _COMMAND_PATTERN.finditer(text):
            command = match.group("command").strip().strip("`").strip("'")
            rationale = match.group("rationale").strip()
            commands.append(ProposedCommandOperation(command=command, rationale=rationale))
        return commands

    def _parse_json_command_operations(self, value: object) -> list[ProposedCommandOperation]:
        from agent_core.models import ProposedCommandOperation
        if not isinstance(value, list):
            return []
        commands: list[ProposedCommandOperation] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            command = str(item.get("command", "")).strip()
            rationale = str(item.get("rationale", "")).strip()
            if command:
                commands.append(ProposedCommandOperation(command=command, rationale=rationale))
        return commands

    def _extract_json_candidate(self, text: str) -> str | None:
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return stripped

        fenced_match = _JSON_FENCED_PATTERN.search(text)
        if fenced_match:
            return fenced_match.group(1)

        inline_match = _JSON_INLINE_PATTERN.search(text)
        if inline_match:
            return inline_match.group(1)

        return None

    def _to_string_list(self, value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    def _parse_json_file_operations(self, value: object) -> list[ProposedFileOperation]:
        if not isinstance(value, list):
            return []

        operations: list[ProposedFileOperation] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path", "")).strip()
            action = str(item.get("action", "")).strip().lower()
            content = str(item.get("content", ""))
            if not path or not action:
                continue
            operations.append(
                ProposedFileOperation(
                    path=path,
                    action=action,
                    content=content,
                )
            )
        return operations

    def _parse_review_findings(self, value: object) -> list[ReviewFinding]:
        if not isinstance(value, list):
            return []

        findings: list[ReviewFinding] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            try:
                finding = ReviewFinding(
                    title=str(item.get("title", "")).strip(),
                    severity=str(item.get("severity", "")).strip().lower(),
                    file=str(item.get("file", "")).strip(),
                    evidence=str(item.get("evidence", "")).strip(),
                    issue=str(item.get("issue", "")).strip(),
                    suggested_fix=str(item.get("suggested_fix", "")).strip(),
                )
            except Exception:
                continue
            if finding.title and finding.file and finding.issue:
                findings.append(finding)
        return findings

    def _string_or_default(self, value: object, fallback: str) -> str:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return fallback.strip() or "No model content returned."
