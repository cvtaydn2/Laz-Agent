from __future__ import annotations

import json
import re

from agent_core.models import ParsedAnswer, ProposedFileOperation, ReviewFinding


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

                matched_heading = next((key for key in self.HEADINGS if line.upper() == key), None)
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
            summary=self._string_or_default(payload.get("summary"), text),
            findings=self._to_string_list(payload.get("findings")),
            suggestions=self._to_string_list(payload.get("suggestions")),
            commands_to_consider=self._to_string_list(payload.get("commands_to_consider")),
            risks=self._to_string_list(payload.get("risks")),
            affected_files=self._to_string_list(payload.get("affected_files")),
            proposed_changes=self._to_string_list(payload.get("proposed_changes")),
            next_steps=self._to_string_list(payload.get("next_steps")),
            file_operations=self._parse_json_file_operations(payload.get("file_operations")),
            raw_text=text,
            parse_strategy="json",
            review_findings=self._parse_review_findings(payload.get("findings")),
            risks_text=self._string_or_default(payload.get("risks"), ""),
            next_steps_text=self._string_or_default(payload.get("next_steps"), ""),
        )

    def _parse_file_operations(self, text: str) -> list[ProposedFileOperation]:
        operations: list[ProposedFileOperation] = []
        
        # 1. Original BEGIN_FILE / END_FILE format (Legacy/Strict)
        strict_pattern = re.compile(
            r"BEGIN_FILE\s+PATH:\s*(?P<path>[^\r\n]+)\s+ACTION:\s*(?P<action>[^\r\n]+)\s+CONTENT:\s*(?P<content>.*?)\s+END_FILE",
            re.DOTALL,
        )
        for match in strict_pattern.finditer(text):
            path = match.group("path").strip().strip("`").strip("'")
            action = match.group("action").strip().lower()
            content = match.group("content")
            operations.append(ProposedFileOperation(path=path, action=action, content=content.rstrip("\r\n")))

        # 2. Markdown Code Block Detection (Smart/Natural)
        # Matches: ```language (optional) [Path hint in comment or heading] ... ```
        # We look for "# File: path/to/file" or "// File: path/to/file" at the start of blocks
        md_pattern = re.compile(r"```[a-zA-Z]*\s+(?P<content>.*?)```", re.DOTALL)
        for match in md_pattern.finditer(text):
            content = match.group("content")
            # Look for path hint in the first 3 lines of the block
            path_hint = None
            first_lines = content.splitlines()[:3]
            for line in first_lines:
                # Matches: # File: path/to/file or // File: path/to/file or Path: path/to/file
                path_match = re.search(r"(?:#|//|Path:)\s*(?:File:)?\s*([a-zA-Z0-9_\-\./\\]+\.[a-zA-Z0-9]{1,5})", line, re.IGNORECASE)
                if path_match:
                    path_hint = path_match.group(1).strip()
                    break
            
            if path_hint and not any(op.path == path_hint for op in operations):
                # If we found a path hint and haven't already added this file via strict pattern
                operations.append(
                    ProposedFileOperation(
                        path=path_hint,
                        action="update", # Default to update for MD blocks
                        content=content.strip("\r\n")
                    )
                )
        
        return operations

    def _extract_json_candidate(self, text: str) -> str | None:
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return stripped

        fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if fenced_match:
            return fenced_match.group(1)

        inline_match = re.search(r"(\{[\s\S]*\"summary\"[\s\S]*\})", text)
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
