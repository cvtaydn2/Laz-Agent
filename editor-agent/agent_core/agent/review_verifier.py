from __future__ import annotations

from pathlib import Path

from agent_core.config import Settings
from agent_core.models import FileContext, ParsedAnswer, ReviewFinding


class ReviewVerifier:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def verify(
        self,
        workspace_path: Path,
        parsed: ParsedAnswer,
        selected_context: list[FileContext],
    ) -> ParsedAnswer:
        verified = parsed.model_copy(deep=True)
        context_map = {item.relative_path.replace("\\", "/"): item.content for item in selected_context}
        findings: list[ReviewFinding] = []

        for finding in verified.review_findings:
            normalized_path = finding.file.replace("\\", "/").strip()
            target_path = (workspace_path / normalized_path).resolve()
            try:
                target_path.relative_to(workspace_path.resolve())
            except ValueError:
                continue
            if not target_path.exists() or not target_path.is_file():
                continue

            file_content = context_map.get(normalized_path)
            if file_content is None:
                try:
                    file_content = target_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue

            evidence = finding.evidence.strip()
            if not evidence:
                continue
            if evidence not in file_content:
                continue

            findings.append(
                finding.model_copy(
                    update={
                        "file": normalized_path,
                        "severity": self._normalize_severity(finding.severity),
                    }
                )
            )

        verified.review_findings = findings
        verified.findings = [
            f"[{item.severity}] {item.file}: {item.title}"
            for item in findings
        ]
        if not verified.risks_text:
            verified.risks_text = (
                "Some findings were filtered during verification."
                if len(findings) < len(parsed.review_findings)
                else "Review findings were verified against the available code context."
            )
        if not verified.next_steps_text:
            verified.next_steps_text = "Address high-severity findings first, then re-run review."
        return verified

    def _normalize_severity(self, severity: str) -> str:
        normalized = severity.strip().lower()
        if normalized not in {"low", "medium", "high"}:
            return "low"
        return normalized
