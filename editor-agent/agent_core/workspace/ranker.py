from __future__ import annotations

from pathlib import Path

from agent_core.config import Settings
from agent_core.models import AgentMode, FileScanResult, RankedFile


class WorkspaceRanker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def rank(
        self,
        workspace_path: Path,
        files: list[FileScanResult],
        mode: AgentMode,
        user_input: str | None,
        preferred_files: list[str] | None = None,
    ) -> list[RankedFile]:
        normalized_preferred = {
            item.replace("\\", "/").strip().lower()
            for item in (preferred_files or [])
            if item and item.strip()
        }
        query_terms = {
            part.lower()
            for part in (user_input or "").replace('"', " ").replace("'", " ").split()
            if len(part) >= 3
        }
        ranked: list[RankedFile] = []

        for file_item in files:
            score = 0.0
            reasons: list[str] = []
            relative_lower = file_item.relative_path.lower()
            file_name_lower = Path(file_item.relative_path).name.lower()

            if "readme" in file_name_lower:
                score += 8
                reasons.append("README files usually describe the project.")
            if file_name_lower in {"pyproject.toml", "package.json", "requirements.txt", "setup.py"}:
                score += 7
                reasons.append("Core project configuration.")
            if "requirements" in file_name_lower or "package.json" in file_name_lower:
                score += 7
                reasons.append("Dependency or package manifest.")
            if "main" in file_name_lower or "app" in file_name_lower:
                score += 5
                reasons.append("Potential entry point.")
            if file_name_lower in {"__main__.py", "manage.py"}:
                score += 6
                reasons.append("Known Python entry point.")
            if file_item.extension in {".py", ".ts", ".tsx", ".js", ".jsx"}:
                score += 3
                reasons.append("Source code file.")
            if file_item.extension in {".md", ".json", ".yml", ".yaml"}:
                score += 2
                reasons.append("Configuration or documentation file.")
            if len(Path(file_item.relative_path).parts) == 1:
                score += 2
                reasons.append("Top-level project file.")

            matched_terms = [term for term in query_terms if term in relative_lower]
            if matched_terms:
                score += 2 * len(matched_terms)
                reasons.append(f"Matches request terms: {', '.join(sorted(matched_terms))}.")

            if relative_lower in normalized_preferred:
                score += 50
                reasons.append("Explicitly requested file focus.")

            if mode == AgentMode.SUGGEST and any(
                marker in relative_lower
                for marker in ("readme", "requirements", "pyproject", "package", "docker", "compose")
            ):
                score += 3
                reasons.append("Likely useful for run/setup suggestions.")
            if mode == AgentMode.REVIEW:
                if relative_lower in normalized_preferred:
                    score += 20
                    reasons.append("Prioritized for review.")
                if any(marker in relative_lower for marker in ("readme", "pyproject", "requirements", "package", "config", "settings")):
                    score += 4
                    reasons.append("Relevant project context for review.")

            ranked.append(
                RankedFile(
                    path=file_item.path,
                    relative_path=file_item.relative_path,
                    extension=file_item.extension,
                    size_bytes=file_item.size_bytes,
                    score=score,
                    reason=" ".join(dict.fromkeys(reasons)) or "General project relevance.",
                )
            )

        ranked.sort(key=lambda item: (-item.score, item.relative_path))
        return self._ensure_required_files(ranked)

    def _ensure_required_files(self, ranked: list[RankedFile]) -> list[RankedFile]:
        required_patterns = (
            "readme",
            "pyproject.toml",
            "requirements.txt",
            "package.json",
            "setup.py",
            "main.py",
            "app.py",
            "__main__.py",
            "manage.py",
        )
        selected: list[RankedFile] = []
        seen: set[str] = set()

        for item in ranked:
            name = Path(item.relative_path).name.lower()
            if any(pattern == name or pattern in name for pattern in required_patterns):
                selected.append(item)
                seen.add(item.relative_path)
            if len(selected) >= self.settings.top_k_files:
                return selected[: self.settings.top_k_files]

        for item in ranked:
            if item.relative_path in seen:
                continue
            selected.append(item)
            seen.add(item.relative_path)
            if len(selected) >= self.settings.top_k_files:
                break

        return selected[: self.settings.top_k_files]
