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
        # Filter out common noise words to focus on real intent
        stop_words = {"the", "and", "for", "with", "from", "this", "that", "how", "what", "why", "can", "you", "den", "dan", "bir", "icin", "nasıl", "neden", "proje", "dosya"}
        query_terms = {
            part.lower()
            for part in (user_input or "").replace('"', " ").replace("'", " ").replace("/", " ").replace("\\", " ").split()
            if len(part) >= 3 and part.lower() not in stop_words
        }
        
        ranked: list[RankedFile] = []

        for file_item in files:
            score = 0.0
            reasons: list[str] = []
            relative_lower = file_item.relative_path.lower().replace("\\", "/")
            file_name_lower = Path(file_item.relative_path).name.lower()
            parts = relative_lower.split("/")

            # 1. Structural Significance
            if "readme" in file_name_lower:
                score += 15  # Increased priority for project maps
                reasons.append("Project overview (README).")
            if file_name_lower in {"package.json", "pyproject.toml", "requirements.txt", "go.mod", "tsconfig.json"}:
                score += 12
                reasons.append("Core project configuration.")
            if "config" in file_name_lower or "settings" in file_name_lower:
                score += 8
                reasons.append("Configuration file.")
            
            # 2. Contextual Term Matching (Smart Search)
            matched_terms = [term for term in query_terms if term in relative_lower]
            if matched_terms:
                # Higher weight for matching the actual filename vs directory name
                for term in matched_terms:
                    if term in file_name_lower:
                        score += 10
                    else:
                        score += 5
                reasons.append(f"Intent match: {', '.join(sorted(matched_terms))}.")

            # 3. Directory Context (e.g., if asking about 'api', boost 'src/api/')
            if any(term in parts for term in query_terms):
                score += 15
                reasons.append("High directory relevance.")

            # 4. Explicit Focus (Continue context or user-provided files)
            if relative_lower in normalized_preferred:
                score += 100 # Maximum priority
                reasons.append("Currently active/requested file.")

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
        return self._ensure_required_files(ranked, normalized_preferred)

    def _ensure_required_files(
        self,
        ranked: list[RankedFile],
        preferred_files: set[str],
    ) -> list[RankedFile]:
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
            normalized_path = item.relative_path.replace("\\", "/").lower()
            if normalized_path in preferred_files:
                selected.append(item)
                seen.add(item.relative_path)
            if len(selected) >= self.settings.top_k_files:
                return selected[: self.settings.top_k_files]

        for item in ranked:
            name = Path(item.relative_path).name.lower()
            if item.relative_path in seen:
                continue
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
