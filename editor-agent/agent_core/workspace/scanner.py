from __future__ import annotations

import os
import asyncio
from collections import Counter
from pathlib import Path

from agent_core.config import Settings
from agent_core.models import FileScanResult, WorkspaceSummary
from agent_core.workspace.filters import is_allowed_file, is_ignored_directory
import time

# Global in-memory cache for scan results to provide near-instant consecutive requests
_SCAN_CACHE: dict[str, tuple[float, list[FileScanResult], WorkspaceSummary]] = {}
_SCAN_TTL = 60.0  # 60 seconds cache life


class WorkspaceScanner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def scan(self, workspace_path: Path) -> tuple[list[FileScanResult], WorkspaceSummary]:
        ws_key = str(workspace_path.resolve())
        now = time.time()
        
        if ws_key in _SCAN_CACHE:
            timestamp, results, summary = _SCAN_CACHE[ws_key]
            if now - timestamp < _SCAN_TTL:
                return results, summary
                
        results, summary = await asyncio.to_thread(self._sync_scan, workspace_path)
        _SCAN_CACHE[ws_key] = (now, results, summary)
        return results, summary

    def _sync_scan(self, workspace_path: Path) -> tuple[list[FileScanResult], WorkspaceSummary]:
        included: list[FileScanResult] = []
        skipped_files = 0
        extension_counter: Counter[str] = Counter()

        for current_root_str, dir_names, file_names in os.walk(workspace_path, topdown=True):
            current_root = Path(current_root_str)
            dir_names[:] = [
                directory
                for directory in dir_names
                if not is_ignored_directory(current_root / directory, self.settings)
            ]

            for file_name in file_names:
                file_path = current_root / file_name
                if not is_allowed_file(file_path, self.settings):
                    skipped_files += 1
                    continue

                try:
                    stat = file_path.stat()
                except OSError:
                    skipped_files += 1
                    continue

                relative_path = file_path.relative_to(workspace_path)
                extension = file_path.suffix.lower() or file_path.name.lower()
                extension_counter[extension] += 1
                included.append(
                    FileScanResult(
                        path=str(file_path.resolve()),
                        relative_path=str(relative_path).replace("\\", "/"),
                        extension=extension,
                        size_bytes=stat.st_size,
                    )
                )

        summary = WorkspaceSummary(
            root_path=str(workspace_path.resolve()),
            total_files_scanned=len(included) + skipped_files,
            included_files=len(included),
            skipped_files=skipped_files,
            top_extensions=dict(extension_counter.most_common(10)),
            sampled_files=[item.relative_path for item in included[:10]],
            notes=[
                "Ignored common build, dependency, cache, and VCS directories.",
                "Allowed file types are limited to safe text/code extensions.",
            ],
        )
        return included, summary
