from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv
from pydantic import BaseModel, Field, HttpUrl


class Settings(BaseModel):
    DEFAULT_IGNORE_DIRS: ClassVar[tuple[str, ...]] = (
        ".git",
        "node_modules",
        "dist",
        "build",
        ".next",
        "coverage",
        ".venv",
        "venv",
        "__pycache__",
    )
    DEFAULT_ALLOWED_EXTENSIONS: ClassVar[tuple[str, ...]] = (
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".json",
        ".md",
        ".txt",
        ".yml",
        ".yaml",
        ".html",
        ".css",
        ".env.example",
    )

    nvidia_api_key: str = Field(default="", alias="NVIDIA_API_KEY")
    nvidia_base_url: HttpUrl = Field(
        default="https://integrate.api.nvidia.com/v1",
        alias="NVIDIA_BASE_URL",
    )
    nvidia_model: str = Field(
        default="minimaxai/minimax-m2.7",
        alias="NVIDIA_MODEL",
    )
    temperature: float = Field(default=0.2, alias="AGENT_TEMPERATURE")
    timeout_seconds: float = Field(default=60.0, alias="AGENT_TIMEOUT_SECONDS")
    max_file_bytes: int = Field(default=200000, alias="AGENT_MAX_FILE_BYTES")
    max_context_chars: int = Field(default=24000, alias="AGENT_MAX_CONTEXT_CHARS")
    top_k_files: int = Field(default=8, alias="AGENT_TOP_K_FILES")
    server_host: str = Field(default="127.0.0.1", alias="AGENT_SERVER_HOST")
    server_port: int = Field(default=8000, alias="AGENT_SERVER_PORT")

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv(override=False)
        settings = cls(
            nvidia_api_key=os.getenv("NVIDIA_API_KEY", ""),
            nvidia_base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            nvidia_model=os.getenv("NVIDIA_MODEL", "minimaxai/minimax-m2.7"),
            temperature=float(os.getenv("AGENT_TEMPERATURE", "0.2")),
            timeout_seconds=float(os.getenv("AGENT_TIMEOUT_SECONDS", "60")),
            max_file_bytes=int(os.getenv("AGENT_MAX_FILE_BYTES", "200000")),
            max_context_chars=int(os.getenv("AGENT_MAX_CONTEXT_CHARS", "24000")),
            top_k_files=int(os.getenv("AGENT_TOP_K_FILES", "8")),
            server_host=os.getenv("AGENT_SERVER_HOST", "127.0.0.1"),
            server_port=int(os.getenv("AGENT_SERVER_PORT", "8000")),
        )
        settings.ensure_state_dirs()
        return settings

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def state_dir(self) -> Path:
        return self.project_root / "state"

    @property
    def session_dir(self) -> Path:
        return self.state_dir / "sessions"

    @property
    def logs_dir(self) -> Path:
        return self.state_dir / "logs"

    @property
    def patches_dir(self) -> Path:
        return self.state_dir / "patches"

    @property
    def backups_dir(self) -> Path:
        return self.state_dir / "backups"

    @property
    def ignored_directories(self) -> set[str]:
        return set(self.DEFAULT_IGNORE_DIRS)

    @property
    def allowed_extensions(self) -> set[str]:
        return set(self.DEFAULT_ALLOWED_EXTENSIONS)

    def ensure_state_dirs(self) -> None:
        for directory in (
            self.state_dir,
            self.session_dir,
            self.logs_dir,
            self.patches_dir,
            self.backups_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
