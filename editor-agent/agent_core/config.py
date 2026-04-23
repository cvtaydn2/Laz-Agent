from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class Settings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

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
        default="moonshotai/kimi-k2-instruct",
        alias="NVIDIA_MODEL",
    )
    temperature: float = Field(default=0.1, alias="AGENT_TEMPERATURE")
    timeout_seconds: float = Field(default=12.0, alias="AGENT_TIMEOUT_SECONDS")
    max_file_bytes: int = Field(default=120000, alias="AGENT_MAX_FILE_BYTES")
    max_chars_per_file: int = Field(default=2500, alias="AGENT_MAX_CHARS_PER_FILE")
    max_context_chars: int = Field(default=12000, alias="AGENT_MAX_CONTEXT_CHARS")
    top_k_files: int = Field(default=5, alias="AGENT_TOP_K_FILES")
    max_completion_tokens: int = Field(default=300, alias="AGENT_MAX_COMPLETION_TOKENS")
    server_host: str = Field(default="127.0.0.1", alias="AGENT_SERVER_HOST")
    server_port: int = Field(default=8000, alias="AGENT_SERVER_PORT")

    @classmethod
    def load(cls) -> "Settings":
        load_environment_from_cwd()
        settings = cls.model_validate(
            {
                "NVIDIA_API_KEY": os.getenv("NVIDIA_API_KEY", ""),
                "NVIDIA_BASE_URL": os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                "NVIDIA_MODEL": os.getenv("NVIDIA_MODEL", "moonshotai/kimi-k2-instruct"),
                "AGENT_TEMPERATURE": float(os.getenv("AGENT_TEMPERATURE", "0.1")),
                "AGENT_TIMEOUT_SECONDS": float(os.getenv("AGENT_TIMEOUT_SECONDS", "12")),
                "AGENT_MAX_FILE_BYTES": int(os.getenv("AGENT_MAX_FILE_BYTES", "120000")),
                "AGENT_MAX_CHARS_PER_FILE": int(os.getenv("AGENT_MAX_CHARS_PER_FILE", "2500")),
                "AGENT_MAX_CONTEXT_CHARS": int(os.getenv("AGENT_MAX_CONTEXT_CHARS", "12000")),
                "AGENT_TOP_K_FILES": int(os.getenv("AGENT_TOP_K_FILES", "5")),
                "AGENT_MAX_COMPLETION_TOKENS": int(os.getenv("AGENT_MAX_COMPLETION_TOKENS", "300")),
                "AGENT_SERVER_HOST": os.getenv("AGENT_SERVER_HOST", "127.0.0.1"),
                "AGENT_SERVER_PORT": int(os.getenv("AGENT_SERVER_PORT", "8000")),
            }
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


def load_environment_from_cwd() -> str:
    dotenv_path = find_dotenv(usecwd=True)
    load_dotenv(dotenv_path=dotenv_path, override=True)
    print(f"Loaded .env from: {dotenv_path}")
    print(f"NVIDIA_API_KEY present: {bool(os.getenv('NVIDIA_API_KEY'))}")
    return dotenv_path


def ensure_environment_ready() -> str:
    dotenv_path = load_environment_from_cwd()
    if not os.getenv("NVIDIA_API_KEY"):
        raise RuntimeError("NVIDIA_API_KEY is missing. Check your .env file or environment variables.")
    return dotenv_path
