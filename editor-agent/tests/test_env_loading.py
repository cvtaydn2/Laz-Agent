from __future__ import annotations

import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_core.config import Settings, ensure_environment_ready, load_environment_from_cwd


class EnvironmentLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = Path(__file__).resolve().parent / ".tmp-env-tests"
        if self.temp_root.exists():
            shutil.rmtree(self.temp_root)
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.temp_root.exists():
            shutil.rmtree(self.temp_root)

    def test_missing_api_key_raises_runtime_error(self) -> None:
        test_dir = self.temp_root / "missing-env"
        test_dir.mkdir(parents=True, exist_ok=True)
        with patch.dict(os.environ, {}, clear=True):
            previous_cwd = Path.cwd()
            try:
                os.chdir(test_dir)
                with patch("agent_core.config.find_dotenv", return_value=""):
                    with self.assertRaises(RuntimeError):
                        ensure_environment_ready()
            finally:
                os.chdir(previous_cwd)

    def test_env_file_in_project_root_is_loaded(self) -> None:
        test_dir = self.temp_root / "present-env"
        test_dir.mkdir(parents=True, exist_ok=True)
        env_path = test_dir / ".env"
        env_path.write_text("NVIDIA_API_KEY=nvapi-test-key\n", encoding="utf-8")

        with patch.dict(os.environ, {}, clear=True):
            previous_cwd = Path.cwd()
            try:
                os.chdir(test_dir)
                with patch("agent_core.config.find_dotenv", return_value=str(env_path)):
                    dotenv_path = ensure_environment_ready()
                    settings = Settings.load()
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(dotenv_path, str(env_path))
        self.assertEqual(settings.nvidia_api_key, "nvapi-test-key")

    def test_existing_environment_value_is_preserved_when_dotenv_missing(self) -> None:
        test_dir = self.temp_root / "system-env"
        test_dir.mkdir(parents=True, exist_ok=True)
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "system-key"}, clear=True):
            previous_cwd = Path.cwd()
            try:
                os.chdir(test_dir)
                with patch("agent_core.config.find_dotenv", return_value=""):
                    dotenv_path = load_environment_from_cwd()
                    settings = Settings.load()
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(dotenv_path, "")
        self.assertEqual(settings.nvidia_api_key, "system-key")


if __name__ == "__main__":
    unittest.main()
