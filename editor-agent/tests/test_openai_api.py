from __future__ import annotations

import os
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient

from agent_core.models import AgentMode, ParsedAnswer, SessionRecord, WorkspaceSummary
from agent_core.nvidia_client import NvidiaTimeoutError
from agent_core.server.api import app


class OpenAIAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.env_patcher = patch.dict(
            os.environ,
            {
                "NVIDIA_API_KEY": "test-key",
                "AGENT_DEFAULT_WORKSPACE": "C:/repo/editor-agent",
            },
            clear=False,
        )
        self.env_patcher.start()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.env_patcher.stop()

    def test_missing_workspace_uses_default_workspace(self) -> None:
        fake_session = SessionRecord(
            session_id="ask-default-workspace",
            created_at=datetime.now(timezone.utc),
            mode=AgentMode.ASK,
            workspace_path="C:/repo/editor-agent",
            prompt="ask",
            user_input="Review this project",
            workspace_summary=WorkspaceSummary(
                root_path="C:/repo/editor-agent",
                total_files_scanned=0,
                included_files=0,
                skipped_files=0,
                top_extensions={},
                sampled_files=[],
            ),
            ranked_files=[],
            selected_context=[],
            raw_response="OK",
            parsed_response=ParsedAnswer(
                summary="OK",
            ),
        )

        with patch("agent_core.server.api.run_agent", return_value=fake_session) as mocked_run_agent:
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "laz-agent",
                    "messages": [{"role": "user", "content": "Review this project"}],
                },
            )

        self.assertEqual(response.status_code, 200)
        mocked_run_agent.assert_called_once()
        self.assertEqual(mocked_run_agent.call_args.args[1], "C:/repo/editor-agent")

    def test_stream_true_returns_sse_response(self) -> None:
        fake_session = SessionRecord(
            session_id="ask-123",
            created_at=datetime.now(timezone.utc),
            mode=AgentMode.ASK,
            workspace_path="C:/repo/editor-agent",
            prompt="ask",
            user_input="Sadece selam yaz.",
            workspace_summary=WorkspaceSummary(
                root_path="C:/repo/editor-agent",
                total_files_scanned=0,
                included_files=0,
                skipped_files=0,
                top_extensions={},
                sampled_files=[],
            ),
            ranked_files=[],
            selected_context=[],
            raw_response="Selam.",
            parsed_response=ParsedAnswer(
                summary="Selam.",
            ),
        )

        with patch("agent_core.server.api.run_agent", return_value=fake_session):
            with self.client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "laz-agent",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Sadece selam yaz."}],
                    "extra_body": {"workspace": "C:/repo/editor-agent", "mode": "ask"},
                },
            ) as response:
                body = response.read().decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertIn("chat.completion.chunk", body)
        self.assertIn("[DONE]", body)

    def test_review_mode_returns_openai_compatible_body(self) -> None:
        fake_session = SessionRecord(
            session_id="review-123",
            created_at=datetime.now(timezone.utc),
            mode=AgentMode.REVIEW,
            workspace_path="C:/repo/editor-agent",
            prompt="review",
            user_input="Review the recent changes.",
            workspace_summary=WorkspaceSummary(
                root_path="C:/repo/editor-agent",
                total_files_scanned=0,
                included_files=0,
                skipped_files=0,
                top_extensions={},
                sampled_files=[],
            ),
            ranked_files=[],
            selected_context=[],
            raw_response="",
            parsed_response=ParsedAnswer(
                summary="Review completed with one verified finding.",
                risks_text="Moderate risk.",
                next_steps_text="Fix the verified finding.",
            ),
        )

        with patch("agent_core.server.api.run_agent", return_value=fake_session):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "laz-agent",
                    "messages": [{"role": "user", "content": "Review the recent changes."}],
                    "extra_body": {
                        "workspace": "C:/repo/editor-agent",
                        "mode": "review",
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["choices"][0]["message"]["role"], "assistant")
        self.assertIn("summary", payload["choices"][0]["message"]["content"])

    def test_backend_timeout_returns_valid_openai_shape(self) -> None:
        with patch("agent_core.server.api.run_agent", side_effect=NvidiaTimeoutError()):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "laz-agent",
                    "messages": [{"role": "user", "content": "What does this project do?"}],
                    "extra_body": {
                        "workspace": "C:/repo/editor-agent",
                        "mode": "ask",
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["choices"][0]["message"]["role"], "assistant")
        self.assertIn("timed out", payload["choices"][0]["message"]["content"].lower())


if __name__ == "__main__":
    unittest.main()
