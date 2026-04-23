from __future__ import annotations

import unittest

from agent_core.agent.response_parser import ResponseParser
from agent_core.server.openai_adapter import extract_request_mode, extract_request_workspace, extract_user_message
from agent_core.server.openai_schemas import OpenAIChatCompletionRequest


class OpenAICompatibilityTests(unittest.TestCase):
    def test_extracts_workspace_and_mode_from_extra_body(self) -> None:
        request = OpenAIChatCompletionRequest(
            model="laz-agent",
            messages=[{"role": "user", "content": "What does this project do?"}],
            extra_body={
                "workspace": "C:/repo/editor-agent",
                "mode": "ask",
            },
        )
        self.assertEqual(extract_request_workspace(request), "C:/repo/editor-agent")
        self.assertEqual(extract_request_mode(request), "ask")

    def test_extracts_last_user_message(self) -> None:
        request = OpenAIChatCompletionRequest(
            model="laz-agent",
            messages=[
                {"role": "system", "content": "You are a code assistant."},
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Reply"},
                {"role": "user", "content": "Second"},
            ],
            extra_body={"workspace": "C:/repo/editor-agent"},
        )
        self.assertEqual(extract_user_message(request.messages), "Second")

    def test_json_parser_fallback(self) -> None:
        parser = ResponseParser()
        parsed = parser.parse(
            '{"summary":"Project summary","findings":["One"],"suggestions":["Two"]}'
        )
        self.assertEqual(parsed.summary, "Project summary")
        self.assertEqual(parsed.findings, ["One"])
        self.assertEqual(parsed.suggestions, ["Two"])
        self.assertEqual(parsed.parse_strategy, "json")


if __name__ == "__main__":
    unittest.main()
