from __future__ import annotations

import unittest

from agent_core.agent.response_parser import ResponseParser
from agent_core.server.openai_adapter import (
    extract_changed_files,
    extract_request_mode,
    extract_request_workspace,
    extract_user_message,
    json_review_text,
    validate_openai_mode,
)
from agent_core.server.openai_schemas import OpenAIChatCompletionRequest
from agent_core.models import AgentMode, ParsedAnswer, SessionRecord, WorkspaceSummary
from datetime import datetime, timezone


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
        self.assertEqual(validate_openai_mode("review"), "review")

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

    def test_extracts_changed_files(self) -> None:
        request = OpenAIChatCompletionRequest(
            model="laz-agent",
            messages=[{"role": "user", "content": "Review this diff"}],
            extra_body={
                "workspace": "C:/repo/editor-agent",
                "mode": "review",
                "changed_files": ["src/main.py", "README.md"],
            },
        )
        self.assertEqual(extract_changed_files(request), ["src/main.py", "README.md"])

    def test_json_parser_fallback(self) -> None:
        parser = ResponseParser()
        parsed = parser.parse(
            '{"summary":"Project summary","findings":["One"],"suggestions":["Two"]}'
        )
        self.assertEqual(parsed.summary, "Project summary")
        self.assertEqual(parsed.findings, ["One"])
        self.assertEqual(parsed.suggestions, ["Two"])
        self.assertEqual(parsed.parse_strategy, "json")

    def test_review_json_parser(self) -> None:
        parser = ResponseParser()
        parsed = parser.parse(
            '{"summary":"Review complete","findings":[{"title":"Bug","severity":"high","file":"app.py","evidence":"dangerous_call()","issue":"Potential bug","suggested_fix":"Add a guard"}],"risks":"Medium release risk","next_steps":"Fix and retest"}'
        )
        self.assertEqual(parsed.summary, "Review complete")
        self.assertEqual(len(parsed.review_findings), 1)
        self.assertEqual(parsed.review_findings[0].file, "app.py")

    def test_review_mode_formats_stable_json_content(self) -> None:
        session = SessionRecord(
            session_id="review-1",
            created_at=datetime.now(timezone.utc),
            mode=AgentMode.REVIEW,
            workspace_path="C:/repo/editor-agent",
            prompt="review",
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
                summary="Stable review",
                risks_text="Low release risk.",
                next_steps_text="Fix the finding.",
                review_findings=[],
            ),
        )
        content = json_review_text(session)
        self.assertIn('"summary"', content)
        self.assertIn('"findings"', content)


if __name__ == "__main__":
    unittest.main()
