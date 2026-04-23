import pytest
import asyncio
import json
import os
import httpx
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from agent_core.config import Settings
from agent_core.workspace.scanner import WorkspaceScanner
from agent_core.output.writers import SessionWriter
from agent_core.llm.nvidia import NvidiaProvider, NvidiaTimeoutError, NvidiaBackendError
from agent_core.models import (
    SessionRecord, AgentMode, WorkspaceSummary,
    ChatMessage, ParsedAnswer,
)
from agent_core.agent.orchestrator import AgentOrchestrator
from agent_core.server.openai_adapter import (
    build_openai_response,
    extract_last_user_text,
    extract_user_message,
    normalize_openai_messages,
    validate_openai_mode,
    extract_preferred_files,
)
from agent_core.server.openai_schemas import OpenAIMessage, OpenAIChatCompletionRequest
from agent_core.agent.response_parser import ResponseParser


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings(tmp_path):
    s = Settings(
        nvidia_api_key="test-key",
        session_dir=tmp_path / "sessions",
        logs_dir=tmp_path / "logs",
        patches_dir=tmp_path / "patches",
        backups_dir=tmp_path / "backups",
    )
    s.session_dir.mkdir(parents=True, exist_ok=True)
    s.logs_dir.mkdir(parents=True, exist_ok=True)
    s.patches_dir.mkdir(parents=True, exist_ok=True)
    s.backups_dir.mkdir(parents=True, exist_ok=True)
    return s


def _make_session(mode=AgentMode.ASK, summary="ok"):
    return SessionRecord(
        session_id="s1",
        created_at=datetime.now(timezone.utc),
        mode=mode,
        workspace_path="/tmp",
        prompt="p",
        user_input="hi",
        workspace_summary=WorkspaceSummary(
            root_path="/", total_files_scanned=0, included_files=0,
            skipped_files=0, top_extensions={}, sampled_files=[],
        ),
        ranked_files=[],
        selected_context=[],
        parsed_response=ParsedAnswer(summary=summary),
        raw_response=summary,
    )


# ---------------------------------------------------------------------------
# Workspace scanner
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_workspace_scanner_full(settings, tmp_path):
    scanner = WorkspaceScanner(settings)
    (tmp_path / "src").mkdir(exist_ok=True)
    (tmp_path / "src" / "main.py").write_text("print('hello')")
    results1, _ = await scanner.scan(tmp_path)
    assert len(results1) == 1
    results2, _ = await scanner.scan(tmp_path)
    assert results1 is results2


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_writers_all(settings):
    s_writer = SessionWriter(settings)
    session = _make_session()
    path = await s_writer.write(session)
    assert path.exists()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_comprehensive(settings, tmp_path):
    orchestrator = AgentOrchestrator(settings)
    orchestrator.client = AsyncMock()
    orchestrator.client.chat.return_value = MagicMock(
        content='{"summary": "ok"}', usage={}, status="ok",
        finish_reason=None, tool_calls=[], raw_response=None,
    )
    orchestrator.scanner = AsyncMock()
    orchestrator.scanner.scan.return_value = (
        [],
        WorkspaceSummary(
            root_path=str(tmp_path), total_files_scanned=0, included_files=0,
            skipped_files=0, top_extensions={}, sampled_files=[],
        ),
    )
    orchestrator.ranker = AsyncMock()
    orchestrator.ranker.rank.return_value = []
    orchestrator.reader = AsyncMock()
    orchestrator.reader.read_ranked_files.return_value = ([], [])

    # Substantive question — must NOT hit the greeting fast-path
    res = await orchestrator.run(AgentMode.ASK, tmp_path, "analyze the code structure")
    assert res.parsed_response.summary == "ok"


@pytest.mark.asyncio
async def test_orchestrator_greeting_fast_path(settings, tmp_path):
    """Short greetings should return without calling the LLM."""
    orchestrator = AgentOrchestrator(settings)
    orchestrator.client = AsyncMock()

    res = await orchestrator.run(AgentMode.ASK, tmp_path, "hi")
    orchestrator.client.chat.assert_not_called()
    assert res.parsed_response.summary  # has a canned response


@pytest.mark.asyncio
async def test_orchestrator_greeting_fast_path_not_triggered_for_long_input(settings, tmp_path):
    """Long messages must NOT hit the fast-path even if they contain a greeting word."""
    orchestrator = AgentOrchestrator(settings)
    orchestrator.client = AsyncMock()
    orchestrator.client.chat.return_value = MagicMock(
        content='{"summary": "ok"}', usage={}, status="ok",
        finish_reason=None, tool_calls=[], raw_response=None,
    )
    orchestrator.scanner = AsyncMock()
    orchestrator.scanner.scan.return_value = (
        [],
        WorkspaceSummary(
            root_path=str(tmp_path), total_files_scanned=0, included_files=0,
            skipped_files=0, top_extensions={}, sampled_files=[],
        ),
    )
    orchestrator.ranker = AsyncMock()
    orchestrator.ranker.rank.return_value = []
    orchestrator.reader = AsyncMock()
    orchestrator.reader.read_ranked_files.return_value = ([], [])

    long_input = "hi, can you please analyze the entire repository structure and tell me what each module does?"
    res = await orchestrator.run(AgentMode.ASK, tmp_path, long_input)
    # LLM must have been called
    orchestrator.client.chat.assert_called_once()


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_api_endpoints():
    os.environ.setdefault("NVIDIA_API_KEY", "test-key-for-health-check")
    from agent_core.server.api import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert "ok" in res.json()

        metrics_res = client.get("/metrics")
        assert metrics_res.status_code == 200


def test_api_health_degraded_without_key(monkeypatch):
    """Server must start and /health must respond even without an API key.

    We patch os.getenv directly so the .env file on disk cannot interfere.
    """
    import os as _os

    original_getenv = _os.getenv

    def patched_getenv(key, default=None):
        if key == "NVIDIA_API_KEY":
            return ""
        return original_getenv(key, default)

    monkeypatch.setattr(_os, "getenv", patched_getenv)

    from agent_core.server.api import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["ok"] is False


# ---------------------------------------------------------------------------
# OpenAI adapter utilities
# ---------------------------------------------------------------------------

def test_openai_adapter_utils():
    session = _make_session()
    resp = build_openai_response(session, "test-model")
    assert resp.choices[0].message.content == "ok"
    assert resp.model == "test-model"

    # extract_last_user_text / extract_user_message (alias)
    msg = OpenAIMessage(role="user", content="hi")
    assert extract_last_user_text([msg]) == "hi"
    assert extract_user_message([msg]) == "hi"

    # Returns last user message, not last message overall
    msgs = [
        OpenAIMessage(role="user", content="first"),
        OpenAIMessage(role="assistant", content="response"),
        OpenAIMessage(role="user", content="second"),
    ]
    assert extract_last_user_text(msgs) == "second"

    assert validate_openai_mode("ask") == AgentMode.ASK.value


def test_normalize_openai_messages():
    msgs = [
        OpenAIMessage(role="system", content="rules"),
        OpenAIMessage(role="developer", content="dev rules"),  # → system
        OpenAIMessage(role="user", content="hello"),
        OpenAIMessage(role="assistant", content="hi there"),
        OpenAIMessage(role="unknown_role", content="ignored"),  # skipped
        OpenAIMessage(role="user", content=None),               # skipped (no content)
    ]
    normalized = normalize_openai_messages(msgs)
    roles = [m.role for m in normalized]
    assert roles == ["system", "system", "user", "assistant"]
    assert normalized[1].content == "dev rules"


def test_normalize_openai_messages_multipart():
    """Multipart content (list of dicts) should be flattened to text."""
    msg = OpenAIMessage(
        role="user",
        content=[{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}],
    )
    normalized = normalize_openai_messages([msg])
    assert len(normalized) == 1
    assert normalized[0].content == "hello\nworld"


def test_extract_preferred_files_heuristic():
    """Heuristic path detection must require a slash and a known extension."""
    request = OpenAIChatCompletionRequest(
        model="laz-agent",
        messages=[
            OpenAIMessage(role="user", content="look at agent_core/server/api.py and README.md"),
        ],
        workspace=".",
    )
    preferred = extract_preferred_files(request)
    # agent_core/server/api.py has a slash + .py → should be included
    assert any("api.py" in p for p in preferred)
    # README.md has no slash → should NOT be included by heuristic
    assert not any(p == "README.md" for p in preferred)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def test_response_parser():
    parser = ResponseParser()
    res = parser.parse('{"summary": "test"}')
    assert res.summary == "test"
    res = parser.parse('Text before\n```json\n{"summary": "test2"}\n```\nText after')
    assert res.summary == "test2"
    res = parser.parse("not json")
    assert res.summary == "not json"
    assert res.parse_strategy == "text"


# ---------------------------------------------------------------------------
# NVIDIA provider — error handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_nvidia_provider_errors(settings):
    provider = NvidiaProvider(settings)
    provider.max_attempts = 1

    with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("timeout")):
        with pytest.raises(NvidiaTimeoutError):
            await provider.chat([ChatMessage(role="user", content="hi")])

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "error", request=MagicMock(), response=mock_resp
    )
    with patch("httpx.AsyncClient.post", return_value=mock_resp):
        with pytest.raises(NvidiaBackendError):
            await provider.chat([ChatMessage(role="user", content="hi")])


# ---------------------------------------------------------------------------
# NVIDIA provider — streaming chunk parsing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_nvidia_stream_parses_chunks(settings):
    """chat_stream() must yield structured dicts, not raw strings."""
    provider = NvidiaProvider(settings)

    async def fake_lines():
        yield 'data: {"choices":[{"delta":{"content":"hel"},"finish_reason":null}]}'
        yield 'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}'
        yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'
        yield "data: [DONE]"

    class FakeStreamResponse:
        def raise_for_status(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        def aiter_lines(self):
            return fake_lines()

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        def stream(self, *args, **kwargs):
            return FakeStreamResponse()

    with patch("httpx.AsyncClient", return_value=FakeClient()):
        chunks = []
        async for chunk in provider.chat_stream([ChatMessage(role="user", content="hi")]):
            chunks.append(chunk)

    assert chunks[0]["content"] == "hel"
    assert chunks[1]["content"] == "lo"
    assert chunks[-1]["finish_reason"] == "stop"
    # All chunks must be dicts
    for c in chunks:
        assert isinstance(c, dict)
        assert "content" in c
        assert "tool_calls" in c
        assert "finish_reason" in c
