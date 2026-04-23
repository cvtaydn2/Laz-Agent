import pytest
import asyncio
import json
import os
import time
import httpx
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from agent_core.config import Settings
from agent_core.workspace.scanner import WorkspaceScanner
from agent_core.output.writers import SessionWriter, PatchProposalWriter, ApplyLogWriter
from agent_core.llm import get_llm_provider
from agent_core.llm.nvidia import NvidiaProvider, NvidiaTimeoutError, NvidiaBackendError
from agent_core.models import (
    SessionRecord, AgentMode, FileScanResult, WorkspaceSummary, 
    ChatMessage, ParsedAnswer, PatchProposal, ApplyLogRecord
)
from agent_core.agent.orchestrator import AgentOrchestrator
from agent_core.workspace.filters import is_allowed_file, is_ignored_directory
from agent_core.server.openai_adapter import build_openai_response, extract_user_message, validate_openai_mode
from agent_core.server.openai_schemas import OpenAIMessage
from agent_core.agent.response_parser import ResponseParser

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

@pytest.mark.asyncio
async def test_workspace_scanner_full(settings, tmp_path):
    scanner = WorkspaceScanner(settings)
    (tmp_path / "src").mkdir(exist_ok=True)
    (tmp_path / "src" / "main.py").write_text("print('hello')")
    results1, summary1 = await scanner.scan(tmp_path)
    assert len(results1) == 1
    results2, _ = await scanner.scan(tmp_path)
    assert results1 is results2

@pytest.mark.asyncio
async def test_writers_all(settings):
    s_writer = SessionWriter(settings)
    summary = WorkspaceSummary(
        root_path="/tmp", total_files_scanned=0, included_files=0, skipped_files=0, top_extensions={}, sampled_files=[]
    )
    parsed = ParsedAnswer(summary="ok")
    session = SessionRecord(
        session_id="s1", created_at=datetime.now(timezone.utc), mode=AgentMode.ASK,
        workspace_path="/tmp", prompt="p", user_input="hi",
        workspace_summary=summary, ranked_files=[], selected_context=[], parsed_response=parsed,
        raw_response="ok"
    )
    path = await s_writer.write(session)
    assert path.exists()

@pytest.mark.asyncio
async def test_orchestrator_comprehensive(settings, tmp_path):
    orchestrator = AgentOrchestrator(settings)
    orchestrator.client = AsyncMock()
    orchestrator.client.chat.return_value = MagicMock(content="{\"summary\": \"ok\"}", usage={}, status="ok")
    orchestrator.scanner = AsyncMock()
    orchestrator.scanner.scan.return_value = ([], WorkspaceSummary(
        root_path=str(tmp_path), total_files_scanned=0, included_files=0, skipped_files=0, top_extensions={}, sampled_files=[]
    ))
    orchestrator.ranker = AsyncMock()
    orchestrator.ranker.rank.return_value = []
    orchestrator.reader = AsyncMock()
    orchestrator.reader.read_ranked_files.return_value = ([], [])
    res = await orchestrator.run(AgentMode.ASK, tmp_path, "hi")
    assert res.parsed_response.summary == "ok"

@pytest.mark.asyncio
async def test_api_endpoints(settings):
    from fastapi.testclient import TestClient
    from agent_core.server.api import app
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["ok"] is True
        
        metrics_res = client.get("/metrics")
        assert metrics_res.status_code == 200

def test_openai_adapter_utils():
    session = SessionRecord(
        session_id="s1", created_at=datetime.now(timezone.utc), mode=AgentMode.ASK,
        workspace_path="/tmp", prompt="p", raw_response="r", user_input="hi",
        workspace_summary=WorkspaceSummary(root_path="/", total_files_scanned=0, included_files=0, skipped_files=0, top_extensions={}, sampled_files=[]),
        ranked_files=[], selected_context=[], parsed_response=ParsedAnswer(summary="ok")
    )
    resp = build_openai_response(session, "test-model")
    assert resp.choices[0].message.content == "ok"
    assert resp.model == "test-model"
    
    msg = OpenAIMessage(role="user", content="hi")
    assert extract_user_message([msg]) == "hi"
    assert validate_openai_mode("ask") == AgentMode.ASK

def test_response_parser():
    parser = ResponseParser()
    res = parser.parse("{\"summary\": \"test\"}")
    assert res.summary == "test"
    res = parser.parse("Text before\n```json\n{\"summary\": \"test2\"}\n```\nText after")
    assert res.summary == "test2"
    res = parser.parse("not json")
    assert res.summary == "not json"
    assert res.parse_strategy == "text"

@pytest.mark.asyncio
async def test_nvidia_provider_errors(settings):
    provider = NvidiaProvider(settings)
    provider.max_attempts = 1
    
    with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("timeout")):
        with pytest.raises(NvidiaTimeoutError):
            await provider.chat([ChatMessage(role="user", content="hi")])
    
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError("error", request=MagicMock(), response=mock_resp)
    with patch("httpx.AsyncClient.post", return_value=mock_resp):
        with pytest.raises(NvidiaBackendError):
            await provider.chat([ChatMessage(role="user", content="hi")])
