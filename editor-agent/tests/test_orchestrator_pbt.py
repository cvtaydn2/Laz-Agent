"""Property-based tests for the orchestrator refactor.

Tests the five correctness properties defined in design.md:
  1. Settings-independent operation
  2. async_load error tolerance
  3. Policy injection
  4. plan_label unknown-mode tolerance
  5. Public API signature preservation
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import given, settings as h_settings, HealthCheck
from hypothesis import strategies as st

from agent_core.agent.orchestrator import AgentOrchestrator, OrchestratorDependencies
from agent_core.agent.planner import AgentPlanner
from agent_core.knowledge import KnowledgeBase
from agent_core.models import AgentMode, ParsedAnswer, WorkspaceSummary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_deps(settings=None) -> OrchestratorDependencies:
    """Build a fully-mocked OrchestratorDependencies for testing."""
    summary = WorkspaceSummary(
        root_path="/tmp", total_files_scanned=0, included_files=0,
        skipped_files=0, top_extensions={}, sampled_files=[],
    )
    scanner = AsyncMock()
    scanner.scan.return_value = ([], summary)
    ranker = AsyncMock()
    ranker.rank.return_value = []
    reader = AsyncMock()
    reader.read_ranked_files.return_value = ([], [])

    llm = AsyncMock()
    llm.chat.return_value = MagicMock(
        content='{"summary": "ok"}', usage={}, status="ok",
        finish_reason=None, tool_calls=[], raw_response=None,
    )

    session_writer = AsyncMock()
    session_writer.write.return_value = Path("/tmp/session.json")

    parsed = ParsedAnswer(summary="ok")

    return OrchestratorDependencies(
        scanner=scanner,
        ranker=ranker,
        reader=reader,
        llm_provider=llm,
        planner=AgentPlanner(),
        response_parser=MagicMock(parse=MagicMock(return_value=parsed)),
        suggestion_policy=MagicMock(apply=MagicMock(side_effect=lambda p: p)),
        patch_preview_policy=MagicMock(apply=MagicMock(side_effect=lambda p: p)),
        apply_policy=MagicMock(apply=MagicMock(side_effect=lambda p: p)),
        review_verifier=MagicMock(verify=MagicMock(side_effect=lambda **kw: kw.get("parsed"))),
        session_writer=session_writer,
        patch_writer=AsyncMock(),
        apply_log_writer=AsyncMock(),
        apply_engine=MagicMock(),
        knowledge_base=KnowledgeBase(),
        settings=settings,
    )


# ---------------------------------------------------------------------------
# Property 1: Settings-independent operation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_property1_settings_independent(tmp_path):
    """Property 1: Orchestrator with mock deps must not access Settings."""
    deps = _make_mock_deps(settings=None)
    orchestrator = AgentOrchestrator(deps=deps)

    # settings shortcut must be None — no Settings object was provided
    assert orchestrator.settings is None

    # run() must complete without touching a Settings object
    result = await orchestrator.run(AgentMode.ASK, tmp_path, "analyze the code")
    assert result is not None


# ---------------------------------------------------------------------------
# Property 2: async_load error tolerance
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@given(st.text())
@h_settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_property2_async_load_invalid_json(invalid_content: str):
    """Property 2: async_load must return empty KnowledgeBase for invalid JSON."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        f.write(invalid_content)
        tmp = Path(f.name)
    try:
        kb = await KnowledgeBase.async_load(tmp)
        assert isinstance(kb, KnowledgeBase)
    finally:
        tmp.unlink(missing_ok=True)


@pytest.mark.asyncio
@given(st.text(min_size=1).filter(lambda s: "\x00" not in s))
@h_settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_property2_async_load_nonexistent_path(path_str: str):
    """Property 2: async_load must return empty KnowledgeBase for missing files."""
    # Construct a path that almost certainly does not exist
    path = Path("/nonexistent_hypothesis_test") / path_str
    kb = await KnowledgeBase.async_load(path)
    assert isinstance(kb, KnowledgeBase)
    assert len(kb.entries) == 0


# ---------------------------------------------------------------------------
# Property 3: Policy injection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_property3_policy_injection(tmp_path):
    """Property 3: Injected stub policy must be called by orchestrator in SUGGEST mode."""
    call_log: list[ParsedAnswer] = []

    def stub_apply(parsed: ParsedAnswer) -> ParsedAnswer:
        call_log.append(parsed)
        return parsed

    deps = _make_mock_deps()
    deps.suggestion_policy = MagicMock(apply=MagicMock(side_effect=stub_apply))

    orchestrator = AgentOrchestrator(deps=deps)
    await orchestrator.run(AgentMode.SUGGEST, tmp_path, "suggest improvements")

    assert len(call_log) >= 1, "Stub policy.apply() was never called"


# ---------------------------------------------------------------------------
# Property 4: plan_label unknown-mode tolerance
# ---------------------------------------------------------------------------

@given(st.text())
@h_settings(max_examples=100)
def test_property4_plan_label_unknown_mode(value: str):
    """Property 4: plan_label must never raise for any input string."""
    # Wrap in a try/except to make the property explicit
    try:
        result = AgentPlanner.plan_label(value)  # type: ignore[arg-type]
    except Exception as exc:
        raise AssertionError(
            f"plan_label raised {type(exc).__name__} for input {value!r}"
        ) from exc
    # For non-AgentMode values the result should be "Unknown"
    valid_labels = {"Analyze", "Ask", "Suggest", "Patch Preview", "Apply",
                    "Review", "Compare", "Bug Hunt", "Fix", "Unknown"}
    assert result in valid_labels


# ---------------------------------------------------------------------------
# Property 5: Public API signature preservation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@given(st.sampled_from(list(AgentMode)), st.text(min_size=1, max_size=200))
@h_settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_property5_run_returns_session_record(mode: AgentMode, user_input: str):
    """Property 5: run() must always return a SessionRecord."""
    import tempfile
    from agent_core.models import SessionRecord

    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        deps = _make_mock_deps()
        orchestrator = AgentOrchestrator(deps=deps)
        result = await orchestrator.run(mode, workspace, user_input)
        assert isinstance(result, SessionRecord)


@pytest.mark.asyncio
async def test_property5_stream_run_is_async_iterable(tmp_path):
    """Property 5: stream_run() must return an async iterable of dicts."""
    deps = _make_mock_deps()

    async def fake_stream(*args, **kwargs):
        yield {"content": "hello", "tool_calls": [], "finish_reason": None, "raw": {}}
        yield {"content": None, "tool_calls": [], "finish_reason": "stop", "raw": {}}

    deps.llm_provider.chat_stream = fake_stream

    orchestrator = AgentOrchestrator(deps=deps)
    chunks = []
    async for chunk in orchestrator.stream_run(AgentMode.ASK, tmp_path, "test"):
        chunks.append(chunk)
        assert isinstance(chunk, dict)

    assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# Backward-compatibility: AgentOrchestrator(settings) must not raise
# ---------------------------------------------------------------------------

def test_legacy_init_with_settings(tmp_path):
    """AgentOrchestrator(settings) must work without TypeError (Requirement 6.4)."""
    from agent_core.config import Settings

    s = Settings(
        nvidia_api_key="test-key",
        session_dir=tmp_path / "sessions",
        logs_dir=tmp_path / "logs",
        patches_dir=tmp_path / "patches",
        backups_dir=tmp_path / "backups",
    )
    for d in [s.session_dir, s.logs_dir, s.patches_dir, s.backups_dir]:
        d.mkdir(parents=True, exist_ok=True)

    orchestrator = AgentOrchestrator(s)
    assert orchestrator.settings is s
    assert orchestrator.scanner is not None


# ---------------------------------------------------------------------------
# OrchestratorDependencies: from agent_core.agent import works cleanly
# ---------------------------------------------------------------------------

def test_public_import():
    """Requirement 5.3: import must not raise CircularImportError."""
    from agent_core.agent import AgentOrchestrator, OrchestratorDependencies  # noqa: F401
    assert AgentOrchestrator is not None
    assert OrchestratorDependencies is not None
