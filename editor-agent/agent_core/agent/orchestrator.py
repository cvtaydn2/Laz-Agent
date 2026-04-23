from __future__ import annotations

from pathlib import Path

from agent_core.agent.apply_mode import ApplyModePolicy
from agent_core.agent.patch_preview import PatchPreviewPolicy
from agent_core.agent.planner import AgentPlanner
from agent_core.agent.response_parser import ResponseParser
from agent_core.agent.suggester import SuggestionPolicy
from agent_core.config import Settings
from agent_core.logger import configure_logger
from agent_core.models import (
    ApplyLogRecord,
    AgentMode,
    ChatMessage,
    ParsedAnswer,
    PatchProposal,
    SessionRecord,
    build_session_id,
    utc_now,
)
from agent_core.nvidia_client import NvidiaClient
from agent_core.output.writers import ApplyLogWriter, PatchProposalWriter, SessionWriter
from agent_core.prompts import build_prompt
from agent_core.tools.apply_tools import ApplyEngine
from agent_core.tools.patch_tools import build_patch_proposal
from agent_core.workspace.ranker import WorkspaceRanker
from agent_core.workspace.reader import WorkspaceReader
from agent_core.workspace.scanner import WorkspaceScanner


class AgentOrchestrator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.scanner = WorkspaceScanner(settings)
        self.ranker = WorkspaceRanker(settings)
        self.reader = WorkspaceReader(settings)
        self.client = NvidiaClient(settings)
        self.planner = AgentPlanner()
        self.response_parser = ResponseParser()
        self.suggestion_policy = SuggestionPolicy()
        self.patch_preview_policy = PatchPreviewPolicy()
        self.apply_policy = ApplyModePolicy()
        self.session_writer = SessionWriter(settings)
        self.patch_writer = PatchProposalWriter(settings)
        self.apply_log_writer = ApplyLogWriter(settings)
        self.apply_engine = ApplyEngine(settings)
        self.logger = configure_logger(settings.logs_dir / "editor-agent.log")

    def run(
        self,
        mode: AgentMode,
        workspace_path: Path,
        user_input: str | None,
        confirm: bool = False,
        temperature_override: float | None = None,
        max_tokens_override: int | None = None,
    ) -> SessionRecord:
        self.logger.info("Starting run: mode=%s workspace=%s", mode.value, workspace_path)

        scan_results, workspace_summary = self.scanner.scan(workspace_path)
        ranked_files = self.ranker.rank(workspace_path, scan_results, mode, user_input)
        selected_context, reader_notes = self.reader.read_ranked_files(workspace_path, ranked_files)
        workspace_summary.notes.extend(reader_notes)

        prompt_bundle = build_prompt(mode, workspace_summary, selected_context, user_input)
        messages = [
            ChatMessage(role="system", content=prompt_bundle.system_prompt),
            ChatMessage(role="user", content=prompt_bundle.user_prompt),
        ]
        model_response = self.client.chat(
            messages,
            temperature_override=temperature_override,
            max_tokens_override=max_tokens_override,
        )
        try:
            parsed = self.response_parser.parse(model_response.content)
        except Exception:
            self.logger.exception("Response parsing failed; falling back to raw model text.")
            parsed = ParsedAnswer(
                summary=model_response.content.strip() or "No model content returned.",
                raw_text=model_response.content,
                parse_strategy="raw_text",
            )
        session_id = build_session_id(mode)
        created_at = utc_now()

        if mode == AgentMode.SUGGEST:
            parsed = self.suggestion_policy.apply(parsed)
        if mode == AgentMode.PATCH_PREVIEW:
            parsed = self.patch_preview_policy.apply(parsed)
        if mode == AgentMode.APPLY:
            parsed = self.apply_policy.apply(parsed)

        patch_proposal: PatchProposal | None = None
        patch_proposal_path: str | None = None
        apply_log: ApplyLogRecord | None = None
        apply_log_path: str | None = None
        if mode in {AgentMode.PATCH_PREVIEW, AgentMode.APPLY}:
            patch_proposal = build_patch_proposal(
                session_id=session_id,
                created_at=created_at,
                workspace_path=str(workspace_path.resolve()),
                request=user_input,
                parsed=parsed,
            )
            patch_proposal_path = str(self.patch_writer.write(patch_proposal))
        if mode == AgentMode.APPLY and confirm:
            apply_log = self.apply_engine.apply(
                workspace_path=workspace_path,
                session_id=session_id,
                created_at=created_at,
                request=user_input,
                operations=parsed.file_operations,
            )
            apply_log_path = str(self.apply_log_writer.write(apply_log))

        session = SessionRecord(
            session_id=session_id,
            created_at=created_at,
            mode=mode,
            workspace_path=str(workspace_path.resolve()),
            prompt=prompt_bundle.user_prompt,
            user_input=user_input,
            workspace_summary=workspace_summary,
            ranked_files=ranked_files,
            selected_context=selected_context,
            raw_response=model_response.content,
            parsed_response=parsed,
            patch_proposal_path=patch_proposal_path,
            apply_log_path=apply_log_path,
            confirmed=confirm,
        )
        self.session_writer.write(session)
        self.logger.info(
            "Completed run: mode=%s workspace=%s session=%s",
            mode.value,
            workspace_path,
            session.session_id,
        )
        return session
