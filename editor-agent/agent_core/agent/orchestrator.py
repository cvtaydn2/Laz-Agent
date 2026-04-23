from __future__ import annotations

from pathlib import Path

from agent_core.agent.review_verifier import ReviewVerifier
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
from agent_core.nvidia_client import NvidiaBackendError, NvidiaClient, NvidiaTimeoutError
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
        self.review_verifier = ReviewVerifier(settings)
        self.session_writer = SessionWriter(settings)
        self.patch_writer = PatchProposalWriter(settings)
        self.apply_log_writer = ApplyLogWriter(settings)
        self.apply_engine = ApplyEngine(settings)
        self.logger = configure_logger(settings.logs_dir / "editor-agent.log")

    def run(
        self,
        mode: AgentMode | str,
        workspace_path: Path,
        user_input: str | None,
        confirm: bool = False,
        temperature_override: float | None = None,
        max_tokens_override: int | None = None,
        changed_files: list[str] | None = None,
        diff_text: str | None = None,
    ) -> SessionRecord:
        mode = self._normalize_mode(mode)
        self.logger.info("Starting run: mode=%s workspace=%s", mode.value, workspace_path)
        use_workspace = self._should_use_workspace(
            mode=mode,
            user_input=user_input,
            changed_files=changed_files,
            diff_text=diff_text,
        )
        print("DEBUG normalized mode:", mode)
        print("DEBUG should_use_workspace:", use_workspace)
        print("DEBUG user_input:", user_input)
        print("DEBUG changed_files:", changed_files)
        print("DEBUG diff_text_present:", bool(diff_text))
        if use_workspace:
            scan_results, workspace_summary = self.scanner.scan(workspace_path)
            ranked_files = self.ranker.rank(
                workspace_path,
                scan_results,
                mode,
                user_input,
                preferred_files=changed_files,
            )
            selected_context, reader_notes = self.reader.read_ranked_files(workspace_path, ranked_files)
            workspace_summary.notes.extend(reader_notes)
            workspace_summary.preferred_files = changed_files or []
        else:
            ranked_files = []
            selected_context = []
            workspace_summary = self._build_empty_workspace_summary(workspace_path, changed_files)
        context_char_count = sum(len(item.content) for item in selected_context)
        self.logger.info(
            "Workspace context selected: workspace=%s mode=%s backend_model=%s files=%s context_chars=%s preferred_files=%s workspace_used=%s",
            workspace_path,
            mode.value,
            self.settings.nvidia_model,
            len(selected_context),
            context_char_count,
            len(changed_files or []),
            use_workspace,
        )

        prompt_bundle = build_prompt(
            mode,
            workspace_summary,
            selected_context,
            user_input,
            changed_files=changed_files,
            diff_text=diff_text,
        )
        messages = [
            ChatMessage(role="system", content=prompt_bundle.system_prompt),
            ChatMessage(role="user", content=prompt_bundle.user_prompt),
        ]
        print("DEBUG system_prompt_len:", len(prompt_bundle.system_prompt))
        print("DEBUG user_prompt_len:", len(prompt_bundle.user_prompt))
        print("DEBUG max_completion_tokens:", self.settings.max_completion_tokens)
        print("DEBUG timeout_seconds:", self.settings.timeout_seconds)
        try:
            model_response = self.client.chat(
                messages,
                temperature_override=temperature_override,
                max_tokens_override=max_tokens_override,
            )
        except (NvidiaTimeoutError, NvidiaBackendError):
            self.logger.exception(
                "Backend inference failed: workspace=%s mode=%s backend_model=%s",
                workspace_path,
                mode.value,
                self.settings.nvidia_model,
            )
            raise
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
        if mode == AgentMode.REVIEW:
            parsed = self.review_verifier.verify(
                workspace_path=workspace_path,
                parsed=parsed,
                selected_context=selected_context,
            )

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
            "Completed run: mode=%s workspace=%s session=%s backend_model=%s files=%s context_chars=%s status=success",
            mode.value,
            workspace_path,
            session.session_id,
            self.settings.nvidia_model,
            len(selected_context),
            context_char_count,
        )
        return session

    def _normalize_mode(self, mode: AgentMode | str) -> AgentMode:
        if isinstance(mode, AgentMode):
            return mode
        if isinstance(mode, str):
            normalized = mode.strip().lower().replace("-", "_")
            return AgentMode(normalized)
        raise ValueError("Mode must be an AgentMode or a valid mode string.")

    def _should_use_workspace(
        self,
        *,
        mode: AgentMode,
        user_input: str | None,
        changed_files: list[str] | None,
        diff_text: str | None,
    ) -> bool:
        if changed_files or diff_text:
            return True
        if mode in {AgentMode.ANALYZE, AgentMode.SUGGEST, AgentMode.PATCH_PREVIEW, AgentMode.APPLY, AgentMode.REVIEW}:
            return True
        text = (user_input or "").strip().lower()
        if not text:
            return False
        repo_keywords = {
            "project",
            "repo",
            "repository",
            "workspace",
            "file",
            "files",
            "class",
            "function",
            "module",
            "bug",
            "error",
            "readme",
            "requirements",
            "config",
            "setting",
            "code",
            "review",
            "implement",
            "neden",
            "nasıl",
            "çalışıyor",
        }
        return any(keyword in text for keyword in repo_keywords)

    def _build_empty_workspace_summary(
        self,
        workspace_path: Path,
        changed_files: list[str] | None,
    ):
        from agent_core.models import WorkspaceSummary

        return WorkspaceSummary(
            root_path=str(workspace_path.resolve()),
            total_files_scanned=0,
            included_files=0,
            skipped_files=0,
            top_extensions={},
            sampled_files=[],
            notes=["Workspace context intentionally skipped for a general request."],
            preferred_files=changed_files or [],
        )
