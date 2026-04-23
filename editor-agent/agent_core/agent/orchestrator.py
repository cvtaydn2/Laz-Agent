from __future__ import annotations

import re
import json
import asyncio
from textwrap import dedent
from typing import Any, Iterable, AsyncIterable
from pathlib import Path

from agent_core.agent.review_verifier import ReviewVerifier
from agent_core.agent.apply_mode import ApplyModePolicy
from agent_core.agent.patch_preview import PatchPreviewPolicy
from agent_core.agent.planner import AgentPlanner
from agent_core.agent.response_parser import ResponseParser
from agent_core.agent.suggester import SuggestionPolicy
from agent_core.knowledge import KnowledgeBase
from agent_core.config import Settings
from agent_core.llm import get_llm_provider
from agent_core.logger import configure_logger
from agent_core.models import (
    ApplyLogRecord,
    AgentMode,
    ChatMessage,
    ModelResponse,
    ParsedAnswer,
    PatchProposal,
    SessionRecord,
    build_session_id,
    utc_now,
    ComparisonResult,
)
from agent_core.llm.nvidia import NvidiaBackendError, NvidiaTimeoutError
from agent_core.output.writers import ApplyLogWriter, PatchProposalWriter, SessionWriter
from agent_core.prompts import build_prompt
from agent_core.tools.apply_tools import ApplyEngine
from agent_core.tools.patch_tools import build_patch_proposal
from agent_core.workspace.ranker import WorkspaceRanker
from agent_core.workspace.reader import WorkspaceReader
from agent_core.workspace.scanner import WorkspaceScanner
from agent_core.tools.command_tools import execute_command


class AgentOrchestrator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.scanner = WorkspaceScanner(settings)
        self.ranker = WorkspaceRanker(settings)
        self.reader = WorkspaceReader(settings)
        self.client = get_llm_provider(settings)
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
        self.knowledge_base = KnowledgeBase.load(settings.state_dir / "knowledge_base.json")
        self.logger = configure_logger(settings.logs_dir / "editor-agent.log")
        self.logger.info("Orchestrator initialized with knowledge base.")

    async def run(
        self,
        mode: AgentMode | str,
        workspace_path: Path,
        user_input: str | None,
        confirm: bool = False,
        temperature_override: float | None = None,
        max_tokens_override: int | None = None,
        changed_files: list[str] | None = None,
        diff_text: str | None = None,
        preferred_files: list[str] | None = None,
    ) -> SessionRecord:
        mode = self._normalize_mode(mode)
        force_workspace_use = False

        # Dynamic workspace override: Check if user_input contains an absolute path
        if user_input:
            # Match Windows paths (C:\...) or Linux absolute paths (/...)
            path_match = re.search(r'([a-zA-Z]:\\[^ :*?"<>|\r\n]+|/[^ :*?"<>|\r\n]+)', user_input)
            if path_match:
                detected_path = Path(path_match.group(1).strip()).resolve()
                if detected_path.exists() and detected_path.is_dir():
                    self.logger.info("Dynamic workspace override detected: %s", detected_path)
                    workspace_path = detected_path
                    # Force workspace scanning when a specific path is provided
                    force_workspace_use = True

        if workspace_path:
            force_workspace_use = True
        self.logger.info("Starting run: mode=%s workspace=%s", mode.value, workspace_path)
        trivial_response = self._build_trivial_ask_response(mode, workspace_path, user_input)
        if trivial_response is not None:
            await self.session_writer.write(trivial_response)
            self.logger.info(
                "Completed run: mode=%s workspace=%s session=%s backend_model=%s files=0 context_chars=0 status=local_fast_path",
                mode.value,
                workspace_path,
                trivial_response.session_id,
                self.settings.nvidia_model,
            )
            return trivial_response

        use_workspace = force_workspace_use or self._should_use_workspace(
            mode=mode,
            user_input=user_input,
            changed_files=changed_files,
            diff_text=diff_text,
        )

        if use_workspace:
            self.logger.info("Scanning workspace: %s", workspace_path)
            scan_results, workspace_summary = await self.scanner.scan(workspace_path)
            self.logger.info("Ranking files (mode=%s)...", mode.value)
            ranked_files = await self.ranker.rank(
                workspace_path,
                scan_results,
                mode,
                user_input,
                preferred_files=preferred_files or changed_files,
            )
            self.logger.info("Reading selected files...")
            selected_context, reader_notes = await self.reader.read_ranked_files(workspace_path, ranked_files)
            workspace_summary.notes.extend(reader_notes)
            workspace_summary.preferred_files = preferred_files or changed_files or []
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
            len(preferred_files or changed_files or []),
            use_workspace,
        )

        # Pre-flight diagnostic check
        diagnostic_notes = ""
        if mode in {AgentMode.FIX, AgentMode.BUG_HUNT, AgentMode.ANALYZE}:
            diag = await self._run_diagnostic(workspace_path)
            if diag:
                diagnostic_notes = f"\n### CURRENT WORKSPACE DIAGNOSTIC:\n{diag}\n"
                self.logger.info("Diagnostic check found issues; attaching to prompt.")

        # Query knowledge base for past lessons
        past_lessons = []
        if user_input:
            lessons = self.knowledge_base.query(user_input)
            past_lessons = [f"{l.pattern}: {l.solution_summary}" for l in lessons[:3]]

        prompt_bundle = build_prompt(
            mode,
            workspace_summary,
            selected_context,
            user_input,
            changed_files=changed_files,
            diff_text=diff_text,
            past_lessons=past_lessons,
        )
        if diagnostic_notes:
            prompt_bundle.user_prompt += diagnostic_notes
        messages = [
            ChatMessage(role="system", content=prompt_bundle.system_prompt),
            ChatMessage(role="user", content=prompt_bundle.user_prompt),
        ]
        self.logger.debug("System prompt length: %d", len(prompt_bundle.system_prompt))
        self.logger.debug("User prompt length: %d", len(prompt_bundle.user_prompt))
        model_response: ModelResponse | None = None
        parsed: ParsedAnswer | None = None

        if mode == AgentMode.COMPARE:
            self.logger.info("Entering comparison mode logic...")
            comparison = await self._run_comparison(
                messages=messages,
                temperature_override=temperature_override,
                max_tokens_override=max_tokens_override,
                workspace_path=workspace_path
            )
            parsed = comparison.final_answer
            model_response = ModelResponse(content=comparison.judge_thought, status="ok", usage={})
        else:
            comparison = None
            try:
                self.logger.info("Sending request to LLM (model=%s)...", self.settings.nvidia_model)
                model_response = await self.client.chat(
                    messages,
                    temperature_override=temperature_override,
                    max_tokens_override=max_tokens_override,
                )
                self.logger.info("Received LLM response.")
            except (NvidiaTimeoutError, NvidiaBackendError):
                self.logger.exception(
                    "Backend inference failed: workspace=%s mode=%s backend_model=%s",
                    workspace_path,
                    mode.value,
                    self.settings.nvidia_model,
                )
                raise
        if not parsed:
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
            patch_proposal_path = str(await self.patch_writer.write(patch_proposal))
        if mode == AgentMode.APPLY and confirm:
            apply_log = self.apply_engine.apply(
                workspace_path=workspace_path,
                session_id=session_id,
                created_at=created_at,
                request=user_input,
                operations=parsed.file_operations,
            )
            
            # New: Execute commands if provided and confirmed
            if parsed.command_operations:
                self.logger.info("Executing %d proposed commands", len(parsed.command_operations))
                from agent_core.models import CommandExecutionRecord
                for op in parsed.command_operations:
                    res = await execute_command(op.command, cwd=str(workspace_path.resolve()))
                    apply_log.commands_executed.append(
                        CommandExecutionRecord(
                            command=op.command,
                            stdout=res.stdout,
                            stderr=res.stderr,
                            returncode=res.returncode,
                            success=(res.returncode == 0)
                        )
                    )
            
            apply_log_path = str(await self.apply_log_writer.write(apply_log))
            
            # Record in knowledge base for self-learning
            if apply_log.success:
                self.knowledge_base.learn(
                    error_pattern=user_input or "automated_fix",
                    solution=parsed.summary,
                    files=parsed.affected_files
                )

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
            comparison=comparison,
        )
        await self.session_writer.write(session)
        self.knowledge_base.save(self.settings.state_dir / "knowledge_base.json")
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

    async def rollback(self, session_id: str) -> bool:
        """
        Reverts the changes made in a specific session.
        """
        # 1. Load the session
        session = await self.session_writer.read(session_id)
        if not session or not session.apply_log_path:
            self.logger.error("No apply log found for session %s", session_id)
            return False
        
        # 2. Load the apply log
        log_path = Path(session.apply_log_path)
        if not log_path.exists():
            return False
            
        with open(log_path, "r", encoding="utf-8") as f:
            from agent_core.models import ApplyLogRecord
            log_data = json.load(f)
            log = ApplyLogRecord(**log_data)
        
        # 3. Perform rollback
        success = self.apply_engine.rollback(Path(session.workspace_path), log)
        if success:
            self.logger.info("Rollback successful for session %s", session_id)
        else:
            self.logger.error("Rollback failed for session %s", session_id)
        return success

    async def self_heal(
        self,
        workspace_path: Path,
        max_retries: int = 2
    ) -> list[SessionRecord]:
        """
        An autonomous loop that attempts to fix errors in the project by running tests
        and feeding errors back to the model.
        """
        sessions = []
        self.logger.info("Starting self-healing loop for workspace: %s", workspace_path)
        
        # 1. Run initial test command to find bugs
        test_command = "pytest" # Default for this project
        
        for attempt in range(max_retries):
            self.logger.info("Self-healing attempt %d/%d", attempt + 1, max_retries)
            res = await execute_command(test_command, cwd=str(workspace_path.resolve()))
            
            if res.returncode == 0:
                self.logger.info("Self-healing successful: All tests passed.")
                break
                
            # 2. Feed error back to model
            healing_request = (
                f"Kendi kendimi iyileştirme (self-healing) modundayım. "
                f"'{test_command}' komutu çalıştırıldı ve şu hata alındı:\n\n"
                f"STDOUT:\n{res.stdout}\n\nSTDERR:\n{res.stderr}\n\n"
                f"Lütfen bu hatayı analiz et, sorunlu dosyaları belirle ve düzeltme (apply) için patch öner."
            )
            
            session = await self.run(
                mode=AgentMode.APPLY,
                workspace_path=workspace_path,
                user_input=healing_request,
                confirm=True # Auto-apply in self-healing mode
            )
            sessions.append(session)
            
            if not session.parsed_response.file_operations:
                self.logger.warning("Model did not suggest any fixes. Breaking loop.")
                break
                
        return sessions

    async def stream_run(
        self,
        mode: AgentMode | str,
        workspace_path: Path,
        user_input: str | None,
        temperature_override: float | None = None,
        max_tokens_override: int | None = None,
        changed_files: list[str] | None = None,
        diff_text: str | None = None,
        preferred_files: list[str] | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> AsyncIterable[dict]:
        """Yield structured chunk dicts from the LLM stream.

        Each dict has the shape:
            {"content": str | None, "tool_calls": list, "finish_reason": str | None, "raw": dict}

        Trivial greeting responses are yielded as a single synthetic chunk.
        """
        mode = self._normalize_mode(mode)

        # Fast-path: trivial greeting — no LLM call needed
        trivial_response = self._build_trivial_ask_response(mode, workspace_path, user_input)
        if trivial_response is not None:
            text = trivial_response.parsed_response.summary or trivial_response.raw_response
            if text:
                yield {"content": text, "tool_calls": [], "finish_reason": None, "raw": {}}
            yield {"content": None, "tool_calls": [], "finish_reason": "stop", "raw": {}}
            return

        # Build workspace context + prompt (shared with run())
        prompt_bundle, selected_context, workspace_summary = await self._prepare_prompt_bundle(
            mode=mode,
            workspace_path=workspace_path,
            user_input=user_input,
            changed_files=changed_files,
            diff_text=diff_text,
            preferred_files=preferred_files,
        )

        messages = [
            ChatMessage(role="system", content=prompt_bundle.system_prompt),
            ChatMessage(role="user", content=prompt_bundle.user_prompt),
        ]

        async for chunk in self.client.chat_stream(
            messages,
            temperature_override=temperature_override,
            max_tokens_override=max_tokens_override,
            tools=tools,
            tool_choice=tool_choice,
        ):
            yield chunk

        self.logger.info(
            "Completed stream: mode=%s workspace=%s backend_model=%s",
            mode.value,
            workspace_path,
            self.settings.nvidia_model,
        )

    async def _prepare_prompt_bundle(
        self,
        *,
        mode: AgentMode,
        workspace_path: Path,
        user_input: str | None,
        changed_files: list[str] | None,
        diff_text: str | None,
        preferred_files: list[str] | None,
    ):
        """Shared workspace scan + prompt build used by both run() and stream_run()."""
        from agent_core.models import PromptBundle

        use_workspace = self._should_use_workspace(
            mode=mode,
            user_input=user_input,
            changed_files=changed_files,
            diff_text=diff_text,
        )

        if use_workspace:
            scan_results, workspace_summary = await self.scanner.scan(workspace_path)
            ranked_files = await self.ranker.rank(
                workspace_path,
                scan_results,
                mode,
                user_input,
                preferred_files=preferred_files or changed_files,
            )
            selected_context, reader_notes = await self.reader.read_ranked_files(
                workspace_path, ranked_files
            )
            workspace_summary.notes.extend(reader_notes)
            workspace_summary.preferred_files = preferred_files or changed_files or []
        else:
            selected_context = []
            workspace_summary = self._build_empty_workspace_summary(workspace_path, changed_files)

        past_lessons: list[str] = []
        if user_input:
            lessons = self.knowledge_base.query(user_input)
            past_lessons = [f"{l.pattern}: {l.solution_summary}" for l in lessons[:3]]

        prompt_bundle = build_prompt(
            mode,
            workspace_summary,
            selected_context,
            user_input,
            changed_files=changed_files,
            diff_text=diff_text,
            past_lessons=past_lessons,
        )
        return prompt_bundle, selected_context, workspace_summary

    def _normalize_mode(self, mode: AgentMode | str) -> AgentMode:
        if isinstance(mode, AgentMode):
            return mode
        if isinstance(mode, str):
            normalized = mode.strip().lower().replace("-", "_")
            return AgentMode(normalized)
        raise ValueError("Mode must be an AgentMode or a valid mode string.")

    def _build_trivial_ask_response(
        self,
        mode: AgentMode,
        workspace_path: Path,
        user_input: str | None,
        changed_files: list[str] | None = None,
        diff_text: str | None = None,
        preferred_files: list[str] | None = None,
    ) -> SessionRecord | None:
        # Only intercept plain ASK requests
        if not user_input or mode != AgentMode.ASK:
            return None

        # If there is any file/diff context, always go to the LLM
        if changed_files or diff_text or preferred_files:
            return None

        stripped = user_input.strip()

        # Long messages are never pure greetings
        if len(stripped) > 60:
            return None

        # Multi-line messages contain real content
        if "\n" in stripped:
            return None

        # Continue and other clients inject boilerplate — skip fast-path
        if "<important_rules>" in stripped.lower():
            return None

        # Check only the tail to ignore any leading boilerplate
        tail = stripped[-120:].lower()

        # Turkish and English greetings / small-talk that don't need LLM
        tr_greetings = {
            "selam", "merhaba", "nasılsın", "naber", "ne haber", "iyi misin",
            "kimsin", "ne yapabilirsin", "ordamısın", "ordasın", "orada mısın",
            "burada mısın", "uyandın mı", "hazır mısın", "çalışıyor musun",
        }
        en_greetings = {
            "hi", "hello", "hey", "who are you", "what can you do",
            "are you there", "you there", "you awake",
        }
        all_greetings = tr_greetings | en_greetings

        is_greeting = False
        is_turkish = False
        for g in all_greetings:
            if re.search(rf"\b{re.escape(g)}\b", tail):
                is_greeting = True
                if g in tr_greetings:
                    is_turkish = True
                break

        if not is_greeting:
            return None

        if is_turkish:
            response_text = (
                "Evet, buradayım! Ben Laz-Agent. "
                "Kod analizi yapabilir, hataları bulabilir veya yeni özellikler ekleyebilirim. "
                "Nasıl yardımcı olabilirim?"
            )
        else:
            response_text = (
                "Yes, I'm here! I am Laz-Agent, your architectural assistant. "
                "I can analyze your code, hunt for bugs, or suggest improvements. "
                "How can I help you today?"
            )

        return SessionRecord(
            session_id=build_session_id(mode),
            created_at=utc_now(),
            mode=mode,
            workspace_path=str(workspace_path),
            prompt=user_input,
            user_input=user_input,
            workspace_summary=self._build_empty_workspace_summary(workspace_path, None),
            ranked_files=[],
            selected_context=[],
            raw_response=response_text,
            parsed_response=ParsedAnswer(
                summary=response_text,
                raw_text=response_text,
                parse_strategy="raw_text",
            ),
        )

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
        if mode in {
            AgentMode.ANALYZE, 
            AgentMode.SUGGEST, 
            AgentMode.PATCH_PREVIEW, 
            AgentMode.APPLY, 
            AgentMode.REVIEW,
            AgentMode.BUG_HUNT,
            AgentMode.FIX,
            AgentMode.COMPARE
        }:
            return True
        text = (user_input or "").strip().lower()
        if not text or len(text) < 3:
            return False

        # Trivial check for greetings to avoid workspace scan
        tail = text[-120:].strip()
        _small_talk = {
            "selam", "merhaba", "nasılsın", "naber", "ne haber", "iyi misin",
            "kimsin", "ne yapabilirsin", "ordamısın", "ordasın", "orada mısın",
            "burada mısın", "uyandın mı", "hazır mısın", "çalışıyor musun",
            "hi", "hello", "hey", "who are you", "what can you do",
            "are you there", "you there",
        }
        if any(g in tail for g in _small_talk):
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
            "oku",
            "anla",
            "incele",
            "tara",
            "tarat",
            "proje",
            "yapısı",
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

    async def _run_comparison(
        self,
        messages: list[ChatMessage],
        temperature_override: float | None,
        max_tokens_override: int | None,
        workspace_path: Path
    ) -> ComparisonResult:
        self.logger.info("Running parallel models for comparison.")
        
        # 1. Run both models in parallel
        primary_task = self.client.chat(
            messages, 
            temperature_override=temperature_override, 
            max_tokens_override=max_tokens_override,
            model_override=self.settings.nvidia_model
        )
        secondary_task = self.client.chat(
            messages,
            temperature_override=temperature_override,
            max_tokens_override=max_tokens_override,
            model_override=self.settings.nvidia_fallback_model
        )
        
        primary_res, secondary_res = await asyncio.gather(primary_task, secondary_task)
        
        primary_parsed = self.response_parser.parse(primary_res.content)
        secondary_parsed = self.response_parser.parse(secondary_res.content)
        
        # 2. Prepare Judge Prompt
        from agent_core.prompts import JUDGE_SYSTEM_PROMPT
        judge_user_prompt = dedent(f"""
            ### MODEL A (Primary):
            {primary_res.content}
            
            ### MODEL B (Secondary):
            {secondary_res.content}
            
            ### GÖREV:
            Hangi çözüm daha mantıklı? Eğer biri açıkça üstünse onu seç (winner: primary veya winner: secondary). 
            Eğer her ikisinin iyi yanları varsa birleştir ve 'winner: merged' olarak sun.
        """).strip()
        
        judge_messages = [
            ChatMessage(role="system", content=JUDGE_SYSTEM_PROMPT),
            ChatMessage(role="user", content=judge_user_prompt)
        ]
        
        # 3. Judge decides (using primary model as judge)
        judge_res = await self.client.chat(judge_messages, temperature_override=0.1)
        
        winner_match = re.search(r"winner:\s*(primary|secondary|merged)", judge_res.content, re.IGNORECASE)
        winner = winner_match.group(1).lower() if winner_match else "primary"
        
        final_parsed = self.response_parser.parse(judge_res.content)
        if winner == "primary":
            final_parsed = primary_parsed
        elif winner == "secondary":
            final_parsed = secondary_parsed
            
        return ComparisonResult(
            primary_answer=primary_parsed,
            secondary_answer=secondary_parsed,
            judge_thought=judge_res.content,
            winner=winner,
            final_answer=final_parsed
        )

    async def _run_diagnostic(self, workspace_path: Path) -> str | None:
        self.logger.info("Pre-flight diagnostic for workspace: %s", workspace_path)
        
        # Check if pytest exists in the environment first
        import shutil
        if not shutil.which("pytest"):
            self.logger.info("pytest not found in environment. Skipping diagnostic.")
            return None

        try:
            # Add a 5 second timeout to prevent hanging the whole request
            res = await asyncio.wait_for(
                execute_command("pytest --version", cwd=str(workspace_path.resolve())),
                timeout=2.0
            )
            if res.returncode != 0:
                self.logger.info("pytest exists but failed to run version check. Skipping diagnostic.")
                return None

            self.logger.info("Running pytest diagnostic...")
            res = await asyncio.wait_for(
                execute_command("pytest --maxfail=1", cwd=str(workspace_path.resolve())),
                timeout=5.0
            )
            if res.returncode != 0:
                return f"Tests are failing in this project:\nSTDOUT: {res.stdout[:300]}"
        except asyncio.TimeoutError:
            self.logger.warning("Diagnostic check timed out. Skipping.")
            return "Diagnostic timed out (tests might be too slow)."
        except Exception as e:
            self.logger.error("Diagnostic check failed: %s", str(e))
            return None
        return None
