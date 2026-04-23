"""
AgentPlanner — Converts an AgentMode into a human-readable label and a
list of planned execution steps.

Responsibilities:
  - ``plan_label(mode)``: Maps an AgentMode value to a short display label
    used in logging and session metadata.
  - ``plan_steps(mode)``: Returns an ordered list of step descriptions for
    the given mode.  Currently returns an empty list; reserved for future
    multi-step planning expansion.

This class is stateless and has no side effects.  It is injected into
AgentOrchestrator via OrchestratorDependencies.
"""
from __future__ import annotations

from agent_core.models import AgentMode


class AgentPlanner:
    @staticmethod
    def plan_label(mode: AgentMode) -> str:
        """Return a human-readable label for *mode*.

        Returns ``"Unknown"`` for any value not explicitly handled rather than
        raising ``ValueError``, so callers never need to guard against exceptions.
        """
        if mode == AgentMode.ANALYZE:
            return "Analyze"
        if mode == AgentMode.ASK:
            return "Ask"
        if mode == AgentMode.SUGGEST:
            return "Suggest"
        if mode == AgentMode.PATCH_PREVIEW:
            return "Patch Preview"
        if mode == AgentMode.APPLY:
            return "Apply"
        if mode == AgentMode.REVIEW:
            return "Review"
        if mode == AgentMode.COMPARE:
            return "Compare"
        if mode == AgentMode.BUG_HUNT:
            return "Bug Hunt"
        if mode == AgentMode.FIX:
            return "Fix"
        return "Unknown"

    @staticmethod
    def plan_steps(mode: AgentMode) -> list[str]:
        """Return an ordered list of step descriptions for *mode*.

        Currently returns an empty list.  Intended as an extension point for
        future multi-step planning (e.g. decomposing a APPLY request into
        scan → rank → read → generate → apply steps).
        """
        return []
