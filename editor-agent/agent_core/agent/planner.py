from __future__ import annotations

from agent_core.models import AgentMode


class AgentPlanner:
    def plan_label(self, mode: AgentMode) -> str:
        if mode == AgentMode.ANALYZE:
            return "Workspace analysis"
        if mode == AgentMode.ASK:
            return "Workspace question answering"
        return "Safe suggestion planning"
