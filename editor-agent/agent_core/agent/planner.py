from __future__ import annotations

from agent_core.models import AgentMode


class AgentPlanner:
    def plan_label(self, mode: AgentMode) -> str:
        if mode == AgentMode.ANALYZE:
            return "Analyze"
        if mode == AgentMode.ASK:
            return "Ask"
        return "Suggest"
