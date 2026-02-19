"""Multi-agent pipeline: Planner -> Coder -> Tester -> Reviewer with revision loops."""

from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional

from claude_runner import ClaudeResult, ClaudeRunner
from config_schema import Config

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    PLANNER = "planner"
    CODER = "coder"
    TESTER = "tester"
    REVIEWER = "reviewer"


@dataclass
class AgentResult:
    role: AgentRole
    success: bool
    output_text: str = ""
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    error: str = ""


@dataclass
class PipelineResult:
    success: bool
    agent_results: List[AgentResult] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_duration_seconds: float = 0.0
    revision_count: int = 0
    final_review_approved: bool = False
    error: str = ""


class AgentWorkspace:
    """Simple file-based workspace for inter-agent communication."""

    def __init__(self, root: str):
        self._root = Path(root)

    def clean(self) -> None:
        """Remove all files in the workspace."""
        if self._root.exists():
            for child in self._root.iterdir():
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    import shutil
                    shutil.rmtree(child)
        self._root.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, content: str) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        (self._root / name).write_text(content)

    def read(self, name: str) -> Optional[str]:
        path = self._root / name
        if path.exists():
            return path.read_text()
        return None

    def exists(self, name: str) -> bool:
        return (self._root / name).exists()


class AgentPipeline:
    """Orchestrates a Planner -> Coder -> Tester -> Reviewer pipeline."""

    def __init__(self, config: Config):
        self.config = config
        self._ws_dir = str(
            Path(config.target_dir) / config.paths.agent_workspace_dir
        )

    def _build_runner_for_agent(self, role: AgentRole) -> ClaudeRunner:
        """Build a ClaudeRunner with per-agent model/timeout overrides."""
        agent_config = copy.deepcopy(self.config)
        role_cfg = getattr(self.config.agent_pipeline, role.value)
        agent_config.claude.model = role_cfg.model
        agent_config.claude.max_turns = role_cfg.max_turns
        agent_config.claude.timeout_seconds = role_cfg.timeout_seconds
        return ClaudeRunner(agent_config)

    @staticmethod
    def _parse_review_verdict(review_text: str) -> bool:
        """Parse VERDICT from reviewer output. Defaults to approved."""
        if not review_text:
            return True
        for line in review_text.splitlines():
            match = re.match(r"\s*VERDICT:\s*(APPROVED|REVISE)\s*", line, re.IGNORECASE)
            if match:
                return match.group(1).upper() == "APPROVED"
        return True

    def _build_task_description(self, tasks: list) -> str:
        """Combine task descriptions into a single prompt block."""
        if len(tasks) == 1:
            return tasks[0].description
        lines = []
        for i, t in enumerate(tasks, 1):
            lines.append(f"{i}. {t.description}")
        return "\n".join(lines)

    def run(
        self,
        tasks: list,
        rollback_fn: Callable[[str], None],
        snapshot: str,
    ) -> PipelineResult:
        """Execute the full pipeline, returning a PipelineResult."""
        ap = self.config.agent_pipeline
        workspace = AgentWorkspace(self._ws_dir)
        task_desc = self._build_task_description(tasks)

        result = PipelineResult(success=False)

        runner = ClaudeRunner(self.config)

        def _run_agent(role: AgentRole, prompt: str) -> AgentResult:
            role_cfg = getattr(ap, role.value)
            if not role_cfg.enabled:
                return AgentResult(
                    role=role, success=True, output_text="(skipped)",
                )
            cr = runner.run(prompt, self.config.target_dir)
            return AgentResult(
                role=role,
                success=cr.success,
                output_text=cr.result_text,
                cost_usd=cr.cost_usd,
                duration_seconds=cr.duration_seconds,
                error=cr.error,
            )

        max_revisions = ap.max_revisions
        revision = 0

        while True:
            workspace.clean()

            # --- Planner ---
            planner_prompt = (
                f"You are the PLANNER agent.\n\n"
                f"TASK:\n{task_desc}\n\n"
                f"Create a detailed plan for implementing the above task. "
                f"Write the plan to {self._ws_dir}/plan.md"
            )
            planner_result = _run_agent(AgentRole.PLANNER, planner_prompt)
            result.agent_results.append(planner_result)
            result.total_cost_usd += planner_result.cost_usd
            result.total_duration_seconds += planner_result.duration_seconds

            if not planner_result.success:
                result.error = f"Planner failed: {planner_result.error}"
                return result

            # Rollback any file changes from planner
            rollback_fn(snapshot)

            # --- Coder ---
            plan_text = workspace.read("plan.md") or planner_result.output_text
            review_text = workspace.read("review.md") or ""

            revision_context = ""
            if revision > 0 and review_text:
                revision_context = (
                    f"\n\nPREVIOUS REVIEW FEEDBACK (revision {revision}):\n{review_text}\n"
                    f"Address the reviewer's feedback in your implementation."
                )

            coder_prompt = (
                f"You are the CODER agent.\n\n"
                f"TASK:\n{task_desc}\n\n"
                f"PLAN:\n{plan_text}\n"
                f"{revision_context}\n"
                f"Implement the changes described in the plan."
            )
            coder_result = _run_agent(AgentRole.CODER, coder_prompt)
            result.agent_results.append(coder_result)
            result.total_cost_usd += coder_result.cost_usd
            result.total_duration_seconds += coder_result.duration_seconds

            if not coder_result.success:
                result.error = f"Coder failed: {coder_result.error}"
                return result

            # --- Tester ---
            tester_prompt = (
                f"You are the TESTER agent.\n\n"
                f"TASK:\n{task_desc}\n\n"
                f"Run the test suite and report any failures."
            )
            tester_result = _run_agent(AgentRole.TESTER, tester_prompt)
            result.agent_results.append(tester_result)
            result.total_cost_usd += tester_result.cost_usd
            result.total_duration_seconds += tester_result.duration_seconds

            # --- Reviewer ---
            reviewer_prompt = (
                f"You are the REVIEWER agent.\n\n"
                f"TASK:\n{task_desc}\n\n"
                f"Review the code changes. Write your review to "
                f"{self._ws_dir}/review.md.\n"
                f"End your review with either:\n"
                f"VERDICT: APPROVED\n"
                f"or:\n"
                f"VERDICT: REVISE"
            )
            reviewer_result = _run_agent(AgentRole.REVIEWER, reviewer_prompt)
            result.agent_results.append(reviewer_result)
            result.total_cost_usd += reviewer_result.cost_usd
            result.total_duration_seconds += reviewer_result.duration_seconds

            # Determine verdict
            if not getattr(ap.reviewer, "enabled", True):
                # Reviewer disabled -> auto-approve
                result.success = True
                result.final_review_approved = True
                return result

            review_content = workspace.read("review.md") or reviewer_result.output_text
            approved = self._parse_review_verdict(review_content)

            if approved:
                result.success = True
                result.final_review_approved = True
                return result

            # Reviewer rejected — try revision if budget allows
            if revision < max_revisions:
                revision += 1
                result.revision_count = revision
                rollback_fn(snapshot)
                # Loop continues with new iteration
            else:
                # Exhausted revisions
                result.success = True
                result.final_review_approved = False
                result.revision_count = revision
                return result
