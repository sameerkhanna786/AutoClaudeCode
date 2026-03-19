"""Multi-agent pipeline: Planner -> Coder -> Tester -> Reviewer with revision loops."""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional

from claude_runner import ClaudeResult, ClaudeRunner
from config_schema import Config
from shared import TASK_TYPE_INSTRUCTIONS

logger = logging.getLogger(__name__)

# Import CycleStateWriter optionally to avoid circular deps
try:
    from cycle_state import CycleStateWriter
except ImportError:
    CycleStateWriter = None  # type: ignore[assignment,misc]


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
class AgentCostSummary:
    """Aggregated cost/duration for a single agent role across all invocations."""
    role: str
    total_cost_usd: float = 0.0
    total_duration_seconds: float = 0.0
    invocation_count: int = 0


@dataclass
class ReviewFinding:
    """A single structured finding from the reviewer."""
    filepath: str
    line_number: int = 0
    severity: str = "warning"       # "error" | "warning" | "information"
    category: str = ""              # "quality" | "security" | "testing" | "architecture"
    description: str = ""
    suggestion: str = ""
    confidence: float = 0.8


@dataclass
class ReviewReport:
    """Structured review output with categorized findings."""
    approved: bool
    findings: List[ReviewFinding] = field(default_factory=list)
    summary: str = ""
    test_suggestions: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    success: bool
    agent_results: List[AgentResult] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_duration_seconds: float = 0.0
    revision_count: int = 0
    final_review_approved: bool = False
    error: str = ""
    agent_cost_summary: Dict[str, AgentCostSummary] = field(default_factory=dict)

    def format_cost_report(self) -> str:
        """Format a human-readable cost breakdown by agent role."""
        if not self.agent_cost_summary:
            return "No agent cost data available."
        lines = ["Agent Cost Breakdown:"]
        for role_name in ("planner", "coder", "tester", "reviewer"):
            if role_name in self.agent_cost_summary:
                s = self.agent_cost_summary[role_name]
                pct = (s.total_cost_usd / self.total_cost_usd * 100) if self.total_cost_usd > 0 else 0
                lines.append(
                    f"  {role_name}: ${s.total_cost_usd:.4f} "
                    f"({pct:.1f}%) | {s.total_duration_seconds:.1f}s "
                    f"| {s.invocation_count} invocation(s)"
                )
        lines.append(f"  TOTAL: ${self.total_cost_usd:.4f} | {self.total_duration_seconds:.1f}s")
        return "\n".join(lines)


class AgentWorkspace:
    """Simple file-based workspace for inter-agent communication."""

    def __init__(self, root: str):
        self._root = Path(root)

    def clean(self) -> None:
        """Remove all files in the workspace, tolerating permission errors."""
        if self._root.exists():
            for child in self._root.iterdir():
                try:
                    if child.is_file():
                        child.unlink()
                    elif child.is_dir():
                        import shutil
                        shutil.rmtree(child)
                except (PermissionError, OSError) as e:
                    logger.warning("Could not remove workspace entry %s: %s", child, e)
        self._root.mkdir(parents=True, exist_ok=True)

    def _safe_path(self, name: str) -> Path:
        """Resolve a filename within the workspace, preventing path traversal."""
        target = (self._root / name).resolve()
        root_resolved = self._root.resolve()
        try:
            target.relative_to(root_resolved)
        except ValueError:
            raise ValueError(f"Path traversal blocked: {name!r} escapes workspace root")
        return target

    def write(self, name: str, content: str) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        self._safe_path(name).write_text(content)

    def read(self, name: str) -> Optional[str]:
        path = self._safe_path(name)
        if path.exists():
            return path.read_text()
        return None

    def exists(self, name: str) -> bool:
        return self._safe_path(name).exists()


class AgentPipeline:
    """Orchestrates a Planner -> Coder -> Tester -> Reviewer pipeline."""

    def __init__(self, config: Config, cycle_state: Optional["CycleStateWriter"] = None):
        self.config = config
        self._ws_dir = str(
            Path(config.target_dir) / config.paths.agent_workspace_dir
        )
        self._active_runner: Optional[ClaudeRunner] = None
        self._runner_lock = threading.Lock()
        self._terminated = False
        self._cycle_state = cycle_state

    def terminate(self) -> None:
        """Terminate the currently running agent subprocess.

        Thread-safe: can be called from a timeout handler in another thread.
        """
        self._terminated = True
        with self._runner_lock:
            runner = self._active_runner
        if runner is not None:
            logger.warning("Terminating active pipeline agent subprocess")
            runner.terminate()

    def _build_runner_for_agent(self, role: AgentRole) -> ClaudeRunner:
        """Build a runner with per-agent model/timeout overrides."""
        agent_config = copy.deepcopy(self.config)
        role_cfg = getattr(self.config.agent_pipeline, role.value)
        required_attrs = ("model", "max_turns", "timeout_seconds")
        missing = [a for a in required_attrs if not hasattr(role_cfg, a)]
        if missing:
            raise ValueError(
                f"Agent role '{role.value}' config is missing required attributes: "
                f"{', '.join(missing)}. Check agent_pipeline.{role.value} in config."
            )
        agent_config.claude.model = role_cfg.model
        agent_config.claude.resolved_model = ""
        agent_config.claude.max_turns = role_cfg.max_turns
        agent_config.claude.timeout_seconds = role_cfg.timeout_seconds
        from provider_runner import create_runner
        return create_runner(agent_config)

    @staticmethod
    def _parse_review_verdict(review_text: str) -> bool:
        """Parse VERDICT from reviewer output. Defaults to REVISE (conservative)."""
        if not review_text:
            return False
        for line in review_text.splitlines():
            match = re.match(r"\s*VERDICT:\s*(APPROVED|REVISE)\s*", line, re.IGNORECASE)
            if match:
                return match.group(1).upper() == "APPROVED"
        return False

    @staticmethod
    def _parse_structured_review(review_text: str) -> Optional[ReviewReport]:
        """Parse structured review JSON from reviewer output.

        Looks for a ```json ... ``` block containing a review report with
        findings. Returns None if no structured review is found (caller
        should fall back to _parse_review_verdict).
        """
        if not review_text:
            return None

        # Extract JSON block from markdown code fence
        json_match = re.search(
            r"```json\s*\n(.*?)\n\s*```", review_text, re.DOTALL
        )
        if not json_match:
            return None

        try:
            data = json.loads(json_match.group(1))
        except (json.JSONDecodeError, ValueError):
            return None

        if not isinstance(data, dict):
            return None

        # Must have "findings" key to be considered structured
        if "findings" not in data:
            return None

        findings = []
        for f in data.get("findings", []):
            if not isinstance(f, dict):
                continue
            try:
                line_number = int(f.get("line_number", 0))
            except (ValueError, TypeError):
                line_number = 0
            try:
                confidence = float(f.get("confidence", 0.8))
            except (ValueError, TypeError):
                confidence = 0.8
            findings.append(ReviewFinding(
                filepath=str(f.get("filepath", "")),
                line_number=line_number,
                severity=str(f.get("severity", "warning")).lower(),
                category=str(f.get("category", "")),
                description=str(f.get("description", "")),
                suggestion=str(f.get("suggestion", "")),
                confidence=confidence,
            ))

        # Parse verdict from the JSON or fall back to text
        approved = bool(data.get("approved", False))
        if "approved" not in data:
            approved = AgentPipeline._parse_review_verdict(review_text)

        return ReviewReport(
            approved=approved,
            findings=findings,
            summary=str(data.get("summary", "")),
            test_suggestions=[
                str(s) for s in data.get("test_suggestions", [])
                if isinstance(s, str)
            ],
        )

    @staticmethod
    def _filter_findings(
        findings: List[ReviewFinding],
        confidence_threshold: float = 0.70,
    ) -> List[ReviewFinding]:
        """Filter findings by severity-dependent confidence thresholds.

        - error: keep if confidence >= max(0.85, threshold)
        - warning: keep if confidence >= threshold
        - information: keep if confidence >= 0.25
        Deduplicates by (filepath, line_number).
        """
        severity_thresholds = {
            "error": max(0.85, confidence_threshold),
            "warning": confidence_threshold,
            "information": 0.25,
        }

        seen = set()
        filtered = []
        for f in findings:
            threshold = severity_thresholds.get(f.severity, confidence_threshold)
            if f.confidence < threshold:
                continue
            key = (f.filepath, f.line_number)
            if key in seen and f.line_number != 0:
                continue
            if f.line_number != 0:
                seen.add(key)
            filtered.append(f)
        return filtered

    @staticmethod
    def _format_structured_feedback(report: ReviewReport) -> str:
        """Format a ReviewReport into structured text for the Coder revision prompt.

        Groups findings by file, includes severity and line numbers,
        and only surfaces error/warning items.
        """
        lines = ["## Review Findings (address in priority order)\n"]

        # Group by file
        by_file: Dict[str, List[ReviewFinding]] = {}
        for f in report.findings:
            if f.severity not in ("error", "warning"):
                continue
            by_file.setdefault(f.filepath or "(general)", []).append(f)

        if not by_file:
            return report.summary or "No actionable findings."

        # Sort: errors first within each file
        severity_order = {"error": 0, "warning": 1}
        for filepath in sorted(by_file.keys()):
            lines.append(f"### {filepath}")
            file_findings = sorted(
                by_file[filepath],
                key=lambda x: (severity_order.get(x.severity, 2), x.line_number),
            )
            for f in file_findings:
                loc = f"line {f.line_number}" if f.line_number else "general"
                lines.append(
                    f"- [{f.severity.upper()}] ({loc}) {f.description}"
                )
                if f.suggestion:
                    lines.append(f"  Suggestion: {f.suggestion}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _format_test_suggestions(report: ReviewReport) -> str:
        """Format test suggestions from a ReviewReport for the Tester agent."""
        parts = []
        if report.test_suggestions:
            parts.append("## Reviewer Test Suggestions\n")
            for i, s in enumerate(report.test_suggestions, 1):
                parts.append(f"{i}. {s}")
            parts.append("")

        # Also include testing-category findings
        test_findings = [
            f for f in report.findings if f.category == "testing"
        ]
        if test_findings:
            parts.append("## Testing Gaps Identified by Reviewer\n")
            for f in test_findings:
                loc = f"{f.filepath}:{f.line_number}" if f.line_number else f.filepath
                parts.append(f"- {loc}: {f.description}")
            parts.append("")

        return "\n".join(parts) if parts else ""

    def _build_task_description(self, tasks: list) -> str:
        """Combine task descriptions into a single prompt block."""
        if len(tasks) == 1:
            return tasks[0].description
        lines = []
        for i, t in enumerate(tasks, 1):
            lines.append(f"{i}. {t.description}")
        return "\n".join(lines)

    @staticmethod
    def _update_cost_summary(
        result: PipelineResult, agent_result: AgentResult
    ) -> None:
        """Update the per-agent cost summary with a new agent result."""
        role_name = agent_result.role.value
        if role_name not in result.agent_cost_summary:
            result.agent_cost_summary[role_name] = AgentCostSummary(role=role_name)
        summary = result.agent_cost_summary[role_name]
        summary.total_cost_usd += agent_result.cost_usd
        summary.total_duration_seconds += agent_result.duration_seconds
        summary.invocation_count += 1

    def run(
        self,
        tasks: list,
        rollback_fn: Callable[..., None],
        snapshot,
    ) -> PipelineResult:
        """Execute the full pipeline, returning a PipelineResult."""
        ap = self.config.agent_pipeline
        workspace = AgentWorkspace(self._ws_dir)
        task_desc = self._build_task_description(tasks)

        result = PipelineResult(success=False)

        def _run_agent(role: AgentRole, prompt: str) -> AgentResult:
            role_cfg = getattr(ap, role.value)
            if not role_cfg.enabled:
                return AgentResult(
                    role=role, success=True, output_text="(skipped)",
                )
            # Update live cycle state
            if self._cycle_state is not None:
                self._cycle_state.update(
                    pipeline_agent=role.value,
                    accumulated_cost=result.total_cost_usd,
                )
            agent_runner = self._build_runner_for_agent(role)
            with self._runner_lock:
                self._active_runner = agent_runner
            try:
                if self._terminated:
                    return AgentResult(
                        role=role, success=False, error="Pipeline was terminated",
                    )
                cr = agent_runner.run(prompt)
            finally:
                with self._runner_lock:
                    self._active_runner = None
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

        # --- Planner (runs once, not on revisions) ---
        # Skip planning for simple task types configured in skip_planning_for
        skip_planning_for = getattr(ap, "skip_planning_for", [])
        task_sources = {getattr(t, "source", "") for t in tasks}
        skip_planner = (
            bool(skip_planning_for)
            and bool(task_sources)
            and all(s in skip_planning_for for s in task_sources)
        )

        if skip_planner:
            logger.info(
                "Skipping planner for simple task type(s): %s",
                ", ".join(sorted(task_sources)),
            )
            plan_text = task_desc
        else:
            # Gather task-type-specific planning instructions
            task_type_guidance = ""
            task_sources = {getattr(t, "source", "") for t in tasks}
            for src in sorted(task_sources):
                instructions = TASK_TYPE_INSTRUCTIONS.get(src, "")
                if instructions:
                    task_type_guidance += f"\n\n## Guidelines for '{src}' tasks:\n{instructions}"

            planner_prompt = (
                f"You are the PLANNER agent.\n\n"
                f"TASK:\n{task_desc}\n\n"
            )
            if task_type_guidance:
                planner_prompt += (
                    f"TASK-TYPE-SPECIFIC GUIDELINES:{task_type_guidance}\n\n"
                )
            planner_prompt += (
                f"Create a detailed plan for implementing the above task. "
                f"Write the plan to {self._ws_dir}/plan.md"
            )
            planner_result = _run_agent(AgentRole.PLANNER, planner_prompt)
            result.agent_results.append(planner_result)
            result.total_cost_usd += planner_result.cost_usd
            result.total_duration_seconds += planner_result.duration_seconds
            self._update_cost_summary(result, planner_result)

            if not planner_result.success:
                result.error = f"Planner failed: {planner_result.error}"
                logger.info(result.format_cost_report())
                return result

            # Rollback any file changes from planner
            rollback_fn(snapshot)

            plan_text = workspace.read("plan.md") or planner_result.output_text

        while True:
            # Cost guard: abort if accumulated cost exceeds pipeline budget
            pipeline_cost_limit = ap.max_pipeline_cost_usd
            if pipeline_cost_limit <= 0:
                pipeline_cost_limit = self.config.safety.max_cost_usd_per_hour * 0.5
            if result.total_cost_usd >= pipeline_cost_limit:
                logger.warning(
                    "Pipeline cost guard: $%.2f accumulated (limit $%.2f), aborting",
                    result.total_cost_usd, pipeline_cost_limit,
                )
                result.error = (
                    f"Pipeline cost limit exceeded "
                    f"(${result.total_cost_usd:.2f} >= ${pipeline_cost_limit:.2f})"
                )
                logger.info(result.format_cost_report())
                return result

            if self._terminated:
                result.error = "Pipeline was terminated"
                return result

            # Read review feedback before cleaning workspace (reviewer wrote it last iteration)
            review_text = workspace.read("review.md") or ""
            last_review_report: Optional[ReviewReport] = None
            if revision > 0 and review_text:
                last_review_report = self._parse_structured_review(review_text)
                if last_review_report is not None:
                    confidence_threshold = getattr(
                        ap, "review_confidence_threshold", 0.70
                    )
                    last_review_report.findings = self._filter_findings(
                        last_review_report.findings, confidence_threshold
                    )
            workspace.clean()

            # --- Coder ---

            revision_context = ""
            if revision > 0 and review_text:
                if last_review_report is not None and last_review_report.findings:
                    structured_feedback = self._format_structured_feedback(
                        last_review_report
                    )
                    revision_context = (
                        f"\n\nPREVIOUS REVIEW FEEDBACK (revision {revision}):\n"
                        f"{structured_feedback}\n"
                        f"Address the reviewer's findings above, prioritizing "
                        f"errors first, then warnings."
                    )
                else:
                    revision_context = (
                        f"\n\nPREVIOUS REVIEW FEEDBACK (revision {revision}):\n"
                        f"{review_text}\n"
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
            self._update_cost_summary(result, coder_result)

            if not coder_result.success:
                result.error = f"Coder failed: {coder_result.error}"
                logger.info(result.format_cost_report())
                return result

            # --- Tester ---
            test_suggestion_context = ""
            if last_review_report is not None:
                test_suggestions_text = self._format_test_suggestions(
                    last_review_report
                )
                if test_suggestions_text:
                    test_suggestion_context = f"\n\n{test_suggestions_text}"

            tester_prompt = (
                f"You are the TESTER agent.\n\n"
                f"TASK:\n{task_desc}\n\n"
                f"Run the test suite and report any failures."
                f"{test_suggestion_context}"
            )
            tester_result = _run_agent(AgentRole.TESTER, tester_prompt)
            result.agent_results.append(tester_result)
            result.total_cost_usd += tester_result.cost_usd
            result.total_duration_seconds += tester_result.duration_seconds
            self._update_cost_summary(result, tester_result)

            # Fix 7: Check tester result — if the tester CLI crashed, treat
            # it as a revision-needed signal rather than silently continuing.
            if not tester_result.success:
                logger.warning(
                    "Tester agent failed: %s — treating as revision needed",
                    tester_result.error,
                )
                if revision < max_revisions:
                    revision += 1
                    result.revision_count = revision
                    rollback_fn(snapshot)
                    workspace.write("review.md", f"VERDICT: REVISE\nTester failed: {tester_result.error}")
                    continue
                else:
                    result.error = f"Tester failed after exhausting revisions: {tester_result.error}"
                    result.revision_count = revision
                    logger.info(result.format_cost_report())
                    return result

            # --- Reviewer ---
            review_detail = getattr(ap, "review_detail", "standard")

            if review_detail == "minimal":
                reviewer_instructions = (
                    f"Review the code changes. Write your review to "
                    f"{self._ws_dir}/review.md.\n"
                    f"End your review with either:\n"
                    f"VERDICT: APPROVED\n"
                    f"or:\n"
                    f"VERDICT: REVISE"
                )
            else:
                detail_guidance = ""
                if review_detail == "thorough":
                    detail_guidance = (
                        "Be thorough: examine every changed file, check edge "
                        "cases, and verify error handling.\n\n"
                    )

                reviewer_instructions = (
                    f"Review the code changes across these categories:\n"
                    f"1. **Code Quality & Performance** — unnecessary code, "
                    f"resource leaks, N+1 patterns, missing error handling\n"
                    f"2. **Best Practices** — anti-patterns, deprecated usage, "
                    f"naming conventions, framework compliance\n"
                    f"3. **Test Coverage** — untested code paths, missing edge "
                    f"cases, suggest specific test cases\n"
                    f"4. **Architectural Concerns** — circular deps, tight "
                    f"coupling, layer violations\n\n"
                    f"{detail_guidance}"
                    f"Write your review to {self._ws_dir}/review.md.\n\n"
                    f"Include a structured JSON block in your review with this format:\n"
                    f"```json\n"
                    f'{{\n'
                    f'  "approved": false,\n'
                    f'  "summary": "Brief overall assessment",\n'
                    f'  "findings": [\n'
                    f'    {{\n'
                    f'      "filepath": "path/to/file.py",\n'
                    f'      "line_number": 42,\n'
                    f'      "severity": "error",\n'
                    f'      "category": "quality",\n'
                    f'      "description": "What is wrong",\n'
                    f'      "suggestion": "How to fix it",\n'
                    f'      "confidence": 0.95\n'
                    f'    }}\n'
                    f'  ],\n'
                    f'  "test_suggestions": [\n'
                    f'    "Test that X handles empty input"\n'
                    f'  ]\n'
                    f'}}\n'
                    f"```\n\n"
                    f"Severity levels: \"error\" (must fix), \"warning\" "
                    f"(should fix), \"information\" (nice to have).\n"
                    f"Categories: \"quality\", \"security\", \"testing\", "
                    f"\"architecture\".\n"
                    f"Confidence: 0.0 to 1.0 — how certain you are this is a "
                    f"real issue.\n\n"
                    f"After the JSON block, end your review with:\n"
                    f"VERDICT: APPROVED\n"
                    f"or:\n"
                    f"VERDICT: REVISE"
                )

            reviewer_prompt = (
                f"You are the REVIEWER agent.\n\n"
                f"TASK:\n{task_desc}\n\n"
                f"{reviewer_instructions}"
            )
            reviewer_result = _run_agent(AgentRole.REVIEWER, reviewer_prompt)
            result.agent_results.append(reviewer_result)
            result.total_cost_usd += reviewer_result.cost_usd
            result.total_duration_seconds += reviewer_result.duration_seconds
            self._update_cost_summary(result, reviewer_result)

            # Determine verdict
            if not getattr(ap.reviewer, "enabled", True):
                # Reviewer disabled -> auto-approve
                result.success = True
                result.final_review_approved = True
                logger.info(result.format_cost_report())
                return result

            # If the reviewer CLI crashed, abort rather than wasting revisions
            if not reviewer_result.success:
                result.error = f"Reviewer agent failed: {reviewer_result.error}"
                logger.warning(result.error)
                logger.info(result.format_cost_report())
                return result

            review_content = workspace.read("review.md") or reviewer_result.output_text

            # If the reviewer produced no VERDICT line, abort rather than
            # defaulting to REVISE and wasting costly revision iterations
            if not review_content or "VERDICT:" not in review_content.upper():
                result.error = (
                    "Reviewer produced no VERDICT line — aborting to avoid "
                    "wasteful revision loop"
                )
                logger.warning(result.error)
                logger.info(result.format_cost_report())
                return result

            # Try structured review parsing first, fall back to simple verdict
            structured_report = self._parse_structured_review(review_content)
            if structured_report is not None:
                approved = structured_report.approved
            else:
                approved = self._parse_review_verdict(review_content)

            if approved:
                result.success = True
                result.final_review_approved = True
                logger.info(result.format_cost_report())
                return result

            # Reviewer rejected — try revision if budget allows
            if revision < max_revisions:
                revision += 1
                result.revision_count = revision
                if self._cycle_state is not None:
                    self._cycle_state.update(pipeline_revision=revision)
                rollback_fn(snapshot)
                # Restore review feedback so the next iteration can read it
                # (rollback's git clean -fd deletes untracked workspace files)
                workspace.write("review.md", review_content)
                # Loop continues with new iteration
            else:
                # Exhausted revisions
                result.success = False
                result.error = "Reviewer rejected after exhausting all revisions"
                result.final_review_approved = False
                result.revision_count = revision
                logger.info(result.format_cost_report())
                return result
