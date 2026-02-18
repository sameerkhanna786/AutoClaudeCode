"""Main orchestration loop tying all components together."""

from __future__ import annotations

import ast
import logging
import os
import shutil
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from claude_runner import ClaudeRunner, ClaudeResult
from config_schema import Config
from feedback import FeedbackManager
from git_manager import GitManager
from safety import SafetyError, SafetyGuard
from state import CycleRecord, StateManager
from task_discovery import Task, TaskDiscovery
from validator import ValidationResult, Validator

logger = logging.getLogger(__name__)

# The prompt template sent to Claude
CLAUDE_PROMPT_TEMPLATE = """\
You are working on the project in the current directory.

TASK: {task_description}

INSTRUCTIONS:
- Make the minimal changes needed to complete this task.
- Do NOT run git commands (add, commit, push). The orchestrator handles git.
- Do NOT modify these protected files: {protected_files}
- Focus on correctness. Run tests if available.
- If the task is unclear or impossible, make your best effort and explain what you did.
"""

CLAUDE_PLAN_TEMPLATE = """\
You are working on the project in the current directory.

TASK: {task_description}

INSTRUCTIONS:
- Analyze the codebase and create a detailed plan to complete this task.
- Do NOT make any changes yet. Only output a plan.
- List the files you would modify and what changes you would make.
- Do NOT modify these protected files: {protected_files}
- Be specific about the changes (function names, line numbers, etc.).
"""

CLAUDE_EXECUTE_TEMPLATE = """\
You are working on the project in the current directory.

TASK: {task_description}

PLAN TO EXECUTE:
{plan}

INSTRUCTIONS:
- Execute the plan above by making the described changes.
- Do NOT run git commands (add, commit, push). The orchestrator handles git.
- Do NOT modify these protected files: {protected_files}
- Focus on correctness. Run tests if available.
- Stick to the plan. Do not deviate unless the plan has an obvious error.
"""

BATCH_PLAN_TEMPLATE = """\
You are working on the project in the current directory.

You have been given a batch of tasks to address in a single comprehensive change.

TASKS:
{task_list}

ADDITIONAL CHECKS (always perform these):
{task_count}. Check whether any of the above changes require NEW tests to be added. \
If new functionality is introduced or existing behavior is changed, plan to add or update tests.
{task_count_plus1}. Check whether README.md needs updating to reflect any of the above changes. \
If user-facing behavior, configuration options, or architecture changed, plan to update README.md.

INSTRUCTIONS:
- Analyze the codebase and create a detailed, comprehensive plan that addresses ALL tasks above.
- Do NOT make any changes yet. Only output a plan.
- List every file you would modify and what changes you would make in each.
- Do NOT modify these protected files: {protected_files}
- Be specific about the changes (function names, line numbers, etc.).
- Group related changes together where possible for clarity.
- Address the tasks in priority order but look for opportunities to combine related changes.
"""

BATCH_EXECUTE_TEMPLATE = """\
You are working on the project in the current directory.

You have been given a batch of tasks to address in a single comprehensive change.

TASKS:
{task_list}

PLAN TO EXECUTE:
{plan}

INSTRUCTIONS:
- Execute the plan above by making ALL described changes.
- Do NOT run git commands (add, commit, push). The orchestrator handles git.
- Do NOT modify these protected files: {protected_files}
- Focus on correctness. Run tests after making changes.
- Stick to the plan. Do not deviate unless the plan has an obvious error.
- Make ALL changes in this single session. This is a comprehensive revamp, not incremental.
"""


class Orchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.state = StateManager(config)
        self.safety = SafetyGuard(config, self.state)
        self.claude = ClaudeRunner(config)
        self.git = GitManager(config.target_dir)
        self.validator = Validator(config)
        self.discovery = TaskDiscovery(config)
        self.feedback = FeedbackManager(config)
        self._running = True

    def _setup_signals(self) -> None:
        """Register signal handlers for graceful shutdown."""
        def handler(signum, frame):
            logger.info("Received signal %d, shutting down gracefully...", signum)
            self._running = False

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _build_prompt(self, task: Task) -> str:
        """Build the Claude prompt for a given task."""
        protected = ", ".join(self.config.safety.protected_files)
        return CLAUDE_PROMPT_TEMPLATE.format(
            task_description=task.description,
            protected_files=protected,
        )

    def _build_plan_prompt(self, task: Task) -> str:
        """Build a planning-only prompt for a task."""
        protected = ", ".join(self.config.safety.protected_files)
        return CLAUDE_PLAN_TEMPLATE.format(
            task_description=task.description,
            protected_files=protected,
        )

    def _build_execute_prompt(self, task: Task, plan: str) -> str:
        """Build an execution prompt with a pre-approved plan."""
        protected = ", ".join(self.config.safety.protected_files)
        return CLAUDE_EXECUTE_TEMPLATE.format(
            task_description=task.description,
            plan=plan,
            protected_files=protected,
        )

    def _pick_task(self) -> Optional[Task]:
        """Pick the highest-priority task, checking feedback first."""
        tasks = self._gather_tasks()
        return tasks[0] if tasks else None

    def _gather_tasks(self) -> List[Task]:
        """Gather all eligible tasks, respecting batch_mode and max_tasks_per_cycle."""
        tasks: List[Task] = []

        # Priority 1: developer feedback
        max_retries = self.config.orchestrator.max_feedback_retries
        for task in self.feedback.get_pending_feedback():
            failure_count = self.state.get_task_failure_count(task.description, "feedback")
            if failure_count >= max_retries:
                logger.warning(
                    "Feedback task failed %d times, moving to failed/", failure_count
                )
                if task.source_file:
                    self.feedback.mark_failed(task.source_file)
                continue
            if not self.state.was_recently_attempted(task.description):
                tasks.append(task)

        # Auto-discovered tasks
        discovered = self.discovery.discover_all()
        for task in discovered:
            if not self.state.was_recently_attempted(task.description):
                tasks.append(task)

        if not self.config.orchestrator.batch_mode:
            return tasks[:1]

        return tasks[:self.config.orchestrator.max_tasks_per_cycle]

    def _format_task_list(self, tasks: List[Task]) -> str:
        """Format tasks as a numbered list with source tags."""
        lines = []
        for i, task in enumerate(tasks, 1):
            lines.append(f"{i}. {task.description} [{task.source}]")
        return "\n".join(lines)

    def _build_batch_plan_prompt(self, tasks: List[Task]) -> str:
        """Build a batch planning prompt for multiple tasks."""
        protected = ", ".join(self.config.safety.protected_files)
        task_list = self._format_task_list(tasks)
        task_count = len(tasks) + 1
        task_count_plus1 = len(tasks) + 2
        return BATCH_PLAN_TEMPLATE.format(
            task_list=task_list,
            task_count=task_count,
            task_count_plus1=task_count_plus1,
            protected_files=protected,
        )

    def _build_batch_execute_prompt(self, tasks: List[Task], plan: str) -> str:
        """Build a batch execution prompt with a pre-approved plan."""
        protected = ", ".join(self.config.safety.protected_files)
        task_list = self._format_task_list(tasks)
        return BATCH_EXECUTE_TEMPLATE.format(
            task_list=task_list,
            plan=plan,
            protected_files=protected,
        )

    def _build_batch_commit_message(self, tasks: List[Task]) -> str:
        """Build a commit message summarizing a batch of tasks."""
        sources = sorted(set(t.source for t in tasks))
        header = f"[auto] batch({len(tasks)}): {', '.join(sources)}"
        body_lines = [f"  - {t.description[:80]}" for t in tasks]
        return header + "\n\n" + "\n".join(body_lines)

    def _make_cycle_record(self, tasks: List[Task], **kwargs) -> CycleRecord:
        """Construct a CycleRecord with both singular and batch list fields."""
        primary = tasks[0] if tasks else Task(description="unknown", priority=99, source="unknown")
        record = CycleRecord(
            timestamp=kwargs.get("timestamp", time.time()),
            task_description=primary.description,
            task_type=primary.source,
            success=kwargs.get("success", False),
            commit_hash=kwargs.get("commit_hash", ""),
            cost_usd=kwargs.get("cost_usd", 0.0),
            duration_seconds=kwargs.get("duration_seconds", 0.0),
            validation_summary=kwargs.get("validation_summary", ""),
            error=kwargs.get("error", ""),
            task_descriptions=[t.description for t in tasks],
            task_types=[t.source for t in tasks],
        )
        return record

    def _syntax_check_files(self, changed_files: List[str]) -> Optional[str]:
        """If self_improve is on, syntax-check any modified .py files."""
        if not self.config.orchestrator.self_improve:
            return None

        for f in changed_files:
            if f.endswith(".py"):
                full_path = Path(self.config.target_dir) / f
                if full_path.exists():
                    try:
                        source = full_path.read_text()
                        ast.parse(source, filename=f)
                    except SyntaxError as e:
                        return f"Syntax error in {f}: {e}"
        return None

    def _backup_orchestrator_files(self) -> None:
        """If self_improve is on, back up key orchestrator files before a cycle."""
        if not self.config.orchestrator.self_improve:
            return

        backup_dir = Path(self.config.paths.backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)

        files_to_backup = [
            "orchestrator.py", "claude_runner.py", "validator.py",
            "git_manager.py", "state.py", "safety.py", "task_discovery.py",
            "feedback.py", "config_schema.py",
        ]
        for fname in files_to_backup:
            src = Path(self.config.target_dir) / fname
            if src.exists():
                dst = backup_dir / fname
                shutil.copy2(str(src), str(dst))

    def _cycle(self) -> None:
        """Run a single orchestration cycle."""
        # 1. Pre-flight safety checks
        try:
            self.safety.pre_flight_checks()
        except SafetyError as e:
            logger.warning("Pre-flight check failed: %s", e)
            return

        # 2-5. Gather tasks
        tasks = self._gather_tasks()
        if not tasks:
            logger.info("No tasks found, sleeping...")
            return

        is_batch = len(tasks) > 1 and self.config.orchestrator.batch_mode

        if is_batch:
            logger.info("Selected %d tasks for batch processing", len(tasks))
            for i, t in enumerate(tasks, 1):
                logger.info("  Task %d [priority=%d]: %s", i, t.priority, t.description)
        else:
            task = tasks[0]
            logger.info("Selected task [priority=%d]: %s", task.priority, task.description)

        # Backup orchestrator files if self-improving
        self._backup_orchestrator_files()

        # 6. Record git snapshot
        snapshot = self.git.create_snapshot()
        pre_existing_files = self.git.capture_worktree_state()

        # 7. Invoke Claude (with optional plan-then-execute)
        total_cost = 0.0
        total_duration = 0.0

        if self.config.orchestrator.plan_changes:
            # Phase 1: Plan
            if is_batch:
                plan_prompt = self._build_batch_plan_prompt(tasks)
            else:
                plan_prompt = self._build_plan_prompt(tasks[0])
            plan_result = self.claude.run(plan_prompt, working_dir=self.config.target_dir)
            total_cost += plan_result.cost_usd
            total_duration += plan_result.duration_seconds

            if not plan_result.success:
                logger.warning("Claude planning failed: %s", plan_result.error)
                self.git.rollback(snapshot)
                self.state.record_cycle(self._make_cycle_record(
                    tasks,
                    success=False,
                    cost_usd=total_cost,
                    duration_seconds=total_duration,
                    error=f"Planning failed: {plan_result.error}",
                ))
                return

            # Clean any accidental changes from planning phase
            self.git.rollback(snapshot)

            logger.info("Plan created, auto-accepting and executing...")

            # Phase 2: Execute the plan
            if is_batch:
                exec_prompt = self._build_batch_execute_prompt(tasks, plan_result.result_text)
            else:
                exec_prompt = self._build_execute_prompt(tasks[0], plan_result.result_text)
            claude_result = self.claude.run(exec_prompt, working_dir=self.config.target_dir)
            total_cost += claude_result.cost_usd
            total_duration += claude_result.duration_seconds
        else:
            prompt = self._build_prompt(tasks[0])
            claude_result = self.claude.run(prompt, working_dir=self.config.target_dir)
            total_cost = claude_result.cost_usd
            total_duration = claude_result.duration_seconds

        if not claude_result.success:
            logger.warning("Claude failed: %s", claude_result.error)
            self.git.rollback(snapshot)
            self.state.record_cycle(self._make_cycle_record(
                tasks,
                success=False,
                cost_usd=total_cost,
                duration_seconds=total_duration,
                error=claude_result.error,
            ))
            return

        # 8. Check changed files
        changed_files = self.git.get_new_changed_files(pre_existing_files)
        if not changed_files:
            logger.info("No files changed by Claude, skipping")
            self.state.record_cycle(self._make_cycle_record(
                tasks,
                success=False,
                cost_usd=total_cost,
                duration_seconds=total_duration,
                error="No files changed",
            ))
            return

        try:
            self.safety.post_claude_checks(changed_files)
        except SafetyError as e:
            logger.warning("Post-Claude safety check failed: %s", e)
            self.git.rollback(snapshot)
            self.state.record_cycle(self._make_cycle_record(
                tasks,
                success=False,
                cost_usd=total_cost,
                duration_seconds=total_duration,
                error=str(e),
            ))
            return

        # Syntax check if self-improving
        syntax_err = self._syntax_check_files(changed_files)
        if syntax_err:
            logger.warning("Syntax check failed: %s", syntax_err)
            self.git.rollback(snapshot)
            self.state.record_cycle(self._make_cycle_record(
                tasks,
                success=False,
                cost_usd=total_cost,
                duration_seconds=total_duration,
                error=syntax_err,
            ))
            return

        # 9. Validate
        validation = self.validator.validate(self.config.target_dir)

        if validation.passed:
            # 10. Commit
            if is_batch:
                commit_msg = self._build_batch_commit_message(tasks)
            else:
                commit_msg = f"[auto] {tasks[0].source}: {tasks[0].description[:80]}"
            commit_hash = self.git.commit(commit_msg, files=changed_files)
            logger.info("Cycle succeeded: %s", commit_msg.split("\n")[0])

            # Push to remote if configured
            if self.config.orchestrator.push_after_commit:
                self.git.push()

            # Mark all feedback tasks as done
            for t in tasks:
                if t.source == "feedback" and t.source_file:
                    self.feedback.mark_done(t.source_file)

            self.state.record_cycle(self._make_cycle_record(
                tasks,
                success=True,
                commit_hash=commit_hash,
                cost_usd=total_cost,
                duration_seconds=total_duration,
                validation_summary=validation.summary,
            ))
        else:
            # 11. Rollback
            logger.warning("Validation failed: %s", validation.summary)
            self.git.rollback(snapshot)
            self.state.record_cycle(self._make_cycle_record(
                tasks,
                success=False,
                cost_usd=total_cost,
                duration_seconds=total_duration,
                validation_summary=validation.summary,
                error="Validation failed",
            ))

    def run(self, once: bool = False) -> None:
        """Run the main loop. If once=True, run a single cycle and exit."""
        self._setup_signals()

        try:
            self.safety.acquire_lock()
        except SafetyError as e:
            logger.error("Cannot start: %s", e)
            return

        try:
            logger.info("Orchestrator started (once=%s)", once)
            while self._running:
                try:
                    self._cycle()
                except Exception:
                    logger.exception("Unexpected error in cycle")

                if once:
                    break

                # 13. Sleep
                logger.debug(
                    "Sleeping %ds...", self.config.orchestrator.loop_interval_seconds
                )
                # Sleep in small increments so we can respond to signals
                sleep_total = self.config.orchestrator.loop_interval_seconds
                while sleep_total > 0 and self._running:
                    time.sleep(min(1, sleep_total))
                    sleep_total -= 1

            logger.info("Orchestrator stopped")
        finally:
            self.safety.release_lock()
