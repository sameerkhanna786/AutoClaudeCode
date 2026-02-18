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

    def _pick_task(self) -> Optional[Task]:
        """Pick the highest-priority task, checking feedback first."""
        # Priority 1: developer feedback
        feedback_tasks = self.feedback.get_pending_feedback()
        for task in feedback_tasks:
            if not self.state.was_recently_attempted(task.description):
                return task

        # Auto-discovered tasks
        discovered = self.discovery.discover_all()
        for task in discovered:
            if not self.state.was_recently_attempted(task.description):
                return task

        return None

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

        # 2-5. Pick a task
        task = self._pick_task()
        if task is None:
            logger.info("No tasks found, sleeping...")
            return

        logger.info("Selected task [priority=%d]: %s", task.priority, task.description)

        # Backup orchestrator files if self-improving
        self._backup_orchestrator_files()

        # 6. Record git snapshot
        snapshot = self.git.create_snapshot()

        # 7. Invoke Claude
        prompt = self._build_prompt(task)
        claude_result = self.claude.run(prompt, working_dir=self.config.target_dir)

        if not claude_result.success:
            logger.warning("Claude failed: %s", claude_result.error)
            self.git.rollback(snapshot)
            self.state.record_cycle(CycleRecord(
                timestamp=time.time(),
                task_description=task.description,
                task_type=task.source,
                success=False,
                cost_usd=claude_result.cost_usd,
                duration_seconds=claude_result.duration_seconds,
                error=claude_result.error,
            ))
            return

        # 8. Check changed files
        changed_files = self.git.get_changed_files()
        if not changed_files:
            logger.info("No files changed by Claude, skipping")
            self.state.record_cycle(CycleRecord(
                timestamp=time.time(),
                task_description=task.description,
                task_type=task.source,
                success=False,
                cost_usd=claude_result.cost_usd,
                duration_seconds=claude_result.duration_seconds,
                error="No files changed",
            ))
            return

        try:
            self.safety.post_claude_checks(changed_files)
        except SafetyError as e:
            logger.warning("Post-Claude safety check failed: %s", e)
            self.git.rollback(snapshot)
            self.state.record_cycle(CycleRecord(
                timestamp=time.time(),
                task_description=task.description,
                task_type=task.source,
                success=False,
                cost_usd=claude_result.cost_usd,
                duration_seconds=claude_result.duration_seconds,
                error=str(e),
            ))
            return

        # Syntax check if self-improving
        syntax_err = self._syntax_check_files(changed_files)
        if syntax_err:
            logger.warning("Syntax check failed: %s", syntax_err)
            self.git.rollback(snapshot)
            self.state.record_cycle(CycleRecord(
                timestamp=time.time(),
                task_description=task.description,
                task_type=task.source,
                success=False,
                cost_usd=claude_result.cost_usd,
                duration_seconds=claude_result.duration_seconds,
                error=syntax_err,
            ))
            return

        # 9. Validate
        validation = self.validator.validate(self.config.target_dir)

        if validation.passed:
            # 10. Commit
            commit_msg = f"[auto] {task.source}: {task.description[:80]}"
            commit_hash = self.git.commit(commit_msg)
            logger.info("Cycle succeeded: %s", commit_msg)

            # Mark feedback as done if applicable
            if task.source == "feedback" and task.source_file:
                self.feedback.mark_done(task.source_file)

            self.state.record_cycle(CycleRecord(
                timestamp=time.time(),
                task_description=task.description,
                task_type=task.source,
                success=True,
                commit_hash=commit_hash,
                cost_usd=claude_result.cost_usd,
                duration_seconds=claude_result.duration_seconds,
                validation_summary=validation.summary,
            ))
        else:
            # 11. Rollback
            logger.warning("Validation failed: %s", validation.summary)
            self.git.rollback(snapshot)
            self.state.record_cycle(CycleRecord(
                timestamp=time.time(),
                task_description=task.description,
                task_type=task.source,
                success=False,
                cost_usd=claude_result.cost_usd,
                duration_seconds=claude_result.duration_seconds,
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
