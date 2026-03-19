"""Main orchestration loop tying all components together."""

from __future__ import annotations

import concurrent.futures
import copy
import logging
import os
import shutil
import signal
import time
from pathlib import Path
from typing import List, Optional

from claude_runner import ClaudeRunner, ClaudeResult
from config_schema import Config
from cycle_state import CycleState, CycleStateWriter
from feedback import FeedbackManager
from git_manager import GitManager, Snapshot
from model_resolver import resolve_model_id
from safety import SafetyError, SafetyGuard, GracefulDegradation
from state import CycleRecord, StateManager
from task_discovery import Task, TaskDiscovery
from validator import ValidationResult, Validator
from agent_pipeline import AgentPipeline
from structured_logging import apply_json_logging
from cost_predictor import check_cost_budget
from notifications import NotificationManager
from shared import (
    format_task_list, syntax_check_files, gather_tasks,
    format_validation_errors as shared_format_validation_errors,
    build_commit_message, build_batch_commit_message,
    clean_description, extract_file_names,
    TASK_TYPE_INSTRUCTIONS,
    build_task_prompt as shared_build_task_prompt,
    build_plan_prompt as shared_build_plan_prompt,
    build_execute_prompt as shared_build_execute_prompt,
    build_retry_prompt as shared_build_retry_prompt,
)
from task_queue import TaskApprovalQueue
from telemetry import compute_metrics

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, config: Config):
        self.config = config

        # target_dir validation is now handled by validate_config() at config load time

        # Fix 11: Resolve model alias to actual model ID with retry
        resolved = resolve_model_id(
            model_alias=config.claude.model,
            claude_command=config.claude.command,
        )
        if resolved:
            config.claude.resolved_model = resolved
        else:
            # Retry once
            resolved = resolve_model_id(
                model_alias=config.claude.model,
                claude_command=config.claude.command,
            )
            if resolved:
                config.claude.resolved_model = resolved
            else:
                logger.error(
                    "Could not resolve model '%s' after retrying. "
                    "Using alias directly — Claude invocations may fail.",
                    config.claude.model,
                )

        self.state = StateManager(config)
        self.safety = SafetyGuard(config, self.state)

        # Apply structured JSON logging if configured
        if config.logging.format == "json":
            apply_json_logging()

        # Create runner (provider-agnostic: Claude CLI, OpenAI, or Gemini)
        from provider_runner import create_runner
        self.claude = create_runner(config)
        self.git = GitManager(config.target_dir)
        self.validator = Validator(config)
        self.discovery = TaskDiscovery(config, state_manager=self.state)
        self.feedback = FeedbackManager(config)
        self.cycle_state = CycleStateWriter(str(Path(config.paths.history_file).parent))
        self.notifier = NotificationManager(config.notifications)
        self._degradation = GracefulDegradation(config)
        self._task_queue = TaskApprovalQueue(str(Path(config.paths.state_dir)))
        self._running = True
        self._consecutive_exceptions = 0
        self._backoff_seconds = 0
        self._active_pipeline: Optional[AgentPipeline] = None
        self._successful_commits = 0
        self._consecutive_empty_plans = 0  # Track planning failures to skip when futile
        self._cycle_counter = 0  # For periodic summary notifications
        self._session_manager = None
        if config.orchestrator.session_recovery:
            from session_manager import SessionManager
            self._session_manager = SessionManager(str(Path(config.paths.state_dir)))

        # Clean stale agent workspace from previous runs
        workspace_dir = Path(self.config.target_dir) / self.config.paths.agent_workspace_dir
        if workspace_dir.exists():
            for child in workspace_dir.iterdir():
                if child.is_file():
                    try:
                        child.unlink()
                    except OSError:
                        pass

    def _setup_signals(self) -> None:
        """Register signal handlers for graceful shutdown."""
        self._signal_received = False

        def handler(signum, frame):
            if self._signal_received:
                return  # Avoid redundant handling on repeated delivery
            self._signal_received = True
            logger.info("Received signal %d, shutting down gracefully...", signum)
            self._running = False
            self.claude.terminate()
            pipeline = self._active_pipeline
            if pipeline is not None:
                pipeline.terminate()

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _build_prompt(self, task: Task) -> str:
        """Build the Claude prompt for a given task."""
        return shared_build_task_prompt([task], self.config.safety.protected_files)

    def _build_plan_prompt(self, task: Task) -> str:
        """Build a planning-only prompt for a task."""
        return shared_build_plan_prompt([task], self.config.safety.protected_files)

    def _build_execute_prompt(self, task: Task, plan: str) -> str:
        """Build an execution prompt with a pre-approved plan."""
        return shared_build_execute_prompt(
            [task], plan, self.config.safety.protected_files,
        )

    def _pick_task(self) -> Optional[Task]:
        """Pick the highest-priority task, checking feedback first."""
        tasks = self._gather_tasks()
        return tasks[0] if tasks else None

    def _gather_tasks(self) -> List[Task]:
        """Gather all eligible tasks, respecting batch_mode and adaptive sizing."""
        dashboard_active = self._task_queue.is_dashboard_active()
        tasks = gather_tasks(
            self.config, self.feedback, self.state, self.discovery,
            dashboard_active=dashboard_active,
            task_approval_queue=self._task_queue,
        )

        if not self.config.orchestrator.batch_mode:
            return tasks[:1]

        batch_size = self.state.compute_adaptive_batch_size()
        if batch_size < 1:
            logger.warning("Adaptive batch size was %d, clamping to 1", batch_size)
            batch_size = 1

        # Apply graceful degradation factor to batch size.
        # Use the already-computed degradation level from _cycle() to avoid
        # redundant history loads via get_cycle_count_last_hour/get_total_cost.
        if self._degradation.is_degraded:
            level = self._degradation.degradation_level
            batch_size_factor = {0: 1.0, 1: 0.75, 2: 0.5, 3: 0.25}.get(level, 1.0)
            adjusted = max(1, int(batch_size * batch_size_factor))
            if adjusted < batch_size:
                logger.info(
                    "Degradation reduced batch size: %d -> %d (level=%d, factor=%.2f)",
                    batch_size, adjusted, level, batch_size_factor,
                )
                batch_size = adjusted

        logger.info("Adaptive batch size: %d", batch_size)
        return tasks[:batch_size]

    def _format_task_list(self, tasks: List[Task]) -> str:
        """Format tasks as a numbered list with source tags and context."""
        return format_task_list(tasks)

    def _build_batch_plan_prompt(self, tasks: List[Task]) -> str:
        """Build a batch planning prompt for multiple tasks."""
        return shared_build_plan_prompt(tasks, self.config.safety.protected_files)

    def _build_batch_execute_prompt(self, tasks: List[Task], plan: str) -> str:
        """Build a batch execution prompt with a pre-approved plan."""
        return shared_build_execute_prompt(
            tasks, plan, self.config.safety.protected_files,
        )

    def _build_batch_prompt(self, tasks: List[Task]) -> str:
        """Build a single-shot prompt for batch tasks (no plan phase)."""
        return shared_build_task_prompt(tasks, self.config.safety.protected_files)

    # ------------------------------------------------------------------
    # Commit-message helpers
    # ------------------------------------------------------------------

    def _build_commit_message(self, task: Task) -> str:
        """Build a conventional, human-style commit message for a single task."""
        return build_commit_message(task)

    def _build_batch_commit_message(self, tasks: List[Task]) -> str:
        """Build a natural commit message summarizing a batch of tasks."""
        return build_batch_commit_message(tasks)

    def _format_validation_errors(self, validation: ValidationResult) -> str:
        """Extract failure details from ValidationResult for the retry prompt."""
        include_full = self.config.orchestrator.retry_include_full_output
        return shared_format_validation_errors(validation, include_full=include_full)

    def _build_retry_prompt(self, tasks: List[Task], validation: ValidationResult,
                            attempt: int, max_attempts: int) -> str:
        """Build a prompt for retrying after validation failure."""
        errors = self._format_validation_errors(validation)
        task_history = self.state.get_task_success_history(
            tasks[0].description,
            task_key=tasks[0].task_key,
        )
        return shared_build_retry_prompt(
            tasks, errors, self.config.safety.protected_files,
            attempt=attempt, max_attempts=max_attempts,
            task_history=task_history,
        )

    def _validate_with_retries(
        self, tasks: List[Task], snapshot, pre_existing_files,
        total_cost: float, total_duration: float,
        is_batch: bool, extra_record_kwargs: Optional[dict] = None,
    ) -> None:
        """Validate changes, retrying with Claude on failure.

        On validation failure, re-invokes Claude with the failure output so it can
        fix the issue in-place. Rollback only happens if all fix attempts are exhausted
        or a non-retryable error occurs (safety check, syntax error).

        Uses a single loop counter (attempt) to avoid dual-counter confusion.
        Clamps max_retries to [0, 50] to prevent runaway loops from misconfiguration.
        """
        max_retries = max(0, min(self.config.orchestrator.max_validation_retries, 50))
        extra = extra_record_kwargs or {}
        retry_count = 0

        # Fast path: if no files changed before we even enter the retry loop,
        # skip the loop entirely to avoid unnecessary overhead.
        changed_files = self.git.get_new_changed_files(pre_existing_files)
        if not changed_files:
            logger.info("No files changed, skipping validation")
            self.state.record_cycle(self._make_cycle_record(
                tasks, success=False,
                cost_usd=total_cost, duration_seconds=total_duration,
                error="No files changed",
                validation_retry_count=0, **extra,
            ))
            return

        for attempt in range(max_retries + 1):
            # Re-capture changed files (may differ after retry)
            changed_files = self.git.get_new_changed_files(pre_existing_files)
            if not changed_files:
                logger.info("No files changed, skipping")
                self.state.record_cycle(self._make_cycle_record(
                    tasks, success=False,
                    cost_usd=total_cost, duration_seconds=total_duration,
                    error="No files changed",
                    validation_retry_count=retry_count, **extra,
                ))
                return

            # Safety checks (non-retryable — immediate rollback)
            try:
                self.safety.post_claude_checks(changed_files)
            except SafetyError as e:
                logger.warning("Post-Claude safety check failed: %s", e)
                self.git.rollback(snapshot, allowed_dirty=pre_existing_files)
                self.state.record_cycle(self._make_cycle_record(
                    tasks, success=False,
                    cost_usd=total_cost, duration_seconds=total_duration,
                    error=str(e),
                    validation_retry_count=retry_count, **extra,
                ))
                return

            # Syntax check (non-retryable)
            syntax_err = self._syntax_check_files(changed_files)
            if syntax_err:
                logger.warning("Syntax check failed: %s", syntax_err)
                self.git.rollback(snapshot, allowed_dirty=pre_existing_files)
                self.state.record_cycle(self._make_cycle_record(
                    tasks, success=False,
                    cost_usd=total_cost, duration_seconds=total_duration,
                    error=syntax_err,
                    validation_retry_count=retry_count, **extra,
                ))
                return

            # Validate
            if self.config.validation.incremental_tests:
                validation = self.validator.validate_incremental(
                    self.config.target_dir, changed_files,
                )
            else:
                validation = self.validator.validate(self.config.target_dir)

            if validation.passed:
                # LLM Judge evaluation (before commit)
                if self.config.judges.enabled:
                    from llm_judges import JudgePanel
                    panel = JudgePanel(self.config)
                    diff_text = self.git.get_diff()
                    panel_result = panel.evaluate(
                        changed_files, diff_text,
                        tasks[0].description,
                    )
                    total_cost += panel_result.total_cost_usd
                    total_duration += panel_result.total_duration_seconds
                    if not panel_result.passed:
                        if self.config.judges.fail_action == "retry" and attempt < max_retries:
                            retry_count += 1
                            retry_prompt = shared_build_retry_prompt(
                                tasks, panel_result.blocking_feedback,
                                self.config.safety.protected_files,
                                attempt=attempt + 1, max_attempts=max_retries + 1,
                            )
                            retry_result = self._run_claude_with_timeout(retry_prompt)
                            total_cost += retry_result.cost_usd
                            total_duration += retry_result.duration_seconds
                            continue  # re-validate
                        elif self.config.judges.fail_action == "rollback":
                            self.git.rollback(snapshot, allowed_dirty=pre_existing_files)
                            self.state.record_cycle(self._make_cycle_record(
                                tasks, success=False,
                                cost_usd=total_cost, duration_seconds=total_duration,
                                error="LLM judges rejected changes",
                                validation_retry_count=retry_count, **extra,
                            ))
                            return
                        # else "warn": log and continue to commit
                        logger.warning(
                            "LLM judges failed but fail_action=%s, continuing",
                            self.config.judges.fail_action,
                        )

                # Commit
                if is_batch:
                    commit_msg = self._build_batch_commit_message(tasks)
                else:
                    commit_msg = self._build_commit_message(tasks[0])

                commit_hash = self.git.commit(commit_msg, files=changed_files)

                if commit_hash is None:
                    # Git command itself failed (distinct from "nothing staged")
                    logger.error("git commit command failed")
                    self.git.rollback(snapshot, allowed_dirty=pre_existing_files)
                    self.state.record_cycle(self._make_cycle_record(
                        tasks, success=False,
                        cost_usd=total_cost, duration_seconds=total_duration,
                        error="git commit command failed",
                        validation_retry_count=retry_count, **extra,
                    ))
                    return

                if not commit_hash:
                    # Nothing was staged despite changed files detected earlier
                    logger.error("Commit failed: no staged changes")
                    self.git.rollback(snapshot, allowed_dirty=pre_existing_files)
                    self.state.record_cycle(self._make_cycle_record(
                        tasks, success=False,
                        cost_usd=total_cost, duration_seconds=total_duration,
                        error="Commit failed (no staged changes)",
                        validation_retry_count=retry_count, **extra,
                    ))
                    return

                logger.info("Cycle succeeded: %s", commit_msg.split("\n")[0])

                # Periodic git gc to clean up loose objects
                self._successful_commits += 1
                if self._successful_commits % self.config.orchestrator.gc_interval == 0:
                    self.git.gc_auto()

                # Run config tuner
                try:
                    from config_tuner import ConfigTuner
                    tuner = ConfigTuner(str(Path(self.config.paths.state_dir)))
                    recs = tuner.analyze(self.state.load_history(), self.config)
                    if recs:
                        tuner.save_recommendations(recs)
                except Exception as e:
                    logger.debug("Config tuner failed: %s", e)

                if self.config.orchestrator.push_after_commit:
                    push_ok = self.git.push()
                    if not push_ok:
                        logger.error(
                            "Push failed for commit %s — local commits may diverge from remote",
                            commit_hash[:8],
                        )
                else:
                    push_ok = None

                for t in tasks:
                    if t.source == "feedback" and t.source_file:
                        self.feedback.mark_done(t.source_file)

                self.state.record_cycle(self._make_cycle_record(
                    tasks, success=True,
                    commit_hash=commit_hash,
                    cost_usd=total_cost, duration_seconds=total_duration,
                    validation_summary=validation.summary,
                    validation_retry_count=retry_count,
                    push_succeeded=push_ok,
                    **extra,
                ))
                self.notifier.notify("cycle_success", {
                    "tasks": [t.description for t in tasks],
                    "commit": commit_hash,
                    "cost_usd": total_cost,
                })
                # Clear session on success
                if self._session_manager:
                    self._session_manager.clear_session()
                return
            if attempt < max_retries:
                # Cost guard: check accumulated cost against hourly budget
                hourly_cost = self.state.get_total_cost(lookback_seconds=3600)
                cost_limit = self.config.safety.max_cost_usd_per_hour
                if hourly_cost + total_cost >= cost_limit * 0.9:
                    logger.warning(
                        "Cost guard: $%.2f accumulated (limit $%.2f), aborting retries",
                        hourly_cost + total_cost, cost_limit,
                    )
                    self.git.rollback(snapshot, allowed_dirty=pre_existing_files)
                    self.state.record_cycle(self._make_cycle_record(
                        tasks, success=False,
                        cost_usd=total_cost, duration_seconds=total_duration,
                        validation_summary=validation.summary,
                        error="Validation failed; cost guard prevented retry",
                        validation_retry_count=retry_count, **extra,
                    ))
                    return

                retry_count += 1
                logger.info(
                    "Validation failed (attempt %d/%d), retrying with failure output...",
                    attempt + 1, max_retries + 1,
                )
                retry_prompt = self._build_retry_prompt(
                    tasks, validation, attempt + 1, max_retries + 1,
                )
                retry_result = self._run_claude_with_timeout(retry_prompt)
                total_cost += retry_result.cost_usd
                total_duration += retry_result.duration_seconds
                self.cycle_state.update(
                    accumulated_cost=total_cost,
                    retry_count=retry_count,
                    phase="retrying",
                )

                if not retry_result.success:
                    logger.warning("Retry Claude invocation failed: %s", retry_result.error)
                    self.git.rollback(snapshot, allowed_dirty=pre_existing_files)
                    self.state.record_cycle(self._make_cycle_record(
                        tasks, success=False,
                        cost_usd=total_cost, duration_seconds=total_duration,
                        validation_summary=validation.summary,
                        error=f"Retry failed: {retry_result.error}",
                        validation_retry_count=retry_count, **extra,
                    ))
                    return
                # Loop back to re-validate
            else:
                # All attempts exhausted
                logger.warning("Validation failed after %d attempts: %s",
                               max_retries + 1, validation.summary)
                self.git.rollback(snapshot, allowed_dirty=pre_existing_files)
                self.state.record_cycle(self._make_cycle_record(
                    tasks, success=False,
                    cost_usd=total_cost, duration_seconds=total_duration,
                    validation_summary=validation.summary,
                    error="Validation failed",
                    validation_retry_count=retry_count, **extra,
                ))
                self.notifier.notify("cycle_failure", {
                    "tasks": [t.description for t in tasks],
                    "error": validation.summary,
                    "retry_count": retry_count,
                })
                return

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
            batch_size=len(tasks),
            task_keys=[t.task_key for t in tasks],
            pipeline_mode=kwargs.get("pipeline_mode", ""),
            pipeline_revision_count=kwargs.get("pipeline_revision_count", 0),
            pipeline_review_approved=kwargs.get("pipeline_review_approved", True),
            validation_retry_count=kwargs.get("validation_retry_count", 0),
            push_succeeded=kwargs.get("push_succeeded", None),
            task_source_files=[t.source_file or "" for t in tasks],
            task_line_numbers=[t.line_number for t in tasks],
        )
        return record

    def _syntax_check_files(self, changed_files: List[str]) -> Optional[str]:
        """If self_improve is on, syntax-check any modified .py files."""
        if not self.config.orchestrator.self_improve:
            return None
        return syntax_check_files(changed_files, self.config.target_dir)

    def _backup_orchestrator_files(self) -> None:
        """If self_improve is on, back up key orchestrator files before a cycle.

        Dynamically discovers all .py files in the project root (excluding
        main.py, test files, and non-project files) to ensure new modules
        are always included.
        """
        if not self.config.orchestrator.self_improve:
            return

        backup_dir = Path(self.config.paths.backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)

        target = Path(self.config.target_dir)
        for py_file in target.glob("*.py"):
            # Skip main.py (protected), test files, and setup files
            if py_file.name in ("main.py", "setup.py"):
                continue
            if py_file.name.startswith("test_"):
                continue
            try:
                dst = backup_dir / py_file.name
                shutil.copy2(str(py_file), str(dst))
            except OSError as e:
                logger.warning("Failed to backup %s: %s", py_file.name, e)

    def _run_claude_with_timeout(self, prompt: str, runner: Optional[ClaudeRunner] = None) -> ClaudeResult:
        """Run Claude CLI with a cycle-level timeout safety net.

        Wraps self.claude.run() in a thread pool with a configurable timeout
        to prevent indefinite hangs even if the subprocess timeout fails.
        On timeout, actively terminates the child process and waits for the
        thread to clean up before shutting down the executor.
        """
        timeout = self.config.orchestrator.cycle_timeout_seconds
        actual_runner = runner or self.claude
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        timed_out = False
        try:
            future = executor.submit(actual_runner.run, prompt)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                timed_out = True
                logger.warning(
                    "Claude CLI cycle timeout fired after %ds — killing subprocess", timeout,
                )
                actual_runner.terminate()
                future.cancel()
                # Short secondary wait for the thread to finish
                try:
                    future.result(timeout=30)
                except (concurrent.futures.TimeoutError, concurrent.futures.CancelledError, Exception):
                    logger.error(
                        "Claude thread still alive 30s after terminate — "
                        "continuing without blocking"
                    )
                return ClaudeResult(
                    success=False,
                    error=f"Cycle timeout after {timeout}s (Claude CLI hung)",
                )
        finally:
            # Use wait=True on the normal path so the thread is joined and
            # its resources are reclaimed.  Only use wait=False when timed
            # out and the thread may be stuck, to avoid blocking the caller.
            executor.shutdown(wait=not timed_out)

    def _cycle(self) -> None:
        """Run a single orchestration cycle."""
        # 1. Pre-flight safety checks
        try:
            self.safety.pre_flight_checks()
        except SafetyError as e:
            logger.warning("Pre-flight check failed: %s", e)
            self.notifier.notify("safety_error", {"error": str(e)})
            return

        # Check for graceful degradation (throttle instead of hard-stop)
        cycles_per_hour = self.state.get_cycle_count_last_hour()
        cost_per_hour = self.state.get_total_cost(lookback_seconds=3600)
        degradation = self._degradation.check_and_adjust(cycles_per_hour, cost_per_hour)
        if degradation["degraded"]:
            logger.warning(
                "Graceful degradation active (level %d): %s",
                degradation["level"], degradation["reason"],
            )

        # 2-5. Gather tasks
        # Clean up stale pending approval tasks periodically
        self._task_queue.clear_stale(max_age=3600)

        tasks = self._gather_tasks()
        if not tasks:
            # Check if tasks are pending approval
            pending_count = self._task_queue.pending_count()
            if pending_count > 0:
                logger.info(
                    "Waiting for task approval (%d tasks pending)", pending_count,
                )
                self.cycle_state.update(
                    phase="waiting_approval",
                    pending_approval_count=pending_count,
                )
                return

            dc = self.config.discovery
            enabled_methods = []
            if dc.enable_test_failures:
                enabled_methods.append("test_failures")
            if dc.enable_lint_errors:
                enabled_methods.append("lint_errors")
            if dc.enable_todos:
                enabled_methods.append("todos")
            if dc.enable_coverage:
                enabled_methods.append("coverage")
            if dc.enable_claude_ideas:
                enabled_methods.append("claude_ideas")
            if dc.enable_quality_review:
                enabled_methods.append("quality_review")
            has_feedback = any(
                f.is_file() and f.suffix in (".md", ".txt")
                for f in self.feedback.feedback_dir.iterdir()
            ) if self.feedback.feedback_dir.exists() else False
            if not enabled_methods and not has_feedback:
                logger.warning(
                    "No tasks found: no discovery methods enabled and no pending feedback"
                )
            else:
                logger.info(
                    "No actionable tasks found (all may have been recently attempted). "
                    "Enabled methods: %s, pending feedback: %s",
                    ", ".join(enabled_methods) if enabled_methods else "none",
                    "yes" if has_feedback else "no",
                )
            return

        is_batch = len(tasks) > 1 and self.config.orchestrator.batch_mode

        if is_batch:
            logger.info("Selected %d tasks for batch processing", len(tasks))
            for i, t in enumerate(tasks, 1):
                logger.info("  Task %d [priority=%d]: %s", i, t.priority, t.description)
        else:
            task = tasks[0]
            logger.info("Selected task [priority=%d]: %s", task.priority, task.description)

        # Write cycle state: task selected
        self.cycle_state.write(CycleState(
            phase="task_selected",
            task_description=tasks[0].description,
            task_type=tasks[0].source,
            task_descriptions=[t.description for t in tasks],
            started_at=time.time(),
            batch_size=len(tasks),
        ))

        # Save session state for crash recovery
        if self._session_manager:
            from session_manager import SessionState
            session = SessionState(
                session_id=self._session_manager.create_session_id(),
                started_at=time.time(),
                tasks=[{"task_key": t.task_key, "task_id": t.task_id,
                        "description": t.description, "source": t.source}
                       for t in tasks],
                phase="task_selected",
            )
            self._session_manager.save_session(session)

        try:
            # Backup orchestrator files if self-improving
            self._backup_orchestrator_files()

            # 6. Record git snapshot
            snapshot = self.git.create_snapshot()
            pre_existing_files = self.git.capture_worktree_state()

            # Cost prediction: estimate whether we can afford this cycle
            cost_ok, est_cost, remaining = check_cost_budget(
                tasks, self.config, self.state,
            )
            if not cost_ok:
                logger.warning(
                    "Skipping cycle: estimated cost $%.4f exceeds remaining "
                    "budget $%.4f",
                    est_cost, remaining,
                )
                self.notifier.notify("cost_limit_exceeded", {
                    "estimated_cost": est_cost,
                    "remaining_budget": remaining,
                })
                return

            # Multi-agent pipeline dispatch
            if self.config.agent_pipeline.enabled:
                self._cycle_multi_agent(tasks, snapshot, pre_existing_files, is_batch)
                return

            # 7. Invoke Claude (with optional plan-then-execute)
            total_cost = 0.0
            total_duration = 0.0

            # Skip planning if it has failed consecutively — go straight to
            # direct execution to avoid wasting API calls on empty plans.
            use_planning = self.config.orchestrator.plan_changes
            if use_planning and self._consecutive_empty_plans >= 3:
                logger.info(
                    "Skipping planning phase: %d consecutive empty plans — "
                    "using direct execution this cycle",
                    self._consecutive_empty_plans,
                )
                use_planning = False

            if use_planning:
                # Phase 1: Plan (scale turns with batch size)
                self.cycle_state.update(phase="planning")
                base_planning_turns = self.config.orchestrator.planning_max_turns
                if is_batch and len(tasks) > 1:
                    # Extra turns per additional task to give Claude time to
                    # read files and formulate a plan for each task.
                    effective_turns = base_planning_turns + (len(tasks) - 1) * 2
                    # Cap at the main max_turns to avoid runaway
                    effective_turns = min(effective_turns, self.config.claude.max_turns)
                else:
                    effective_turns = base_planning_turns
                # Create a separate config/runner for planning to avoid
                # mutating self.config.claude.max_turns (race condition).
                plan_config = copy.deepcopy(self.config)
                plan_config.claude.max_turns = effective_turns
                from provider_runner import create_runner
                plan_runner = create_runner(plan_config)
                logger.debug(
                    "Planning with max_turns=%d (base=%d, batch_size=%d)",
                    effective_turns, base_planning_turns, len(tasks),
                )
                if is_batch:
                    plan_prompt = self._build_batch_plan_prompt(tasks)
                else:
                    plan_prompt = self._build_plan_prompt(tasks[0])
                plan_result = self._run_claude_with_timeout(plan_prompt, runner=plan_runner)
                total_cost += plan_result.cost_usd
                total_duration += plan_result.duration_seconds
                self.cycle_state.update(accumulated_cost=total_cost)

                if not plan_result.success:
                    logger.warning("Claude planning failed: %s", plan_result.error)
                    self.git.rollback(snapshot, allowed_dirty=pre_existing_files)
                    self.state.record_cycle(self._make_cycle_record(
                        tasks,
                        success=False,
                        cost_usd=total_cost,
                        duration_seconds=total_duration,
                        error=f"Planning failed: {plan_result.error}",
                    ))
                    return

                # Clean any accidental changes from planning phase
                self.git.rollback(snapshot, allowed_dirty=pre_existing_files)

                # Check if planning produced a usable plan text.  When the
                # Claude response has subtype=error_max_turns or is missing
                # the "result" field, result_text is empty.  Fall back to
                # direct execution without a plan to avoid wasting an API
                # call on an empty PLAN TO EXECUTE section.
                if not plan_result.result_text.strip():
                    self._consecutive_empty_plans += 1
                    logger.warning(
                        "Planning phase returned empty result_text "
                        "(error=%s, consecutive=%d) — falling back to direct execution",
                        plan_result.error or "none",
                        self._consecutive_empty_plans,
                    )
                    self.cycle_state.update(phase="executing")
                    if is_batch:
                        exec_prompt = self._build_batch_prompt(tasks)
                    else:
                        exec_prompt = self._build_prompt(tasks[0])
                else:
                    self._consecutive_empty_plans = 0  # Reset on successful plan
                    logger.info("Plan created, auto-accepting and executing...")

                    # Phase 2: Execute the plan
                    self.cycle_state.update(phase="executing")
                    if is_batch:
                        exec_prompt = self._build_batch_execute_prompt(tasks, plan_result.result_text)
                    else:
                        exec_prompt = self._build_execute_prompt(tasks[0], plan_result.result_text)

                claude_result = self._run_claude_with_timeout(exec_prompt)
                total_cost += claude_result.cost_usd
                total_duration += claude_result.duration_seconds
                self.cycle_state.update(accumulated_cost=total_cost)
            else:
                self.cycle_state.update(phase="executing")
                if is_batch:
                    prompt = self._build_batch_prompt(tasks)
                else:
                    prompt = self._build_prompt(tasks[0])
                claude_result = self._run_claude_with_timeout(prompt)
                total_cost += claude_result.cost_usd
                total_duration += claude_result.duration_seconds
                self.cycle_state.update(accumulated_cost=total_cost)

            if not claude_result.success:
                logger.warning("Claude failed: %s", claude_result.error)
                self.git.rollback(snapshot, allowed_dirty=pre_existing_files)
                self.state.record_cycle(self._make_cycle_record(
                    tasks,
                    success=False,
                    cost_usd=total_cost,
                    duration_seconds=total_duration,
                    error=claude_result.error,
                ))
                return

            # Context isolation: warn if context window usage is high
            if self.config.orchestrator.context_isolation and claude_result.success:
                if claude_result.context_window_pct > self.config.orchestrator.max_context_pct:
                    logger.warning(
                        "Context window %.1f%% used (threshold %.1f%%)",
                        claude_result.context_window_pct,
                        self.config.orchestrator.max_context_pct,
                    )

            # Smart zone: auto-split tasks when context is exhausted
            if self.config.orchestrator.smart_split and claude_result.success:
                from context_monitor import ContextMonitor, write_split_tasks_as_feedback
                monitor = ContextMonitor(self.config)
                signals = monitor.extract_signals(claude_result)
                if monitor.should_split(signals):
                    split_tasks = monitor.generate_split_tasks(tasks[0], claude_result)
                    if split_tasks:
                        written = write_split_tasks_as_feedback(
                            split_tasks, self.config.paths.feedback_dir,
                        )
                        logger.info(
                            "Smart split: wrote %d follow-up tasks to feedback/",
                            written,
                        )

            # 8-11. Validate with retries, commit or rollback
            self.cycle_state.update(phase="validating")
            self._validate_with_retries(
                tasks=tasks, snapshot=snapshot,
                pre_existing_files=pre_existing_files,
                total_cost=total_cost, total_duration=total_duration,
                is_batch=is_batch,
            )
        finally:
            self.cycle_state.clear()

    def _cycle_multi_agent(
        self, tasks: List[Task], snapshot: Snapshot,
        pre_existing_files: set, is_batch: bool,
    ) -> None:
        """Run a cycle using the multi-agent pipeline."""
        logger.info("Running multi-agent pipeline")
        self.cycle_state.update(phase="pipeline")
        pipeline = AgentPipeline(self.config, cycle_state=self.cycle_state)
        self._active_pipeline = pipeline

        # Wrap pipeline.run() in a thread pool with cycle-level timeout
        timeout = self.config.orchestrator.cycle_timeout_seconds
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        timed_out = False
        try:
            future = executor.submit(pipeline.run, tasks, self.git.rollback, snapshot)
            try:
                pipeline_result = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                timed_out = True
                logger.warning(
                    "Multi-agent pipeline cycle timeout fired after %ds", timeout,
                )
                pipeline.terminate()
                future.cancel()
                # Short secondary wait
                try:
                    future.result(timeout=30)
                except (concurrent.futures.TimeoutError, concurrent.futures.CancelledError, Exception):
                    logger.error(
                        "Pipeline thread still alive 30s after terminate — "
                        "continuing without blocking"
                    )
                self.git.rollback(snapshot, allowed_dirty=pre_existing_files)
                self.state.record_cycle(self._make_cycle_record(
                    tasks,
                    success=False,
                    cost_usd=0.0,
                    duration_seconds=float(timeout),
                    error=f"Pipeline cycle timeout after {timeout}s",
                    pipeline_mode="multi_agent",
                ))
                return
        finally:
            executor.shutdown(wait=not timed_out)
            self._active_pipeline = None

        total_cost = pipeline_result.total_cost_usd
        total_duration = pipeline_result.total_duration_seconds

        if not pipeline_result.success:
            logger.warning("Multi-agent pipeline failed: %s", pipeline_result.error)
            self.git.rollback(snapshot, allowed_dirty=pre_existing_files)
            self.state.record_cycle(self._make_cycle_record(
                tasks,
                success=False,
                cost_usd=total_cost,
                duration_seconds=total_duration,
                error=pipeline_result.error,
                pipeline_mode="multi_agent",
                pipeline_revision_count=pipeline_result.revision_count,
                pipeline_review_approved=pipeline_result.final_review_approved,
            ))
            return

        # Steps 8-11: validate with retries, commit or rollback
        self._validate_with_retries(
            tasks=tasks, snapshot=snapshot,
            pre_existing_files=pre_existing_files,
            total_cost=total_cost, total_duration=total_duration,
            is_batch=is_batch,
            extra_record_kwargs={
                "pipeline_mode": "multi_agent",
                "pipeline_revision_count": pipeline_result.revision_count,
                "pipeline_review_approved": pipeline_result.final_review_approved,
            },
        )

    def _send_periodic_summary(self) -> None:
        """Send a periodic summary notification every N cycles.

        Uses telemetry.compute_metrics() to generate a structured report
        of recent orchestrator performance.
        """
        interval = self.config.notifications.events.summary_interval_cycles
        if interval <= 0 or self._cycle_counter % interval != 0:
            return

        records = self.state.load_history()
        # Use a fixed 1-hour lookback; interval is a cycle count, not hours
        metrics = compute_metrics(records, lookback_seconds=3600)

        summary = {
            "total_cycles": metrics["total_cycles"],
            "successes": metrics["successes"],
            "failures": metrics["failures"],
            "success_rate": f"{metrics['success_rate']}%",
            "total_cost_usd": f"${metrics['cost']['total']:.2f}",
            "top_task_types": list(metrics.get("type_breakdown", {}).keys())[:5],
            "degradation_level": self._degradation.degradation_level,
        }
        self.notifier.notify("periodic_summary", summary)
        logger.info(
            "Periodic summary (cycle %d): %d cycles, %d/%d pass/fail, $%.2f cost",
            self._cycle_counter, metrics["total_cycles"],
            metrics["successes"], metrics["failures"],
            metrics["cost"]["total"],
        )

    def run(self, once: bool = False) -> None:
        """Run the main loop. If once=True, run a single cycle and exit."""
        if self.config.parallel.enabled:
            from coordinator import ParallelCoordinator
            coordinator = ParallelCoordinator(self.config)
            try:
                coordinator.run(once=once)
            finally:
                self.notifier.shutdown()
            return

        self._setup_signals()

        try:
            self.safety.acquire_lock()
        except SafetyError as e:
            logger.error("Cannot start: %s", e)
            return

        try:
            logger.info("Orchestrator started (once=%s)", once)

            # Log strategy performance from recent history
            perf_report = self.state.get_strategy_performance_report()
            logger.info(perf_report)

            # Check for incomplete session from a previous crash
            if self._session_manager and self._session_manager.has_incomplete_session():
                prev_session = self._session_manager.load_session()
                if prev_session:
                    logger.info(
                        "Resuming incomplete session %s (phase=%s, cost=$%.2f)",
                        prev_session.session_id, prev_session.phase,
                        prev_session.total_cost_usd,
                    )
                    self._session_manager.clear_session()

            while self._running:
                try:
                    self._cycle()
                    self._cycle_counter += 1
                    self._send_periodic_summary()
                    # Reset backoff on successful cycle (no exception)
                    self._consecutive_exceptions = 0
                    self._backoff_seconds = 0
                except (FileNotFoundError, PermissionError) as e:
                    # Likely permanent errors — don't retry at normal pace
                    self._consecutive_exceptions += 1
                    logger.error(
                        "Likely permanent error in cycle (%s): %s "
                        "(consecutive errors: %d)",
                        type(e).__name__, e, self._consecutive_exceptions,
                    )
                    if self._consecutive_exceptions >= 3:
                        logger.error(
                            "Pausing after %d consecutive %s errors — "
                            "check config and permissions",
                            self._consecutive_exceptions, type(e).__name__,
                        )
                        self._backoff_seconds = min(
                            self._backoff_seconds * 2 if self._backoff_seconds else 60,
                            600,
                        )
                except Exception:
                    self._consecutive_exceptions += 1
                    self._backoff_seconds = min(
                        self._backoff_seconds * 2 if self._backoff_seconds else 30,
                        600,
                    )
                    logger.exception(
                        "Unexpected error in cycle (consecutive: %d, backoff: %ds)",
                        self._consecutive_exceptions, self._backoff_seconds,
                    )

                if once:
                    break

                # 13. Sleep (with exponential backoff if exceptions are recurring)
                sleep_total = self.config.orchestrator.loop_interval_seconds + self._backoff_seconds
                # Apply graceful degradation sleep multiplier (using the
                # level already computed by _cycle to avoid redundant
                # history scans).
                if self._degradation.is_degraded:
                    sleep_total = int(sleep_total * self._degradation.current_sleep_multiplier)
                if self._backoff_seconds > 0:
                    logger.info(
                        "Sleeping %ds (includes %ds backoff after %d consecutive errors)",
                        sleep_total, self._backoff_seconds, self._consecutive_exceptions,
                    )
                else:
                    logger.debug("Sleeping %ds...", sleep_total)
                # Sleep in small increments so we can respond to signals
                while sleep_total > 0 and self._running:
                    time.sleep(min(1, sleep_total))
                    sleep_total -= 1

            logger.info("Orchestrator stopped")
        finally:
            self.notifier.shutdown()
            self.safety.release_lock()
