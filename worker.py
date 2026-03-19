"""Worker: runs a task group in an isolated git worktree."""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

from claude_runner import ClaudeRunner, ClaudeResult
from config_schema import Config
from cycle_state import CycleState, CycleStateWriter
from git_manager import GitManager
from safety import SafetyError, SafetyGuard
from state_lock import LockedStateManager
from task_discovery import Task
from validator import Validator
from shared import (
    format_task_list as _shared_format_task_list,
    format_validation_errors as _shared_format_validation_errors,
    syntax_check_files as _shared_syntax_check_files,
    build_commit_message as _shared_build_commit_message,
    build_batch_commit_message as _shared_build_batch_commit_message,
    TASK_TYPE_INSTRUCTIONS,
    build_task_prompt as _shared_build_task_prompt,
    build_plan_prompt as _shared_build_plan_prompt,
    build_execute_prompt as _shared_build_execute_prompt,
    build_retry_prompt as _shared_build_retry_prompt,
)

logger = logging.getLogger(__name__)


@dataclass
class WorkerResult:
    success: bool
    branch_name: str = ""
    commit_hash: str = ""
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    error: str = ""
    tasks: List[Task] = field(default_factory=list)


class Worker:
    """Runs a single task group in an isolated git worktree.

    Each worker creates its own worktree on a dedicated branch, runs the
    standard cycle logic (build prompt -> invoke Claude -> validate -> commit),
    and returns a WorkerResult.  The coordinator is responsible for merging
    the branch back into main.
    """

    def __init__(
        self,
        config: Config,
        tasks: List[Task],
        state: LockedStateManager,
        worker_id: int,
        main_repo_dir: str,
        baseline_failures: Optional[Set[str]] = None,
    ):
        self.config = config
        self.tasks = tasks
        self.worker_id = worker_id
        self.state = state
        self.main_repo_dir = main_repo_dir
        self.baseline_failures = baseline_failures or set()
        self.branch_name = f"auto-claude/{int(time.time_ns())}-{worker_id}"

        worktree_base = config.parallel.worktree_base_dir
        # Resolve relative to the main repo
        base = Path(main_repo_dir) / worktree_base
        self.worktree_dir = str(base / f"worker-{worker_id}")

        # Will be set up during execute()
        self._git: Optional[GitManager] = None
        self._claude: Optional[ClaudeRunner] = None

    def execute(self) -> WorkerResult:
        """Full worker lifecycle: create worktree -> plan -> execute -> validate -> commit.

        Supports plan-then-execute mode (when config.orchestrator.plan_changes
        is True) and validation retries (re-invokes Claude with failure output).
        """
        start_time = time.time()
        total_cost = 0.0

        try:
            self._setup_worktree()
        except Exception as e:
            logger.error("Worker %d: failed to create worktree: %s", self.worker_id, e)
            return WorkerResult(
                success=False,
                branch_name=self.branch_name,
                error=f"Worktree setup failed: {e}",
                tasks=self.tasks,
            )

        try:
            # Create worker-local components pointing at the worktree
            self._git = GitManager(self.worktree_dir)
            from provider_runner import create_runner
            self._claude = create_runner(self.config)

            # Create worker-specific cycle state writer
            state_dir = str(Path(self.config.paths.state_dir))
            cycle_state = CycleStateWriter(state_dir, worker_id=self.worker_id)

            is_batch = len(self.tasks) > 1

            # Write cycle state
            cycle_state.write(CycleState(
                phase="planning" if self.config.orchestrator.plan_changes else "executing",
                task_description=self.tasks[0].description,
                task_type=self.tasks[0].source,
                task_descriptions=[t.description for t in self.tasks],
                started_at=start_time,
                batch_size=len(self.tasks),
            ))

            # Snapshot main repo state before invoking Claude
            main_repo_git = GitManager(self.main_repo_dir)
            main_repo_pre_state = set(main_repo_git.get_changed_files())

            # --- Plan-then-execute or direct execution ---
            exec_prompt = self._build_prompt(self.tasks, is_batch)

            if self.config.orchestrator.plan_changes:
                # Cost guard before planning phase
                if self._cost_limit_exceeded(total_cost):
                    return WorkerResult(
                        success=False,
                        branch_name=self.branch_name,
                        cost_usd=total_cost,
                        duration_seconds=time.time() - start_time,
                        error="Cost limit approaching, aborting before planning",
                        tasks=self.tasks,
                    )

                # Planning phase — use a local variable for effective turns
                # instead of mutating self.config.claude.max_turns (which is
                # shared across workers and would cause a race condition).
                plan_prompt = self._build_plan_prompt(self.tasks, is_batch)
                original_max_turns = self.config.claude.max_turns
                base_turns = self.config.orchestrator.planning_max_turns
                effective_turns = base_turns + max(0, len(self.tasks) - 1) * 2
                effective_turns = min(effective_turns, original_max_turns)

                logger.info(
                    "Worker %d: planning with max_turns=%d",
                    self.worker_id, effective_turns,
                )
                # Create a worker-local ClaudeRunner with overridden max_turns
                import copy
                plan_config = copy.deepcopy(self.config)
                plan_config.claude = copy.deepcopy(self.config.claude)
                plan_config.claude.max_turns = effective_turns
                from provider_runner import create_runner
                plan_runner = create_runner(plan_config)
                plan_result = plan_runner.run(
                    plan_prompt,
                    add_dirs=[str(Path(self.worktree_dir).resolve())],
                )
                total_cost += plan_result.cost_usd

                if not plan_result.success:
                    logger.warning(
                        "Worker %d: planning failed: %s",
                        self.worker_id, plan_result.error,
                    )
                    return WorkerResult(
                        success=False,
                        branch_name=self.branch_name,
                        cost_usd=total_cost,
                        duration_seconds=time.time() - start_time,
                        error=f"Planning failed: {plan_result.error}",
                        tasks=self.tasks,
                    )

                # Revert any accidental changes from planning
                self._git.rollback()

                if plan_result.result_text.strip():
                    logger.info("Worker %d: plan created, executing...", self.worker_id)
                    exec_prompt = self._build_execute_prompt(
                        self.tasks, is_batch, plan_result.result_text,
                    )
                else:
                    if plan_result.error and "max_turns" in plan_result.error:
                        logger.warning(
                            "Worker %d: planning exhausted max_turns with no output, "
                            "skipping planning",
                            self.worker_id,
                        )
                    else:
                        logger.warning(
                            "Worker %d: planning returned empty result, "
                            "falling back to direct execution",
                            self.worker_id,
                        )
                    # exec_prompt already set to direct prompt above

            # --- Execution phase ---
            cycle_state.update(phase="executing")
            logger.info(
                "Worker %d: invoking Claude for %d task(s) in %s",
                self.worker_id, len(self.tasks), self.worktree_dir,
            )
            claude_result = self._claude.run(
                exec_prompt,
                add_dirs=[str(Path(self.worktree_dir).resolve())],
            )
            total_cost += claude_result.cost_usd

            if not claude_result.success:
                logger.warning("Worker %d: Claude failed: %s", self.worker_id, claude_result.error)
                return WorkerResult(
                    success=False,
                    branch_name=self.branch_name,
                    cost_usd=total_cost,
                    duration_seconds=time.time() - start_time,
                    error=claude_result.error,
                    tasks=self.tasks,
                )

            # Check if Claude accidentally modified the main repo
            main_repo_post_state = set(main_repo_git.get_changed_files())
            new_main_dirty = sorted(main_repo_post_state - main_repo_pre_state)
            if new_main_dirty:
                worktree_abs = str(Path(self.worktree_dir).resolve())
                files_list = "\n  ".join(new_main_dirty[:10])
                extra = (
                    f" (and {len(new_main_dirty) - 10} more)"
                    if len(new_main_dirty) > 10 else ""
                )
                logger.error(
                    "Worker %d: Claude modified %d file(s) in the main repo "
                    "instead of the worktree.\n"
                    "  Modified files:\n  %s%s\n"
                    "  Likely cause: Claude used relative paths or the main "
                    "repo path (%s) instead of absolute paths to the "
                    "worktree (%s).\n"
                    "  Prompt working_dir was set to: %s",
                    self.worker_id, len(new_main_dirty),
                    files_list, extra,
                    self.main_repo_dir, worktree_abs, worktree_abs,
                )
                error_msg = (
                    f"Claude modified {len(new_main_dirty)} main repo "
                    f"file(s) instead of worktree: {new_main_dirty[:5]}. "
                    f"Likely cause: used relative paths instead of absolute "
                    f"paths to the worktree ({worktree_abs})"
                )
                return WorkerResult(
                    success=False,
                    branch_name=self.branch_name,
                    cost_usd=total_cost,
                    duration_seconds=time.time() - start_time,
                    error=error_msg,
                    tasks=self.tasks,
                )

            # Check changed files
            changed_files = self._git.get_changed_files()
            if not changed_files:
                logger.info("Worker %d: no files changed", self.worker_id)
                return WorkerResult(
                    success=False,
                    branch_name=self.branch_name,
                    cost_usd=total_cost,
                    duration_seconds=time.time() - start_time,
                    error="No files changed",
                    tasks=self.tasks,
                )

            # Safety checks on changed files
            try:
                safety = SafetyGuard(self.config, self.state)
                safety.check_protected_files(changed_files)
                safety.check_file_count(changed_files)
            except SafetyError as e:
                logger.warning("Worker %d: safety check failed: %s", self.worker_id, e)
                return WorkerResult(
                    success=False,
                    branch_name=self.branch_name,
                    cost_usd=total_cost,
                    duration_seconds=time.time() - start_time,
                    error=str(e),
                    tasks=self.tasks,
                )

            # Syntax check if self_improve is on
            if self.config.orchestrator.self_improve:
                syntax_err = self._syntax_check_files(changed_files)
                if syntax_err:
                    logger.warning("Worker %d: syntax check failed: %s", self.worker_id, syntax_err)
                    return WorkerResult(
                        success=False,
                        branch_name=self.branch_name,
                        cost_usd=total_cost,
                        duration_seconds=time.time() - start_time,
                        error=syntax_err,
                        tasks=self.tasks,
                    )

            # --- Validate with retries ---
            max_retries = self.config.orchestrator.max_validation_retries
            cycle_state.update(phase="validating")
            validator = Validator(self.config)
            validation = validator.validate_with_baseline(
                self.worktree_dir, self.baseline_failures,
            )

            retry = 0
            while not validation.passed and retry < max_retries:
                retry += 1

                # Cost guard before retry
                if self._cost_limit_exceeded(total_cost):
                    logger.warning(
                        "Worker %d: cost limit approaching, aborting retries",
                        self.worker_id,
                    )
                    break

                logger.info(
                    "Worker %d: validation failed (attempt %d/%d), retrying...",
                    self.worker_id, retry, max_retries + 1,
                )
                cycle_state.update(phase="retrying", retry_count=retry)

                # Build retry prompt with full failure output
                failure_output = self._format_validation_errors(validation)
                retry_prompt = self._build_retry_prompt(
                    self.tasks, is_batch, failure_output,
                )
                retry_result = self._claude.run(
                    retry_prompt,
                    add_dirs=[str(Path(self.worktree_dir).resolve())],
                )
                total_cost += retry_result.cost_usd

                if not retry_result.success:
                    logger.warning(
                        "Worker %d: retry Claude call failed: %s",
                        self.worker_id, retry_result.error,
                    )
                    break

                # Re-validate
                cycle_state.update(phase="validating")
                validation = validator.validate_with_baseline(
                    self.worktree_dir, self.baseline_failures,
                )

            if not validation.passed:
                logger.warning(
                    "Worker %d: validation failed after %d retries: %s",
                    self.worker_id, retry, validation.summary,
                )
                return WorkerResult(
                    success=False,
                    branch_name=self.branch_name,
                    cost_usd=total_cost,
                    duration_seconds=time.time() - start_time,
                    error=f"Validation failed: {validation.summary}",
                    tasks=self.tasks,
                )

            # LLM Judge evaluation (before commit)
            if self.config.judges.enabled:
                from llm_judges import JudgePanel
                panel = JudgePanel(self.config)
                diff_text = self._git.get_diff()
                panel_result = panel.evaluate(
                    changed_files, diff_text,
                    self.tasks[0].description,
                )
                total_cost += panel_result.total_cost_usd
                if not panel_result.passed:
                    if self.config.judges.fail_action == "rollback":
                        logger.warning(
                            "Worker %d: LLM judges rejected changes",
                            self.worker_id,
                        )
                        return WorkerResult(
                            success=False,
                            branch_name=self.branch_name,
                            cost_usd=total_cost,
                            duration_seconds=time.time() - start_time,
                            error="LLM judges rejected changes",
                            tasks=self.tasks,
                        )
                    elif self.config.judges.fail_action != "warn":
                        logger.warning(
                            "Worker %d: LLM judges failed (action=%s)",
                            self.worker_id, self.config.judges.fail_action,
                        )

            # Commit locally on the branch
            changed_files = self._git.get_changed_files()
            commit_msg = self._build_commit_message(self.tasks, is_batch)
            commit_hash = self._git.commit(commit_msg, files=changed_files)

            if commit_hash is None:
                return WorkerResult(
                    success=False,
                    branch_name=self.branch_name,
                    cost_usd=total_cost,
                    duration_seconds=time.time() - start_time,
                    error="Commit failed (git error)",
                    tasks=self.tasks,
                )

            if not commit_hash:
                return WorkerResult(
                    success=False,
                    branch_name=self.branch_name,
                    cost_usd=total_cost,
                    duration_seconds=time.time() - start_time,
                    error="Commit failed (no staged changes)",
                    tasks=self.tasks,
                )

            logger.info(
                "Worker %d: committed %s on branch %s",
                self.worker_id, commit_hash[:8], self.branch_name,
            )

            return WorkerResult(
                success=True,
                branch_name=self.branch_name,
                commit_hash=commit_hash,
                cost_usd=total_cost,
                duration_seconds=time.time() - start_time,
                tasks=self.tasks,
            )

        except Exception as e:
            logger.exception("Worker %d: unexpected error", self.worker_id)
            return WorkerResult(
                success=False,
                branch_name=self.branch_name,
                cost_usd=total_cost,
                duration_seconds=time.time() - start_time,
                error=f"Unexpected error: {e}",
                tasks=self.tasks,
            )
        finally:
            cycle_state.clear()

    def _setup_worktree(self) -> None:
        """Create a git worktree with a new branch."""
        Path(self.worktree_dir).parent.mkdir(parents=True, exist_ok=True)
        main_git = GitManager(self.main_repo_dir)
        main_git.create_worktree(self.worktree_dir, self.branch_name)
        logger.info(
            "Worker %d: created worktree at %s (branch %s)",
            self.worker_id, self.worktree_dir, self.branch_name,
        )

    def cleanup(self) -> None:
        """Remove the worktree and delete the branch."""
        main_git = GitManager(self.main_repo_dir)
        main_git.remove_worktree(self.worktree_dir, force=True)
        main_git.delete_branch(self.branch_name, force=True)
        # Clean up the directory if it still exists
        wt_path = Path(self.worktree_dir)
        if wt_path.exists():
            shutil.rmtree(str(wt_path), ignore_errors=True)
        main_git.prune_worktrees()
        logger.info("Worker %d: cleaned up worktree and branch", self.worker_id)

    def _build_prompt(self, tasks: List[Task], is_batch: bool) -> str:
        """Build the Claude prompt for the task(s)."""
        return _shared_build_task_prompt(
            tasks,
            self.config.safety.protected_files,
            working_dir=str(Path(self.worktree_dir).resolve()),
        )

    def _format_task_list(self, tasks: List[Task]) -> str:
        """Format tasks as a numbered list."""
        return _shared_format_task_list(tasks)

    def _build_plan_prompt(self, tasks: List[Task], is_batch: bool) -> str:
        """Build a planning-only prompt (no file changes)."""
        return _shared_build_plan_prompt(
            tasks,
            self.config.safety.protected_files,
            working_dir=str(Path(self.worktree_dir).resolve()),
        )

    def _build_execute_prompt(
        self, tasks: List[Task], is_batch: bool, plan_text: str,
    ) -> str:
        """Build an execution prompt that includes a pre-made plan."""
        return _shared_build_execute_prompt(
            tasks,
            plan_text,
            self.config.safety.protected_files,
            working_dir=str(Path(self.worktree_dir).resolve()),
        )

    def _format_validation_errors(self, validation) -> str:
        """Extract failure details from ValidationResult for the retry prompt."""
        return _shared_format_validation_errors(validation, include_full=True)

    def _build_retry_prompt(
        self, tasks: List[Task], is_batch: bool, failure_output: str,
    ) -> str:
        """Build a retry prompt with validation failure output."""
        task_history = self.state.get_task_success_history(
            tasks[0].description,
            task_key=tasks[0].task_key,
        )
        return _shared_build_retry_prompt(
            tasks,
            failure_output,
            self.config.safety.protected_files,
            working_dir=str(Path(self.worktree_dir).resolve()),
            task_history=task_history,
        )

    def _build_commit_message(self, tasks: List[Task], is_batch: bool) -> str:
        """Build a commit message for the worker's changes."""
        if is_batch:
            return _shared_build_batch_commit_message(tasks)
        return _shared_build_commit_message(tasks[0])

    def _syntax_check_files(self, changed_files: List[str]) -> Optional[str]:
        """Syntax-check modified .py files."""
        return _shared_syntax_check_files(changed_files, self.worktree_dir)

    def _cost_limit_exceeded(self, worker_cost: float) -> bool:
        """Check if the hourly cost budget is nearly exhausted.

        Returns True if accumulated cost (hourly + this worker) exceeds 90%
        of the configured limit, signaling that the worker should stop.
        """
        try:
            hourly_cost = self.state.get_total_cost(lookback_seconds=3600)
            cost_limit = self.config.safety.max_cost_usd_per_hour
            if hourly_cost + worker_cost >= cost_limit * 0.9:
                logger.warning(
                    "Worker %d: cost guard triggered — $%.2f accumulated "
                    "(limit $%.2f)",
                    self.worker_id, hourly_cost + worker_cost, cost_limit,
                )
                return True
        except Exception as e:
            logger.debug("Worker %d: cost check failed: %s", self.worker_id, e)
        return False
