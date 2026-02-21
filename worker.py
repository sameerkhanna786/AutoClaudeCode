"""Worker: runs a task group in an isolated git worktree."""

from __future__ import annotations

import ast
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from claude_runner import ClaudeRunner, ClaudeResult
from config_schema import Config
from cycle_state import CycleState, CycleStateWriter
from git_manager import GitManager
from safety import SafetyError, SafetyGuard
from state import CycleRecord
from state_lock import LockedStateManager
from task_discovery import Task
from validator import Validator

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
    ):
        self.config = config
        self.tasks = tasks
        self.worker_id = worker_id
        self.state = state
        self.main_repo_dir = main_repo_dir
        self.branch_name = f"auto-claude/{int(time.time())}-{worker_id}"

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
            self._claude = ClaudeRunner(self.config)

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
                # Planning phase
                plan_prompt = self._build_plan_prompt(self.tasks, is_batch)
                original_max_turns = self.config.claude.max_turns
                base_turns = self.config.orchestrator.planning_max_turns
                effective_turns = base_turns + max(0, len(self.tasks) - 1) * 2
                effective_turns = min(effective_turns, original_max_turns)
                self.config.claude.max_turns = effective_turns

                logger.info(
                    "Worker %d: planning with max_turns=%d",
                    self.worker_id, effective_turns,
                )
                plan_result = self._claude.run(
                    plan_prompt,
                    add_dirs=[str(Path(self.worktree_dir).resolve())],
                )
                self.config.claude.max_turns = original_max_turns
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
                logger.error(
                    "Worker %d: Claude modified files in the main repo "
                    "instead of the worktree: %s",
                    self.worker_id, new_main_dirty[:5],
                )
                return WorkerResult(
                    success=False,
                    branch_name=self.branch_name,
                    cost_usd=total_cost,
                    duration_seconds=time.time() - start_time,
                    error=f"Claude modified main repo files: {new_main_dirty[:5]}",
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
            validation = validator.validate(self.worktree_dir)

            retry = 0
            while not validation.passed and retry < max_retries:
                retry += 1
                logger.info(
                    "Worker %d: validation failed (attempt %d/%d), retrying...",
                    self.worker_id, retry, max_retries + 1,
                )
                cycle_state.update(phase="retrying", retry_count=retry)

                # Build retry prompt with failure output
                retry_prompt = self._build_retry_prompt(
                    self.tasks, is_batch, validation.summary,
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
                validation = validator.validate(self.worktree_dir)

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

            # Commit locally on the branch
            changed_files = self._git.get_changed_files()
            commit_msg = self._build_commit_message(self.tasks, is_batch)
            commit_hash = self._git.commit(commit_msg, files=changed_files)

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
        protected = ", ".join(self.config.safety.protected_files)

        if is_batch:
            task_list = self._format_task_list(tasks)
            return (
                f"You are working on the project at {Path(self.worktree_dir).resolve()}.\n"
                "All file reads, writes, and edits MUST use absolute paths within that directory.\n"
                "WARNING: Do NOT modify any files outside that directory. Do NOT use relative paths.\n\n"
                "You have been given a batch of tasks to address in a single comprehensive change.\n\n"
                f"TASKS:\n{task_list}\n\n"
                "INSTRUCTIONS:\n"
                "- Make the minimal changes needed to complete ALL tasks above.\n"
                "- Do NOT run git commands (add, commit, push). The orchestrator handles git.\n"
                f"- Do NOT modify these protected files: {protected}\n"
                "- Focus on correctness. Do NOT run tests — the orchestrator handles testing.\n"
                "- If a task is unclear or impossible, make your best effort and explain what you did.\n"
            )

        task = tasks[0]
        context_section = ""
        if task.context:
            context_section = f"\nCONTEXT:\n{task.context}\n"
        return (
            f"You are working on the project at {Path(self.worktree_dir).resolve()}.\n"
            "All file reads, writes, and edits MUST use absolute paths within that directory.\n"
            "WARNING: Do NOT modify any files outside that directory. Do NOT use relative paths.\n\n"
            f"TASK: {task.description}\n"
            f"{context_section}\n"
            "INSTRUCTIONS:\n"
            "- Make the minimal changes needed to complete this task.\n"
            "- Do NOT run git commands (add, commit, push). The orchestrator handles git.\n"
            f"- Do NOT modify these protected files: {protected}\n"
            "- Focus on correctness. Do NOT run tests — the orchestrator handles testing.\n"
            "- If the task is unclear or impossible, make your best effort and explain what you did.\n"
        )

    def _format_task_list(self, tasks: List[Task]) -> str:
        """Format tasks as a numbered list."""
        lines = []
        for i, task in enumerate(tasks, 1):
            lines.append(f"{i}. {task.description} [{task.source}]")
            if task.context:
                lines.append(f"   CONTEXT:")
                for ctx_line in task.context.split("\n"):
                    lines.append(f"   {ctx_line}")
        return "\n".join(lines)

    def _build_plan_prompt(self, tasks: List[Task], is_batch: bool) -> str:
        """Build a planning-only prompt (no file changes)."""
        protected = ", ".join(self.config.safety.protected_files)
        wt = Path(self.worktree_dir).resolve()

        if is_batch:
            task_list = self._format_task_list(tasks)
            task_section = f"TASKS:\n{task_list}"
        else:
            task = tasks[0]
            ctx = f"\nCONTEXT:\n{task.context}\n" if task.context else ""
            task_section = f"TASK: {task.description}\n{ctx}"

        return (
            f"You are working on the project at {wt}.\n"
            "All file reads MUST use absolute paths within that directory.\n\n"
            f"{task_section}\n\n"
            "INSTRUCTIONS:\n"
            "- Analyze the codebase and create a detailed plan.\n"
            "- Do NOT make any changes yet. Only output a plan.\n"
            "- List the files you would modify and what changes you would make.\n"
            f"- Do NOT modify these protected files: {protected}\n"
            "- Be specific about the changes (function names, line numbers, etc.).\n"
        )

    def _build_execute_prompt(
        self, tasks: List[Task], is_batch: bool, plan_text: str,
    ) -> str:
        """Build an execution prompt that includes a pre-made plan."""
        protected = ", ".join(self.config.safety.protected_files)
        wt = Path(self.worktree_dir).resolve()

        if is_batch:
            task_list = self._format_task_list(tasks)
            task_section = f"TASKS:\n{task_list}"
        else:
            task = tasks[0]
            ctx = f"\nCONTEXT:\n{task.context}\n" if task.context else ""
            task_section = f"TASK: {task.description}\n{ctx}"

        return (
            f"You are working on the project at {wt}.\n"
            "All file reads, writes, and edits MUST use absolute paths within that directory.\n"
            "WARNING: Do NOT modify any files outside that directory.\n\n"
            f"{task_section}\n\n"
            f"PLAN TO EXECUTE:\n{plan_text}\n\n"
            "INSTRUCTIONS:\n"
            "- Execute the plan above. Make the exact changes described.\n"
            "- Do NOT run git commands (add, commit, push). The orchestrator handles git.\n"
            f"- Do NOT modify these protected files: {protected}\n"
            "- Focus on correctness. Do NOT run tests — the orchestrator handles testing.\n"
        )

    def _build_retry_prompt(
        self, tasks: List[Task], is_batch: bool, failure_output: str,
    ) -> str:
        """Build a retry prompt with validation failure output."""
        protected = ", ".join(self.config.safety.protected_files)
        wt = Path(self.worktree_dir).resolve()

        if is_batch:
            task_list = self._format_task_list(tasks)
            task_section = f"TASKS:\n{task_list}"
        else:
            task = tasks[0]
            task_section = f"TASK: {task.description}"

        # Truncate failure output to avoid exceeding prompt limits
        max_output = 8000
        if len(failure_output) > max_output:
            failure_output = failure_output[:max_output] + "\n... (truncated)"

        return (
            f"You are working on the project at {wt}.\n"
            "All file reads, writes, and edits MUST use absolute paths within that directory.\n"
            "WARNING: Do NOT modify any files outside that directory.\n\n"
            f"{task_section}\n\n"
            "The previous attempt FAILED validation. Here is the failure output:\n\n"
            f"```\n{failure_output}\n```\n\n"
            "INSTRUCTIONS:\n"
            "- Fix the issues shown in the failure output above.\n"
            "- Do NOT run git commands. The orchestrator handles git.\n"
            f"- Do NOT modify these protected files: {protected}\n"
            "- Focus on fixing the test/lint/build failures.\n"
        )

    def _build_commit_message(self, tasks: List[Task], is_batch: bool) -> str:
        """Build a commit message for the worker's changes."""
        if is_batch:
            subject = f"Auto-fix {len(tasks)} tasks"
            body_lines = [f"- {t.description[:100]}" for t in tasks]
            return subject + "\n\n" + "\n".join(body_lines)
        task = tasks[0]
        desc = task.description[:72]
        if len(task.description) > 72:
            desc = task.description[:69] + "..."
        return desc

    def _syntax_check_files(self, changed_files: List[str]) -> Optional[str]:
        """Syntax-check modified .py files."""
        for f in changed_files:
            if f.endswith(".py"):
                full_path = Path(self.worktree_dir) / f
                if full_path.exists():
                    try:
                        source = full_path.read_text()
                        ast.parse(source, filename=f)
                    except SyntaxError as e:
                        return f"Syntax error in {f} at line {e.lineno}: {e.msg}"
        return None
