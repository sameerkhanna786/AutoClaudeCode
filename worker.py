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
        """Full worker lifecycle: create worktree -> run claude -> validate -> commit."""
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

            # Build prompt
            is_batch = len(self.tasks) > 1
            prompt = self._build_prompt(self.tasks, is_batch)

            # Write cycle state
            cycle_state.write(CycleState(
                phase="executing",
                task_description=self.tasks[0].description,
                task_type=self.tasks[0].source,
                task_descriptions=[t.description for t in self.tasks],
                started_at=start_time,
                batch_size=len(self.tasks),
            ))

            # Invoke Claude in the worktree directory
            logger.info(
                "Worker %d: invoking Claude for %d task(s) in %s",
                self.worker_id, len(self.tasks), self.worktree_dir,
            )
            claude_result = self._claude.run(prompt, working_dir=self.worktree_dir)
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

            # Validate (run tests, lint, build)
            cycle_state.update(phase="validating")
            validator = Validator(self.config)
            validation = validator.validate(self.worktree_dir)

            if not validation.passed:
                logger.warning(
                    "Worker %d: validation failed: %s",
                    self.worker_id, validation.summary,
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
                "You are working on the project in the current directory.\n\n"
                "You have been given a batch of tasks to address in a single comprehensive change.\n\n"
                f"TASKS:\n{task_list}\n\n"
                "INSTRUCTIONS:\n"
                "- Make the minimal changes needed to complete ALL tasks above.\n"
                "- Do NOT run git commands (add, commit, push). The orchestrator handles git.\n"
                f"- Do NOT modify these protected files: {protected}\n"
                "- Focus on correctness. Run tests if available.\n"
                "- If a task is unclear or impossible, make your best effort and explain what you did.\n"
            )

        task = tasks[0]
        context_section = ""
        if task.context:
            context_section = f"\nCONTEXT:\n{task.context}\n"
        return (
            "You are working on the project in the current directory.\n\n"
            f"TASK: {task.description}\n"
            f"{context_section}\n"
            "INSTRUCTIONS:\n"
            "- Make the minimal changes needed to complete this task.\n"
            "- Do NOT run git commands (add, commit, push). The orchestrator handles git.\n"
            f"- Do NOT modify these protected files: {protected}\n"
            "- Focus on correctness. Run tests if available.\n"
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
