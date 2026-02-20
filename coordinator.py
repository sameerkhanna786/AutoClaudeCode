"""Parallel coordinator: distributes tasks to workers, merges results."""

from __future__ import annotations

import logging
import shutil
import signal
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from config_schema import Config
from feedback import FeedbackManager
from git_manager import GitManager
from model_resolver import resolve_model_id
from safety import SafetyError, SafetyGuard
from state import CycleRecord
from state_lock import LockedStateManager
from task_discovery import Task, TaskDiscovery
from validator import Validator
from worker import Worker, WorkerResult

logger = logging.getLogger(__name__)


class ParallelCoordinator:
    """Manages parallel Claude Code workers using git worktrees.

    Discovers tasks, distributes them to workers running in separate
    worktrees, and merges validated results back to main.
    """

    def __init__(self, config: Config):
        self.config = config
        self.git = GitManager(config.target_dir)
        self.state = LockedStateManager(config)
        self.safety = SafetyGuard(config, self.state)
        self.discovery = TaskDiscovery(config, state_manager=self.state)
        self.feedback = FeedbackManager(config)
        self.max_workers = config.parallel.max_workers
        self._running = True
        self._workers: List[Worker] = []

    def run(self, once: bool = False) -> None:
        """Main loop: discover tasks, dispatch to workers, merge results."""
        self._setup_signals()

        try:
            self.safety.acquire_lock()
        except SafetyError as e:
            logger.error("Cannot start: %s", e)
            return

        try:
            logger.info(
                "ParallelCoordinator started (max_workers=%d, once=%s)",
                self.max_workers, once,
            )

            while self._running:
                try:
                    self._run_cycle()
                except SafetyError as e:
                    logger.warning("Pre-flight check failed: %s", e)
                except Exception:
                    logger.exception("Unexpected error in parallel cycle")

                if once:
                    break

                # Sleep in small increments for signal responsiveness
                sleep_time = self.config.orchestrator.loop_interval_seconds
                while sleep_time > 0 and self._running:
                    time.sleep(min(1, sleep_time))
                    sleep_time -= 1

            logger.info("ParallelCoordinator stopped")
        finally:
            self._cleanup_all_worktrees()
            self.safety.release_lock()

    def _run_cycle(self) -> None:
        """Run a single parallel cycle."""
        self.safety.pre_flight_checks()

        tasks = self._gather_tasks()
        if not tasks:
            logger.info("No actionable tasks found")
            return

        groups = self._partition_tasks(tasks)
        if not groups:
            return

        logger.info(
            "Dispatching %d task group(s) to parallel workers",
            len(groups),
        )

        # Claim feedback files before dispatching
        for group in groups:
            for task in group:
                if task.source == "feedback" and task.source_file:
                    if not self.feedback.claim_feedback(task.source_file):
                        logger.warning(
                            "Could not claim feedback file %s, skipping",
                            task.source_file,
                        )
                        group.remove(task)

        # Remove empty groups
        groups = [g for g in groups if g]
        if not groups:
            return

        # Dispatch workers
        results: List[tuple] = []  # (WorkerResult, Worker)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {}
            for i, task_group in enumerate(groups):
                worker = Worker(
                    config=self.config,
                    tasks=task_group,
                    state=self.state,
                    worker_id=i,
                    main_repo_dir=self.config.target_dir,
                )
                self._workers.append(worker)
                futures[pool.submit(worker.execute)] = worker

            for future in as_completed(futures):
                worker = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    logger.error(
                        "Worker %d raised exception: %s",
                        worker.worker_id, e,
                    )
                    result = WorkerResult(
                        success=False,
                        branch_name=worker.branch_name,
                        error=str(e),
                        tasks=worker.tasks,
                    )
                results.append((result, worker))

        # Merge successful branches and record cycles
        for result, worker in results:
            try:
                self._process_result(result, worker)
            except Exception:
                logger.exception(
                    "Error processing result for worker %d",
                    worker.worker_id,
                )
            finally:
                self._cleanup_worker_with_timeout(worker)

        self._workers.clear()
        self.git.prune_worktrees()

    def _process_result(self, result: WorkerResult, worker: Worker) -> None:
        """Process a single worker result: merge if successful, record cycle."""
        if result.success:
            merged = self._merge_worker_branch(worker, result)
            if merged:
                # Mark feedback as done
                for task in result.tasks:
                    if task.source == "feedback" and task.source_file:
                        self.feedback.mark_done_claimed(task.source_file)
            else:
                # Merge failed — record as failure
                for task in result.tasks:
                    if task.source == "feedback" and task.source_file:
                        self.feedback.unclaim_feedback(task.source_file)
                result = WorkerResult(
                    success=False,
                    branch_name=result.branch_name,
                    cost_usd=result.cost_usd,
                    duration_seconds=result.duration_seconds,
                    error="Merge to main failed",
                    tasks=result.tasks,
                )
        else:
            # Worker failed — unclaim feedback files
            for task in result.tasks:
                if task.source == "feedback" and task.source_file:
                    self.feedback.unclaim_feedback(task.source_file)

        # Record cycle
        self.state.record_cycle(CycleRecord(
            timestamp=time.time(),
            task_description=result.tasks[0].description if result.tasks else "unknown",
            task_type=result.tasks[0].source if result.tasks else "unknown",
            success=result.success,
            commit_hash=result.commit_hash,
            cost_usd=result.cost_usd,
            duration_seconds=result.duration_seconds,
            error=result.error,
            task_descriptions=[t.description for t in result.tasks],
            task_types=[t.source for t in result.tasks],
            batch_size=len(result.tasks),
            task_keys=[t.task_key for t in result.tasks],
            task_source_files=[t.source_file or "" for t in result.tasks],
            task_line_numbers=[t.line_number for t in result.tasks],
        ))

    def _merge_worker_branch(self, worker: Worker, result: WorkerResult) -> bool:
        """Merge a worker's branch back into main.

        Strategy:
        1. Try fast-forward merge
        2. Try auto-merge
        3. Try rebase + fast-forward
        4. Re-validate after rebase if needed
        5. Give up and leave branch for manual review
        """
        strategy = self.config.parallel.merge_strategy
        max_retries = self.config.parallel.max_merge_retries

        # Remember current branch (should be main)
        original_branch = self.git.get_current_branch()

        for attempt in range(max_retries + 1):
            # Ensure we're on the main branch
            try:
                self.git.checkout(original_branch)
            except Exception as e:
                logger.error("Failed to checkout %s: %s", original_branch, e)
                return False

            # 1. Try fast-forward merge
            if self.git.merge_ff_only(worker.branch_name):
                logger.info(
                    "Worker %d: fast-forward merged branch %s into %s",
                    worker.worker_id, worker.branch_name, original_branch,
                )
                return True

            if strategy == "merge":
                # 2. Try auto-merge
                if self.git.merge_branch(worker.branch_name):
                    logger.info(
                        "Worker %d: auto-merged branch %s into %s",
                        worker.worker_id, worker.branch_name, original_branch,
                    )
                    return True
                # Merge had conflicts
                self.git.abort_merge()
                logger.warning(
                    "Worker %d: merge conflicts on attempt %d/%d",
                    worker.worker_id, attempt + 1, max_retries + 1,
                )

            elif strategy == "rebase":
                # 3. Rebase the worker branch onto main
                if self.git.rebase_onto(original_branch, worker.branch_name):
                    # Now try fast-forward merge
                    self.git.checkout(original_branch)
                    if self.git.merge_ff_only(worker.branch_name):
                        # Re-validate after rebase
                        validator = Validator(self.config)
                        validation = validator.validate(self.config.target_dir)
                        if validation.passed:
                            logger.info(
                                "Worker %d: rebased and merged branch %s into %s",
                                worker.worker_id, worker.branch_name, original_branch,
                            )
                            return True
                        else:
                            # Validation failed after rebase — undo the merge
                            logger.warning(
                                "Worker %d: validation failed after rebase: %s",
                                worker.worker_id, validation.summary,
                            )
                            # Reset main back to before the merge
                            self.git.rollback()
                            return False
                    else:
                        logger.warning(
                            "Worker %d: fast-forward failed after rebase",
                            worker.worker_id,
                        )
                else:
                    logger.warning(
                        "Worker %d: rebase failed on attempt %d/%d",
                        worker.worker_id, attempt + 1, max_retries + 1,
                    )

        # All attempts exhausted
        logger.error(
            "Worker %d: all merge strategies failed for branch %s. "
            "Leaving branch for manual review.",
            worker.worker_id, worker.branch_name,
        )
        # Ensure we're back on original branch
        try:
            self.git.checkout(original_branch)
        except Exception:
            pass
        return False

    def _gather_tasks(self) -> List[Task]:
        """Gather all eligible tasks (same logic as Orchestrator._gather_tasks)."""
        tasks: List[Task] = []

        # Priority 1: developer feedback
        max_retries = self.config.orchestrator.max_feedback_retries
        for task in self.feedback.get_pending_feedback():
            failure_count = self.state.get_task_failure_count(
                task.description, "feedback", task_key=task.task_key,
            )
            if failure_count >= max_retries:
                logger.warning(
                    "Feedback task failed %d times, moving to failed/",
                    failure_count,
                )
                if task.source_file:
                    self.feedback.mark_failed(task.source_file)
                continue
            if not self.state.was_recently_attempted(
                task.description, task_key=task.task_key,
            ):
                tasks.append(task)

        # Auto-discovered tasks
        discovered = self.discovery.discover_all()
        for task in discovered:
            if not self.state.was_recently_attempted(
                task.description, task_key=task.task_key,
            ):
                tasks.append(task)

        return tasks

    def _partition_tasks(self, tasks: List[Task]) -> List[List[Task]]:
        """Split tasks into groups for parallel workers.

        Each feedback task gets its own worker (high priority, human-written).
        Remaining slots get auto-discovered tasks grouped by source type.
        """
        groups: List[List[Task]] = []
        feedback_tasks = [t for t in tasks if t.source == "feedback"]
        auto_tasks = [t for t in tasks if t.source != "feedback"]

        # Each feedback task gets its own worker
        for t in feedback_tasks:
            if len(groups) < self.max_workers:
                groups.append([t])

        # Remaining slots get auto-discovered tasks, split into chunks
        # and round-robin'd across available worker slots.
        remaining_slots = self.max_workers - len(groups)
        if remaining_slots > 0 and auto_tasks:
            max_batch = self.config.orchestrator.max_batch_size
            by_source: dict = defaultdict(list)
            for t in auto_tasks:
                by_source[t.source].append(t)

            # Split each source's tasks into chunks of max_batch_size
            source_chunks: list = []
            for source_tasks in by_source.values():
                for i in range(0, len(source_tasks), max_batch):
                    source_chunks.append(source_tasks[i:i + max_batch])

            # Round-robin chunks across remaining worker slots
            for chunk in source_chunks:
                if len(groups) >= self.max_workers:
                    break
                groups.append(chunk)

        return groups

    def _setup_signals(self) -> None:
        """Register signal handlers for graceful shutdown."""
        def handler(signum, frame):
            logger.info("Received signal %d, shutting down workers...", signum)
            self._running = False
            # Terminate any running Claude subprocesses in workers
            for worker in self._workers:
                if worker._claude is not None:
                    try:
                        worker._claude.terminate()
                    except Exception:
                        pass

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _cleanup_worker_with_timeout(self, worker: Worker, timeout: float = 30) -> None:
        """Clean up a single worker's worktree and branch with a timeout.

        Isolates errors between remove_worktree, delete_branch, and
        prune_worktrees so that a failure in one step doesn't prevent
        the others from running. Falls back to shutil.rmtree if
        git worktree remove fails or times out.
        """
        def _do_cleanup():
            main_git = GitManager(self.config.target_dir)
            # Step 1: remove worktree via git
            try:
                main_git.remove_worktree(worker.worktree_dir, force=True)
            except Exception:
                logger.warning(
                    "Worker %d: git worktree remove failed, falling back to rmtree",
                    worker.worker_id,
                )
                wt_path = Path(worker.worktree_dir)
                if wt_path.exists():
                    shutil.rmtree(str(wt_path), ignore_errors=True)

            # Step 2: force-remove the directory if it still exists
            wt_path = Path(worker.worktree_dir)
            if wt_path.exists():
                shutil.rmtree(str(wt_path), ignore_errors=True)

            # Step 3: delete the branch
            try:
                main_git.delete_branch(worker.branch_name, force=True)
            except Exception:
                logger.warning(
                    "Worker %d: branch deletion failed for %s",
                    worker.worker_id, worker.branch_name,
                )

        thread = threading.Thread(target=_do_cleanup, daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        if thread.is_alive():
            logger.warning(
                "Worker %d: cleanup timed out after %.0fs, abandoning",
                worker.worker_id, timeout,
            )

    def _cleanup_all_worktrees(self) -> None:
        """Remove all worktree directories on shutdown.

        Runs the cleanup in a daemon thread with a 60s overall timeout
        to prevent indefinite hangs when worktrees are locked or
        corrupted. Each per-worktree operation is error-isolated with
        a shutil.rmtree fallback.
        """
        if not self.config.parallel.cleanup_on_exit:
            return

        def _do_cleanup():
            worktree_base = Path(self.config.target_dir) / self.config.parallel.worktree_base_dir
            if worktree_base.exists():
                # Clean up each worker directory with error isolation
                for child in worktree_base.iterdir():
                    if child.is_dir() and child.name.startswith("worker-"):
                        try:
                            self.git.remove_worktree(str(child), force=True)
                        except Exception:
                            logger.warning(
                                "Failed to git-remove worktree %s, falling back to rmtree",
                                child,
                            )
                            shutil.rmtree(str(child), ignore_errors=True)

                # Remove the base directory if empty
                try:
                    if worktree_base.exists() and not any(worktree_base.iterdir()):
                        worktree_base.rmdir()
                except OSError:
                    pass

            try:
                self.git.prune_worktrees()
            except Exception:
                logger.warning("Failed to prune worktrees during cleanup")

        thread = threading.Thread(target=_do_cleanup, daemon=True)
        thread.start()
        thread.join(timeout=self.config.parallel.cleanup_timeout)
        if thread.is_alive():
            logger.warning(
                "Worktree cleanup timed out after %ds, abandoning remaining cleanup",
                self.config.parallel.cleanup_timeout,
            )
        else:
            logger.info("Cleaned up all worktrees")
