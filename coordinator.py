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
from safety import SafetyError, SafetyGuard, GracefulDegradation
from state import CycleRecord
from state_lock import LockedStateManager
from task_discovery import Task, TaskDiscovery
from validator import Validator
from worker import Worker, WorkerResult
from shared import gather_tasks as _shared_gather_tasks
from task_queue import TaskApprovalQueue

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
        self._degradation = GracefulDegradation(config)
        self.max_workers = config.parallel.max_workers
        self._running = True
        self._workers: List[Worker] = []
        self._consecutive_merge_failures: int = 0
        self._task_queue = TaskApprovalQueue(str(Path(self.config.paths.state_dir)))

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

            # Log strategy performance from recent history
            perf_report = self.state.get_strategy_performance_report()
            logger.info(perf_report)

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
                # Apply graceful degradation sleep multiplier
                if self._degradation.is_degraded:
                    deg = self._degradation.check_and_adjust(
                        self.state.get_cycle_count_last_hour(),
                        self.state.get_total_cost(lookback_seconds=3600),
                    )
                    sleep_time = int(sleep_time * deg["sleep_multiplier"])
                while sleep_time > 0 and self._running:
                    time.sleep(min(1, sleep_time))
                    sleep_time -= 1

            logger.info("ParallelCoordinator stopped")
        finally:
            self._cleanup_all_worktrees()
            self.safety.release_lock()

    def _run_cycle(self) -> None:
        """Run a single parallel cycle."""
        cycle_start = time.time()
        self.safety.pre_flight_checks()
        self._check_worktree_disk_space()

        # Check for orphaned worktrees from a previous crash
        if self.config.orchestrator.session_recovery:
            from session_manager import SessionManager
            session_mgr = SessionManager(str(Path(self.config.paths.state_dir)))
            orphaned = session_mgr.recover_orphaned_worktrees(self.config.target_dir)
            if orphaned:
                logger.warning(
                    "Found %d orphaned worktrees from a previous session: %s",
                    len(orphaned),
                    [o.get("branch", "?") for o in orphaned],
                )
                for wt in orphaned:
                    wt_path = wt.get("path", "")
                    branch = wt.get("branch", "")
                    if wt_path:
                        try:
                            self.git.remove_worktree(wt_path, force=True)
                        except Exception:
                            import shutil
                            shutil.rmtree(wt_path, ignore_errors=True)
                    if branch:
                        self.git.delete_branch(branch, force=True)
                self.git.prune_worktrees()

        # Check for graceful degradation
        cycles_per_hour = self.state.get_cycle_count_last_hour()
        cost_per_hour = self.state.get_total_cost(lookback_seconds=3600)
        degradation = self._degradation.check_and_adjust(cycles_per_hour, cost_per_hour)
        if degradation["degraded"]:
            logger.warning(
                "Graceful degradation active (level %d): %s",
                degradation["level"], degradation["reason"],
            )

        tasks = self._gather_tasks()
        if not tasks:
            logger.info("No actionable tasks found")
            return

        # Circuit breaker: if merges have failed too many times consecutively,
        # skip dispatching workers to avoid wasting Claude invocations
        merge_threshold = self.config.parallel.max_merge_retries * 2
        if self._consecutive_merge_failures >= merge_threshold:
            logger.error(
                "Merge circuit breaker tripped: %d consecutive merge failures "
                "(threshold %d). Skipping worker dispatch — manual intervention "
                "or conflict resolution may be needed.",
                self._consecutive_merge_failures, merge_threshold,
            )
            return

        groups = self._partition_tasks(tasks)
        if not groups:
            return

        # Capture baseline test failures once before dispatching workers
        validator = Validator(self.config)
        baseline_failures = validator.capture_baseline(self.config.target_dir)
        if baseline_failures:
            logger.warning(
                "Baseline: %d pre-existing test failure(s)", len(baseline_failures),
            )
        else:
            logger.info("Baseline: all tests pass")

        logger.info(
            "Dispatching %d task group(s) to parallel workers",
            len(groups),
        )

        # Claim feedback files before dispatching
        for group in groups:
            to_remove = []
            for task in group:
                if task.source == "feedback" and task.source_file:
                    if not self.feedback.claim_feedback(task.source_file):
                        logger.warning(
                            "Could not claim feedback file %s, skipping",
                            task.source_file,
                        )
                        to_remove.append(task)
            for task in to_remove:
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
                    baseline_failures=baseline_failures,
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

        self._log_cycle_summary(results, cycle_start)
        self._workers.clear()
        self.git.prune_worktrees()

    def _process_result(self, result: WorkerResult, worker: Worker) -> None:
        """Process a single worker result: merge if successful, record cycle."""
        if result.success:
            merged = self._merge_worker_branch(worker, result)
            if merged:
                self._consecutive_merge_failures = 0
                # Mark feedback as done
                for task in result.tasks:
                    if task.source == "feedback" and task.source_file:
                        self.feedback.mark_done_claimed(task.source_file)
            else:
                # Merge failed — record as failure
                self._consecutive_merge_failures += 1
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

    def _log_cycle_summary(
        self, results: List[tuple], cycle_start: float,
    ) -> None:
        """Log a summary of the parallel cycle's outcomes."""
        if not results:
            return

        total_duration = time.time() - cycle_start
        total_tasks = 0
        succeeded = 0
        failed = 0
        total_cost = 0.0
        succeeded_types: List[str] = []
        failed_types: List[str] = []

        for result, _worker in results:
            n_tasks = len(result.tasks) if result.tasks else 1
            total_tasks += n_tasks
            total_cost += result.cost_usd
            task_types = [t.source for t in result.tasks] if result.tasks else ["unknown"]
            if result.success:
                succeeded += n_tasks
                succeeded_types.extend(task_types)
            else:
                failed += n_tasks
                failed_types.extend(task_types)

        logger.info(
            "Cycle summary: %d task(s) dispatched, %d succeeded, %d failed | "
            "cost=$%.4f | duration=%.1fs | "
            "succeeded_types=%s | failed_types=%s",
            total_tasks, succeeded, failed,
            total_cost, total_duration,
            list(set(succeeded_types)) if succeeded_types else [],
            list(set(failed_types)) if failed_types else [],
        )

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
        pre_merge_snapshot = self.git.create_snapshot()

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
                            self.git.rollback(pre_merge_snapshot)
                            return False
                    else:
                        logger.warning(
                            "Worker %d: fast-forward failed after rebase",
                            worker.worker_id,
                        )
                else:
                    logger.warning(
                        "Worker %d: rebase failed on attempt %d/%d, trying merge fallback",
                        worker.worker_id, attempt + 1, max_retries + 1,
                    )
                    # Fallback: try a regular merge when rebase fails
                    try:
                        self.git.checkout(original_branch)
                    except Exception as e:
                        logger.error("Failed to checkout %s for merge fallback: %s", original_branch, e)
                        continue
                    if self.git.merge_branch(worker.branch_name):
                        logger.info(
                            "Worker %d: merge fallback succeeded for branch %s into %s",
                            worker.worker_id, worker.branch_name, original_branch,
                        )
                        return True
                    # Merge fallback also had conflicts
                    self.git.abort_merge()
                    logger.warning(
                        "Worker %d: merge fallback also failed on attempt %d/%d",
                        worker.worker_id, attempt + 1, max_retries + 1,
                    )

        # --- AI Conflict Resolution ---
        if self.config.parallel.ai_conflict_resolution:
            logger.info(
                "Worker %d: attempting AI conflict resolution for branch %s",
                worker.worker_id, worker.branch_name,
            )
            ai_snapshot = self.git.create_snapshot()

            # Attempt merge leaving conflicts in working tree
            self.git.merge_no_commit(worker.branch_name)

            conflicted = self.git.get_conflicted_files()
            if conflicted:
                from conflict_resolver import ConflictResolver
                resolver = ConflictResolver(self.config)
                success, cost = resolver.resolve_conflicts(
                    self.config.target_dir, conflicted, worker.branch_name, original_branch,
                )

                if success:
                    commit_msg = f"Merge branch '{worker.branch_name}' (AI-resolved conflicts)"
                    commit_hash = self.git.mark_resolved_and_commit(conflicted, commit_msg)
                    if commit_hash:
                        validator = Validator(self.config)
                        validation = validator.validate(self.config.target_dir)
                        if validation.passed:
                            logger.info("Worker %d: AI conflict resolution succeeded", worker.worker_id)
                            return True
                        else:
                            logger.warning("Worker %d: validation failed after AI resolution: %s",
                                           worker.worker_id, validation.summary)

                # AI resolution failed — rollback
                self.git.rollback(ai_snapshot)
            else:
                self.git.abort_merge()

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

    def _check_worktree_disk_space(self) -> None:
        """Check disk space and clean up stale worktree directories if low.

        Prevents parallel workers from exhausting disk when multiple worktrees
        accumulate. Uses shutil.disk_usage() for an efficient O(1) disk space
        check instead of recursively walking worktree directories.
        """
        worktree_base = Path(self.config.target_dir) / self.config.parallel.worktree_base_dir
        if not worktree_base.exists():
            return

        # Collect worktree directories (cheap iterdir, no recursive walk)
        worktree_dirs = []
        try:
            for child in worktree_base.iterdir():
                if child.is_dir() and child.name.startswith("worker-"):
                    worktree_dirs.append(child)
        except OSError:
            return

        if worktree_dirs:
            logger.debug(
                "Found %d worktree directories", len(worktree_dirs),
            )

        # Check overall disk space (O(1) syscall, no recursive walk)
        min_disk_mb = self.config.safety.min_disk_space_mb
        try:
            usage = shutil.disk_usage(self.config.target_dir)
            free_mb = usage.free / (1024 * 1024)
        except OSError:
            return

        warning_threshold = min_disk_mb * 1.5
        if free_mb < warning_threshold:
            logger.warning(
                "Disk space low (%.0f MB free) with %d worktree directories. "
                "Cleaning up stale worktree directories.",
                free_mb, len(worktree_dirs),
            )
            # Remove worktree dirs that don't correspond to active workers
            active_ids = {w.worker_id for w in self._workers}
            for wt_dir in worktree_dirs:
                # Parse worker id from directory name (e.g., "worker-0")
                try:
                    wt_id = int(wt_dir.name.split("-", 1)[1])
                except (ValueError, IndexError):
                    wt_id = -1
                if wt_id not in active_ids:
                    logger.info("Removing stale worktree: %s", wt_dir)
                    try:
                        self.git.remove_worktree(str(wt_dir), force=True)
                    except Exception:
                        shutil.rmtree(str(wt_dir), ignore_errors=True)

    def _gather_tasks(self) -> List[Task]:
        """Gather all eligible tasks (delegates to shared implementation)."""
        dashboard_active = self._task_queue.is_dashboard_active()
        return _shared_gather_tasks(
            self.config, self.feedback, self.state, self.discovery,
            dashboard_active=dashboard_active,
            task_approval_queue=self._task_queue,
        )

    def _partition_tasks(self, tasks: List[Task]) -> List[List[Task]]:
        """Assign one task per worker, up to max_workers.

        Feedback tasks get priority ordering (appear first), then
        auto-discovered tasks fill remaining worker slots.
        Tasks with unmet dependencies are filtered out.
        """
        # Filter out tasks with unmet dependencies
        completed_keys = set()
        try:
            history = self.state.load_history()
            for record in history:
                if record.get("success"):
                    for key in record.get("task_keys", []):
                        completed_keys.add(key)
        except Exception:
            pass

        eligible = []
        for t in tasks:
            if t.depends_on:
                unmet = [dep for dep in t.depends_on if dep not in completed_keys]
                if unmet:
                    logger.info(
                        "Skipping task %s: unmet dependencies %s",
                        t.task_id, unmet,
                    )
                    continue
            eligible.append(t)

        # Sort: feedback first (priority-ordered), then auto-discovered
        feedback_tasks = sorted(
            [t for t in eligible if t.source == "feedback"],
            key=lambda t: t.priority,
        )
        auto_tasks = sorted(
            [t for t in eligible if t.source != "feedback"],
            key=lambda t: t.priority,
        )
        ordered = feedback_tasks + auto_tasks

        # One task per worker, capped at max_workers (reduced by degradation)
        effective_workers = self.max_workers
        if self._degradation.is_degraded:
            deg = self._degradation.check_and_adjust(
                self.state.get_cycle_count_last_hour(),
                self.state.get_total_cost(lookback_seconds=3600),
            )
            effective_workers = max(1, int(self.max_workers * deg["batch_size_factor"]))

        # Group tasks that reference the same source_file to avoid merge
        # conflicts when different workers modify the same file.
        groups: List[List[Task]] = []
        file_to_group: dict = {}  # source_file -> group index

        for t in ordered:
            if len(groups) >= effective_workers and (
                t.source_file is None or t.source_file not in file_to_group
            ):
                break  # No room for a new group

            if t.source_file and t.source_file in file_to_group:
                # Append to existing group that already handles this file
                groups[file_to_group[t.source_file]].append(t)
            else:
                if len(groups) >= effective_workers:
                    break
                if t.source_file:
                    file_to_group[t.source_file] = len(groups)
                groups.append([t])

        return groups

    def _setup_signals(self) -> None:
        """Register signal handlers for graceful shutdown."""
        self._signal_received = False

        def handler(signum, frame):
            if self._signal_received:
                return  # Avoid redundant handling on repeated delivery
            self._signal_received = True
            logger.info("Received signal %d, shutting down workers...", signum)
            self._running = False
            # Terminate any running Claude subprocesses in workers
            # Snapshot the list to avoid iterating while main thread mutates it
            for worker in list(self._workers):
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
