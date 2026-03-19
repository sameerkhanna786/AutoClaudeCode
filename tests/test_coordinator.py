"""Tests for coordinator module."""

import logging
import signal
import threading
import time
from collections import defaultdict
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from config_schema import Config, ParallelConfig
from coordinator import ParallelCoordinator
from task_discovery import Task
from worker import WorkerResult


@pytest.fixture
def parallel_config(tmp_git_repo):
    """Config with parallel enabled, targeting a temp git repo."""
    config = Config()
    config.target_dir = tmp_git_repo
    config.parallel = ParallelConfig(
        enabled=True,
        max_workers=3,
        worktree_base_dir=".worktrees",
        merge_strategy="rebase",
        max_merge_retries=2,
        cleanup_on_exit=True,
    )
    config.paths.state_dir = str(Path(tmp_git_repo) / "state")
    config.paths.history_file = str(Path(tmp_git_repo) / "state" / "history.json")
    config.paths.lock_file = str(Path(tmp_git_repo) / "state" / "lock.pid")
    config.paths.feedback_dir = str(Path(tmp_git_repo) / "feedback")
    config.paths.feedback_done_dir = str(Path(tmp_git_repo) / "feedback" / "done")
    config.paths.feedback_failed_dir = str(Path(tmp_git_repo) / "feedback" / "failed")
    return config


class TestPartitionTasks:
    def test_feedback_tasks_get_own_workers(self, parallel_config):
        coord = ParallelCoordinator(parallel_config)
        tasks = [
            Task(description="Fix bug A", priority=1, source="feedback", source_file="f1.md"),
            Task(description="Fix bug B", priority=1, source="feedback", source_file="f2.md"),
            Task(description="Fix lint", priority=3, source="lint"),
        ]
        groups = coord._partition_tasks(tasks)
        # Each feedback task gets its own group
        assert len(groups) >= 2
        feedback_groups = [g for g in groups if g[0].source == "feedback"]
        assert len(feedback_groups) == 2
        for fg in feedback_groups:
            assert len(fg) == 1

    def test_auto_tasks_one_per_worker(self, parallel_config):
        coord = ParallelCoordinator(parallel_config)
        tasks = [
            Task(description="Lint 1", priority=3, source="lint"),
            Task(description="Lint 2", priority=3, source="lint"),
            Task(description="Todo 1", priority=4, source="todo"),
        ]
        groups = coord._partition_tasks(tasks)
        # Each task gets its own worker
        assert len(groups) == 3
        for g in groups:
            assert len(g) == 1

    def test_respects_max_workers(self, parallel_config):
        parallel_config.parallel.max_workers = 2
        coord = ParallelCoordinator(parallel_config)
        tasks = [
            Task(description="Feedback 1", priority=1, source="feedback", source_file="f1.md"),
            Task(description="Feedback 2", priority=1, source="feedback", source_file="f2.md"),
            Task(description="Feedback 3", priority=1, source="feedback", source_file="f3.md"),
        ]
        groups = coord._partition_tasks(tasks)
        assert len(groups) <= 2

    def test_empty_tasks_returns_empty(self, parallel_config):
        coord = ParallelCoordinator(parallel_config)
        groups = coord._partition_tasks([])
        assert groups == []

    def test_mixed_sources_partitioned(self, parallel_config):
        coord = ParallelCoordinator(parallel_config)
        tasks = [
            Task(description="FB task", priority=1, source="feedback", source_file="fb.md"),
            Task(description="Lint 1", priority=3, source="lint"),
            Task(description="Todo 1", priority=4, source="todo"),
            Task(description="Test fail", priority=2, source="test_failure"),
        ]
        groups = coord._partition_tasks(tasks)
        # Should have feedback in its own group + auto tasks by source
        assert len(groups) >= 2
        assert len(groups) <= parallel_config.parallel.max_workers

    def test_single_source_one_per_worker(self, parallel_config):
        """8 tasks of same source, 4 workers → 4 groups of 1 each."""
        parallel_config.parallel.max_workers = 4
        coord = ParallelCoordinator(parallel_config)
        tasks = [
            Task(description=f"Idea {i}", priority=5, source="claude_idea")
            for i in range(8)
        ]
        groups = coord._partition_tasks(tasks)
        assert len(groups) == 4
        for g in groups:
            assert len(g) == 1

    def test_single_source_capped_by_max_workers(self, parallel_config):
        """9 tasks of same source, 2 workers → 2 groups of 1 each."""
        parallel_config.parallel.max_workers = 2
        coord = ParallelCoordinator(parallel_config)
        tasks = [
            Task(description=f"Idea {i}", priority=5, source="claude_idea")
            for i in range(9)
        ]
        groups = coord._partition_tasks(tasks)
        assert len(groups) == 2
        for g in groups:
            assert len(g) == 1

    def test_round_robin_across_sources(self, parallel_config):
        """Lint + todo tasks each get their own worker."""
        parallel_config.parallel.max_workers = 6
        coord = ParallelCoordinator(parallel_config)
        tasks = [
            Task(description="Lint 1", priority=3, source="lint"),
            Task(description="Lint 2", priority=3, source="lint"),
            Task(description="Lint 3", priority=3, source="lint"),
            Task(description="Lint 4", priority=3, source="lint"),
            Task(description="Todo 1", priority=4, source="todo"),
            Task(description="Todo 2", priority=4, source="todo"),
        ]
        groups = coord._partition_tasks(tasks)
        # Each task gets its own worker, all 6 fit within max_workers=6
        assert len(groups) == 6
        for g in groups:
            assert len(g) == 1


    def test_same_source_file_grouped_together(self, parallel_config):
        """Tasks referencing the same source_file go to the same worker."""
        parallel_config.parallel.max_workers = 4
        coord = ParallelCoordinator(parallel_config)
        tasks = [
            Task(description="Fix func A in foo.py", priority=3, source="lint", source_file="foo.py"),
            Task(description="Fix func B in foo.py", priority=3, source="lint", source_file="foo.py"),
            Task(description="Fix bar.py", priority=3, source="lint", source_file="bar.py"),
        ]
        groups = coord._partition_tasks(tasks)
        # foo.py tasks grouped together, bar.py separate
        assert len(groups) == 2
        foo_group = [g for g in groups if any(t.source_file == "foo.py" for t in g)]
        assert len(foo_group) == 1
        assert len(foo_group[0]) == 2
        bar_group = [g for g in groups if any(t.source_file == "bar.py" for t in g)]
        assert len(bar_group) == 1
        assert len(bar_group[0]) == 1

    def test_same_source_file_respects_max_workers(self, parallel_config):
        """Same-file grouping still respects max_workers for new groups."""
        parallel_config.parallel.max_workers = 2
        coord = ParallelCoordinator(parallel_config)
        tasks = [
            Task(description="Fix A", priority=3, source="lint", source_file="a.py"),
            Task(description="Fix B", priority=3, source="lint", source_file="b.py"),
            Task(description="Fix A2", priority=4, source="lint", source_file="a.py"),
            Task(description="Fix C", priority=4, source="lint", source_file="c.py"),
        ]
        groups = coord._partition_tasks(tasks)
        # Only 2 groups max, but a.py tasks should be grouped
        assert len(groups) == 2
        a_group = [g for g in groups if any(t.source_file == "a.py" for t in g)]
        assert len(a_group) == 1
        assert len(a_group[0]) == 2  # both a.py tasks grouped


    def test_tasks_after_ungroupable_still_join_existing_groups(self, parallel_config):
        """Tasks appearing after an ungroupable task should still be added
        to existing groups if their source_file matches."""
        parallel_config.parallel.max_workers = 2
        coord = ParallelCoordinator(parallel_config)
        tasks = [
            Task(description="Fix A", priority=1, source="lint", source_file="a.py"),
            Task(description="Fix B", priority=2, source="lint", source_file="b.py"),
            # This task has no source_file and groups are full — should be skipped
            Task(description="Fix C", priority=3, source="lint", source_file=None),
            # This task maps to existing group a.py — should still be included
            Task(description="Fix A2", priority=4, source="lint", source_file="a.py"),
        ]
        groups = coord._partition_tasks(tasks)
        assert len(groups) == 2
        # a.py group should contain both "Fix A" and "Fix A2"
        a_group = [g for g in groups if any(t.source_file == "a.py" for t in g)]
        assert len(a_group) == 1
        a_descriptions = [t.description for t in a_group[0]]
        assert "Fix A" in a_descriptions
        assert "Fix A2" in a_descriptions

    def test_degradation_param_reduces_workers(self, parallel_config):
        """Pre-computed degradation result reduces effective workers."""
        parallel_config.parallel.max_workers = 4
        coord = ParallelCoordinator(parallel_config)
        tasks = [
            Task(description=f"Task {i}", priority=3, source="lint")
            for i in range(6)
        ]
        degradation = {
            "degraded": True,
            "batch_size_factor": 0.5,
            "level": 1,
            "reason": "test",
        }
        groups = coord._partition_tasks(tasks, degradation=degradation)
        # 4 workers * 0.5 factor = 2 effective workers
        assert len(groups) <= 2

    def test_no_degradation_param_uses_max_workers(self, parallel_config):
        """Without degradation param, all max_workers are available."""
        parallel_config.parallel.max_workers = 3
        coord = ParallelCoordinator(parallel_config)
        tasks = [
            Task(description=f"Task {i}", priority=3, source="lint")
            for i in range(5)
        ]
        groups = coord._partition_tasks(tasks, degradation=None)
        assert len(groups) == 3

    def test_non_degraded_param_uses_max_workers(self, parallel_config):
        """A non-degraded result doesn't reduce workers."""
        parallel_config.parallel.max_workers = 4
        coord = ParallelCoordinator(parallel_config)
        tasks = [
            Task(description=f"Task {i}", priority=3, source="lint")
            for i in range(5)
        ]
        degradation = {"degraded": False, "batch_size_factor": 1.0}
        groups = coord._partition_tasks(tasks, degradation=degradation)
        assert len(groups) == 4


class TestMergeWorkerBranch:
    @pytest.mark.requires_subprocess
    def test_fast_forward_merge(self, tmp_git_repo, parallel_config):
        """Test that fast-forward merge works when main hasn't moved."""
        from git_manager import GitManager

        coord = ParallelCoordinator(parallel_config)
        main_git = GitManager(tmp_git_repo)

        # Create a branch with a commit
        branch_name = "auto-claude/test-ff"
        worktree_dir = str(Path(tmp_git_repo) / ".worktrees" / "test-ff")
        Path(worktree_dir).parent.mkdir(parents=True, exist_ok=True)
        main_git.create_worktree(worktree_dir, branch_name)

        # Make a commit in the worktree
        wt_git = GitManager(worktree_dir)
        Path(worktree_dir, "new_file.txt").write_text("hello")
        wt_git.commit("Add new file", files=["new_file.txt"])

        # Remove worktree but keep branch
        main_git.remove_worktree(worktree_dir, force=True)

        # Create a mock worker
        worker = MagicMock()
        worker.branch_name = branch_name
        worker.worker_id = 0

        result = coord._merge_worker_branch(
            worker,
            WorkerResult(success=True, branch_name=branch_name, tasks=[]),
        )
        assert result is True

        # Verify the file exists on main
        assert Path(tmp_git_repo, "new_file.txt").exists()

        # Cleanup
        main_git.delete_branch(branch_name, force=True)


class TestGatherTasks:
    def test_gather_tasks_deduplicates(self, parallel_config):
        """Tasks recently attempted are excluded."""
        coord = ParallelCoordinator(parallel_config)

        with patch.object(coord.feedback, "get_pending_feedback", return_value=[]):
            with patch.object(coord.discovery, "discover_all", return_value=[
                Task(description="Fix X", priority=3, source="lint"),
            ]):
                with patch.object(coord.state, "was_recently_attempted", return_value=True):
                    tasks = coord._gather_tasks()
        assert tasks == []

    def test_gather_tasks_returns_new(self, parallel_config):
        """New tasks are returned."""
        coord = ParallelCoordinator(parallel_config)

        with patch.object(coord.feedback, "get_pending_feedback", return_value=[]):
            with patch.object(coord.discovery, "discover_all", return_value=[
                Task(description="Fix Y", priority=3, source="lint"),
            ]):
                with patch.object(coord.state, "was_recently_attempted", return_value=False):
                    tasks = coord._gather_tasks()
        assert len(tasks) == 1
        assert tasks[0].description == "Fix Y"


class TestSignalHandlerIdempotent:
    def test_repeated_signals_handled_once(self, parallel_config, caplog):
        """Signal handler only logs once even if called multiple times."""
        import logging
        import signal

        coord = ParallelCoordinator(parallel_config)
        coord._setup_signals()

        # Get the registered handler
        handler = signal.getsignal(signal.SIGTERM)

        with caplog.at_level(logging.INFO):
            handler(signal.SIGTERM, None)
            handler(signal.SIGTERM, None)
            handler(signal.SIGTERM, None)

        signal_msgs = [r for r in caplog.records if "Received signal" in r.message]
        assert len(signal_msgs) == 1
        assert not coord._running


class TestCleanupAllWorktreesTimeout:
    def test_cleanup_completes_normally(self, parallel_config):
        """Normal cleanup finishes within the timeout."""
        coord = ParallelCoordinator(parallel_config)
        # Create a worktree directory structure
        worktree_base = Path(parallel_config.target_dir) / ".worktrees"
        worker_dir = worktree_base / "worker-0"
        worker_dir.mkdir(parents=True, exist_ok=True)

        with patch.object(coord.git, "remove_worktree"):
            with patch.object(coord.git, "prune_worktrees"):
                coord._cleanup_all_worktrees()
        # Should complete without hanging

    def test_cleanup_times_out_on_hang(self, parallel_config, caplog):
        """Cleanup that hangs is abandoned after the timeout."""
        import logging

        parallel_config.parallel.cleanup_timeout = 1  # Short timeout for test
        coord = ParallelCoordinator(parallel_config)
        worktree_base = Path(parallel_config.target_dir) / ".worktrees"
        worker_dir = worktree_base / "worker-0"
        worker_dir.mkdir(parents=True, exist_ok=True)

        def hang_forever(*args, **kwargs):
            time.sleep(120)

        with patch.object(coord.git, "remove_worktree", side_effect=hang_forever):
            with patch.object(coord.git, "prune_worktrees"):
                with caplog.at_level(logging.WARNING):
                    coord._cleanup_all_worktrees()

        assert any("timed out" in r.message for r in caplog.records)

    def test_cleanup_error_isolation_per_worktree(self, parallel_config):
        """Error in one worktree cleanup doesn't prevent others from being attempted."""
        import shutil as _shutil

        coord = ParallelCoordinator(parallel_config)
        worktree_base = Path(parallel_config.target_dir) / ".worktrees"
        (worktree_base / "worker-0").mkdir(parents=True, exist_ok=True)
        (worktree_base / "worker-1").mkdir(parents=True, exist_ok=True)

        call_log = []

        def tracking_remove(path, force=False):
            call_log.append(path)
            if "worker-0" in path:
                raise RuntimeError("worktree locked")

        with patch.object(coord.git, "remove_worktree", side_effect=tracking_remove):
            with patch.object(coord.git, "prune_worktrees"):
                coord._cleanup_all_worktrees()

        # Both worktrees should have been attempted
        assert len(call_log) == 2
        # worker-0 directory should have been cleaned up via fallback rmtree
        # (since error isolation means we continue)


class TestCleanupWorkerWithTimeout:
    def test_worker_cleanup_completes(self, parallel_config):
        """Worker cleanup that finishes in time completes normally."""
        coord = ParallelCoordinator(parallel_config)
        worker = MagicMock()
        worker.worker_id = 0
        worker.worktree_dir = str(Path(parallel_config.target_dir) / ".worktrees" / "worker-0")
        worker.branch_name = "auto-claude/test-0"

        coord._cleanup_worker_with_timeout(worker, timeout=5)
        # Should complete without error

    def test_worker_cleanup_timeout_logged(self, parallel_config, caplog):
        """Worker cleanup that hangs is abandoned after the timeout."""
        import logging

        coord = ParallelCoordinator(parallel_config)
        worker = MagicMock()
        worker.worker_id = 42
        worker.worktree_dir = str(Path(parallel_config.target_dir) / ".worktrees" / "worker-42")
        worker.branch_name = "auto-claude/test-42"

        # Override to hang
        original_init = None

        with patch("coordinator.GitManager") as MockGit:
            mock_git_instance = MockGit.return_value
            mock_git_instance.remove_worktree.side_effect = lambda *a, **kw: time.sleep(60)

            with caplog.at_level(logging.WARNING):
                coord._cleanup_worker_with_timeout(worker, timeout=1)

        assert any("cleanup timed out" in r.message and "42" in r.message for r in caplog.records)


class TestMergeValidation:
    """Verify that non-fast-forward merges are validated before returning True."""

    def test_merge_strategy_validates_after_merge(self, parallel_config):
        """Merge strategy must validate after a non-ff merge."""
        parallel_config.parallel.merge_strategy = "merge"
        coord = ParallelCoordinator(parallel_config)

        worker = MagicMock()
        worker.branch_name = "auto-claude/test-merge"
        worker.worker_id = 0

        worker_result = WorkerResult(success=True, branch_name="auto-claude/test-merge", tasks=[])

        # Simulate: ff fails, auto-merge succeeds, but validation fails
        coord.git.merge_ff_only = MagicMock(return_value=False)
        coord.git.merge_branch = MagicMock(return_value=True)
        coord.git.rollback = MagicMock()
        coord.git.take_snapshot = MagicMock(return_value="snap")

        mock_validation = MagicMock()
        mock_validation.passed = False
        mock_validation.summary = "tests failed"

        with patch("coordinator.Validator") as MockValidator:
            MockValidator.return_value.validate.return_value = mock_validation
            result = coord._merge_worker_branch(worker, worker_result)

        assert result is False
        coord.git.rollback.assert_called()

    def test_rebase_fallback_merge_validates(self, parallel_config):
        """Rebase fallback to merge must also validate."""
        parallel_config.parallel.merge_strategy = "rebase"
        coord = ParallelCoordinator(parallel_config)

        worker = MagicMock()
        worker.branch_name = "auto-claude/test-rebase"
        worker.worker_id = 0

        worker_result = WorkerResult(success=True, branch_name="auto-claude/test-rebase", tasks=[])

        # Simulate: ff fails, rebase fails, merge fallback succeeds, validation fails
        coord.git.merge_ff_only = MagicMock(return_value=False)
        coord.git.rebase_onto = MagicMock(return_value=False)
        coord.git.checkout = MagicMock()
        coord.git.merge_branch = MagicMock(return_value=True)
        coord.git.abort_merge = MagicMock()
        coord.git.rollback = MagicMock()
        coord.git.take_snapshot = MagicMock(return_value="snap")

        mock_validation = MagicMock()
        mock_validation.passed = False
        mock_validation.summary = "lint failed"

        with patch("coordinator.Validator") as MockValidator:
            MockValidator.return_value.validate.return_value = mock_validation
            result = coord._merge_worker_branch(worker, worker_result)

        assert result is False
        coord.git.rollback.assert_called()


class TestMergeCheckoutFailureAfterRebase:
    """Tests that checkout failures during merge are handled, not silently ignored."""

    def test_checkout_failure_after_rebase_returns_false(self, parallel_config):
        """If checkout fails after rebase, merge should return False, not crash."""
        parallel_config.parallel.merge_strategy = "rebase"
        coord = ParallelCoordinator(parallel_config)

        worker = MagicMock()
        worker.branch_name = "auto-claude/test-checkout-fail"
        worker.worker_id = 0

        worker_result = WorkerResult(success=True, branch_name=worker.branch_name, tasks=[])

        # ff fails, rebase succeeds, but checkout back to main fails
        coord.git.merge_ff_only = MagicMock(return_value=False)
        coord.git.rebase_onto = MagicMock(return_value=True)
        coord.git.checkout = MagicMock(side_effect=Exception("checkout failed"))

        result = coord._merge_worker_branch(worker, worker_result)
        assert result is False

    def test_final_checkout_failure_logged(self, parallel_config, caplog):
        """The final 'ensure we're on original branch' logs on checkout failure."""
        import logging

        parallel_config.parallel.merge_strategy = "merge"
        parallel_config.parallel.max_merge_retries = 0
        coord = ParallelCoordinator(parallel_config)

        worker = MagicMock()
        worker.branch_name = "auto-claude/test-final-checkout"
        worker.worker_id = 0

        worker_result = WorkerResult(success=True, branch_name=worker.branch_name, tasks=[])

        # ff fails, merge has conflicts, abort succeeds
        coord.git.merge_ff_only = MagicMock(return_value=False)
        coord.git.merge_branch = MagicMock(return_value=False)
        coord.git.abort_merge = MagicMock()
        # Final checkout fails
        call_count = [0]
        def checkout_side_effect(branch):
            call_count[0] += 1
            if call_count[0] == 1:
                return  # First checkout succeeds (start of loop)
            raise Exception("checkout stuck")
        coord.git.checkout = MagicMock(side_effect=checkout_side_effect)
        coord.config.parallel.ai_conflict_resolution = False

        with caplog.at_level(logging.WARNING):
            result = coord._merge_worker_branch(worker, worker_result)

        assert result is False
        assert any("checkout" in r.message.lower() for r in caplog.records if r.levelno >= logging.WARNING)


class TestPartitionTasksNewFileAfterGroupsFull:
    """Test that tasks with unknown source_files don't abort the entire loop."""

    def test_new_source_file_skipped_but_later_known_file_still_grouped(self, parallel_config):
        """When groups are full and a task has a new source_file, it should be
        skipped (continue), not abort the loop (break). Later tasks matching
        existing groups must still be added."""
        parallel_config.parallel.max_workers = 2
        coord = ParallelCoordinator(parallel_config)
        tasks = [
            Task(description="Fix A", priority=1, source="lint", source_file="a.py"),
            Task(description="Fix B", priority=2, source="lint", source_file="b.py"),
            # Groups are full (2). This task has a NEW source_file — should be skipped.
            Task(description="Fix C", priority=3, source="lint", source_file="c.py"),
            # This task matches existing group a.py — must still be grouped.
            Task(description="Fix A2", priority=4, source="lint", source_file="a.py"),
        ]
        groups = coord._partition_tasks(tasks)
        assert len(groups) == 2
        a_group = [g for g in groups if any(t.source_file == "a.py" for t in g)]
        assert len(a_group) == 1
        a_descs = [t.description for t in a_group[0]]
        assert "Fix A" in a_descs
        assert "Fix A2" in a_descs, (
            "Fix A2 should still be grouped with a.py even though c.py was skipped"
        )


class TestDegradationMissingBatchSizeFactor:
    """Test that _partition_tasks doesn't crash when degradation dict lacks batch_size_factor."""

    def test_degradation_missing_batch_size_factor(self, parallel_config):
        """Degradation dict with 'degraded' but no 'batch_size_factor' should use default."""
        parallel_config.parallel.max_workers = 4
        coord = ParallelCoordinator(parallel_config)
        tasks = [
            Task(description=f"Task {i}", priority=3, source="lint")
            for i in range(5)
        ]
        # Missing 'batch_size_factor' key — should not raise KeyError
        degradation = {"degraded": True}
        groups = coord._partition_tasks(tasks, degradation=degradation)
        # With default factor 1.0, all 4 workers should be available
        assert len(groups) == 4


class TestSignalHandlerWorkersCopy:
    """Test that the signal handler snapshots _workers to avoid mutation races."""

    def test_signal_handler_snapshots_workers(self, parallel_config):
        """Signal handler should use list() copy so mutating _workers is safe."""
        import inspect
        from coordinator import ParallelCoordinator
        source = inspect.getsource(ParallelCoordinator)
        assert "list(self._workers)" in source, (
            "Signal handler should snapshot _workers with list() to avoid race"
        )


class TestWorkersLock:
    """Test that _workers list mutations are protected by a lock."""

    def test_workers_lock_exists(self, parallel_config):
        """ParallelCoordinator should have a _workers_lock attribute."""
        coord = ParallelCoordinator(parallel_config)
        assert hasattr(coord, "_workers_lock")
        assert isinstance(coord._workers_lock, type(threading.Lock()))

    def test_signal_handler_acquires_lock(self, parallel_config):
        """Signal handler should acquire _workers_lock when snapshotting workers."""
        import inspect
        from coordinator import ParallelCoordinator
        source = inspect.getsource(ParallelCoordinator)
        assert "self._workers_lock" in source, (
            "Signal handler should use _workers_lock to protect worker list access"
        )


class TestSignalHandlerClaudeRace:
    """Signal handler must not race on worker._claude access."""

    def test_signal_handler_uses_local_variable_for_claude(self, parallel_config):
        """Signal handler should capture worker._claude in a local variable
        to avoid TOCTOU race where _claude becomes None between check and use."""
        import inspect
        from coordinator import ParallelCoordinator
        source = inspect.getsource(ParallelCoordinator)
        # The handler should NOT do `if worker._claude is not None: worker._claude.terminate()`
        # It should capture in a local: `runner = worker._claude; if runner is not None: runner.terminate()`
        assert "runner = worker._claude" in source or "proc = worker._claude" in source or "claude = worker._claude" in source, (
            "Signal handler should capture worker._claude in a local variable "
            "to avoid TOCTOU race between None check and terminate() call"
        )


class TestWorktreeDiskSpaceUnparsableName:
    """Regression: unparsable worktree directory names must not be deleted.

    Previously, if a worktree directory name couldn't be parsed to extract a
    worker ID (e.g. 'worker-data' or 'my-worktree'), wt_id was set to -1,
    which never matched active_ids, causing the directory to be deleted
    unconditionally. The fix skips directories with unparsable names.
    """

    def test_unparsable_worktree_name_not_deleted(self, parallel_config):
        """Worktree dirs with non-numeric suffixes should be skipped, not removed."""
        import shutil

        coord = ParallelCoordinator(parallel_config)
        worktree_base = Path(parallel_config.target_dir) / ".worktrees"

        # Create a worktree with a non-numeric suffix
        (worktree_base / "worker-data").mkdir(parents=True, exist_ok=True)
        # And a parsable but inactive one
        (worktree_base / "worker-99").mkdir(parents=True, exist_ok=True)

        remove_calls = []

        def tracking_remove(path, force=False):
            remove_calls.append(path)

        rmtree_calls = []
        original_rmtree = shutil.rmtree

        def tracking_rmtree(path, **kwargs):
            rmtree_calls.append(path)

        with patch.object(coord.git, "remove_worktree", side_effect=tracking_remove), \
             patch.object(coord.git, "prune_worktrees"), \
             patch("coordinator.shutil.rmtree", side_effect=tracking_rmtree), \
             patch("coordinator.shutil.disk_usage") as mock_usage:
            # Simulate low disk space to trigger cleanup
            mock_usage.return_value = MagicMock(free=10 * 1024 * 1024)  # 10 MB free
            coord._check_worktree_disk_space()

        # "worker-data" should NOT have been touched (unparsable name)
        all_paths = remove_calls + rmtree_calls
        assert not any("worker-data" in str(p) for p in all_paths), (
            "Worktree with unparsable name 'worker-data' should be skipped, not removed"
        )
        # "worker-99" should have been removed (parsable, not active)
        assert any("worker-99" in str(p) for p in all_paths), (
            "Worktree with parsable inactive ID 'worker-99' should be cleaned up"
        )

    def test_worktree_with_no_dash_not_deleted(self, parallel_config):
        """Worktree dir with no dash in name should be skipped."""
        import shutil

        coord = ParallelCoordinator(parallel_config)
        worktree_base = Path(parallel_config.target_dir) / ".worktrees"

        (worktree_base / "tempdir").mkdir(parents=True, exist_ok=True)

        remove_calls = []

        def tracking_remove(path, force=False):
            remove_calls.append(path)

        with patch.object(coord.git, "remove_worktree", side_effect=tracking_remove), \
             patch.object(coord.git, "prune_worktrees"), \
             patch("coordinator.shutil.rmtree"), \
             patch("coordinator.shutil.disk_usage") as mock_usage:
            mock_usage.return_value = MagicMock(free=10 * 1024 * 1024)
            coord._check_worktree_disk_space()

        assert not any("tempdir" in str(p) for p in remove_calls), (
            "Worktree dir without dash should be skipped"
        )


class TestMergeWorkerBranchCleanNoCommit:
    """Test that AI conflict resolution handles clean merges correctly."""

    def test_clean_merge_no_commit_is_completed(self, parallel_config):
        """When merge_no_commit returns True (clean merge), the merge should be
        completed with a commit rather than aborting the clean merge."""
        import inspect
        from coordinator import ParallelCoordinator
        source = inspect.getsource(ParallelCoordinator._merge_worker_branch)
        # The code should check merge_no_commit return value
        assert "merge_clean" in source or "merge_no_commit" in source, (
            "_merge_worker_branch should check merge_no_commit return value"
        )
        # When merge is clean, it should NOT call abort_merge
        # The code should have a path that handles clean merge differently from conflicts
        assert "if merge_clean" in source or "if not merge_clean" in source, (
            "_merge_worker_branch should handle clean merge_no_commit separately"
        )
