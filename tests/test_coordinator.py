"""Tests for coordinator module."""

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


class TestMergeWorkerBranch:
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
