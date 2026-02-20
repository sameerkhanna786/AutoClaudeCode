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

    def test_auto_tasks_grouped_by_source(self, parallel_config):
        coord = ParallelCoordinator(parallel_config)
        tasks = [
            Task(description="Lint 1", priority=3, source="lint"),
            Task(description="Lint 2", priority=3, source="lint"),
            Task(description="Todo 1", priority=4, source="todo"),
        ]
        groups = coord._partition_tasks(tasks)
        assert len(groups) >= 1
        # All lint tasks should be in one group
        lint_groups = [g for g in groups if g[0].source == "lint"]
        assert len(lint_groups) == 1

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
