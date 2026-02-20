"""Tests for worker module."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from claude_runner import ClaudeResult
from config_schema import Config, ParallelConfig
from state_lock import LockedStateManager
from task_discovery import Task
from worker import Worker, WorkerResult


@pytest.fixture
def worker_config(tmp_git_repo):
    """Config for worker tests."""
    config = Config()
    config.target_dir = tmp_git_repo
    config.parallel = ParallelConfig(
        enabled=True,
        max_workers=3,
        worktree_base_dir=".worktrees",
    )
    config.paths.state_dir = str(Path(tmp_git_repo) / "state")
    config.paths.history_file = str(Path(tmp_git_repo) / "state" / "history.json")
    config.paths.lock_file = str(Path(tmp_git_repo) / "state" / "lock.pid")
    return config


class TestWorkerInit:
    def test_worker_branch_name(self, worker_config, tmp_git_repo):
        state = MagicMock(spec=LockedStateManager)
        tasks = [Task(description="Fix bug", priority=1, source="lint")]
        worker = Worker(worker_config, tasks, state, worker_id=0, main_repo_dir=tmp_git_repo)
        assert worker.branch_name.startswith("auto-claude/")
        assert "-0" in worker.branch_name

    def test_worker_worktree_dir(self, worker_config, tmp_git_repo):
        state = MagicMock(spec=LockedStateManager)
        tasks = [Task(description="Fix bug", priority=1, source="lint")]
        worker = Worker(worker_config, tasks, state, worker_id=2, main_repo_dir=tmp_git_repo)
        assert "worker-2" in worker.worktree_dir


class TestWorkerWorktree:
    def test_setup_and_cleanup_worktree(self, worker_config, tmp_git_repo):
        """Worker can create and clean up a worktree."""
        state = MagicMock(spec=LockedStateManager)
        tasks = [Task(description="Fix bug", priority=1, source="lint")]
        worker = Worker(worker_config, tasks, state, worker_id=0, main_repo_dir=tmp_git_repo)

        worker._setup_worktree()
        assert Path(worker.worktree_dir).exists()
        assert Path(worker.worktree_dir, "README.md").exists()

        worker.cleanup()
        assert not Path(worker.worktree_dir).exists()


class TestWorkerExecute:
    def test_execute_no_changes(self, worker_config, tmp_git_repo):
        """Worker returns failure when Claude makes no changes."""
        state = MagicMock(spec=LockedStateManager)
        tasks = [Task(description="Fix nothing", priority=1, source="lint")]
        worker = Worker(worker_config, tasks, state, worker_id=0, main_repo_dir=tmp_git_repo)

        with patch.object(Worker, '_setup_worktree'):
            worker.worktree_dir = tmp_git_repo  # use the repo directly
            worker._git = MagicMock()
            worker._git.get_changed_files.return_value = []

            with patch('worker.ClaudeRunner') as mock_cr:
                mock_runner = MagicMock()
                mock_runner.run.return_value = ClaudeResult(success=True, cost_usd=0.5)
                mock_cr.return_value = mock_runner

                with patch('worker.CycleStateWriter'):
                    result = worker.execute()

                # Verify add_dirs is passed (not working_dir)
                run_call = mock_runner.run.call_args
                assert "add_dirs" in run_call.kwargs or (
                    len(run_call.args) == 1  # only prompt positional
                )
                if "add_dirs" in run_call.kwargs:
                    add_dirs = run_call.kwargs["add_dirs"]
                    assert len(add_dirs) == 1
                    assert str(Path(tmp_git_repo).resolve()) in add_dirs[0]

        assert result.success is False
        assert "No files changed" in result.error

    def test_execute_claude_failure(self, worker_config, tmp_git_repo):
        """Worker returns failure when Claude invocation fails."""
        state = MagicMock(spec=LockedStateManager)
        tasks = [Task(description="Fix bug", priority=1, source="lint")]
        worker = Worker(worker_config, tasks, state, worker_id=0, main_repo_dir=tmp_git_repo)

        with patch.object(Worker, '_setup_worktree'):
            worker.worktree_dir = tmp_git_repo

            with patch('worker.ClaudeRunner') as mock_cr:
                mock_runner = MagicMock()
                mock_runner.run.return_value = ClaudeResult(
                    success=False, error="API error", cost_usd=0.1,
                )
                mock_cr.return_value = mock_runner

                with patch('worker.CycleStateWriter'):
                    result = worker.execute()

        assert result.success is False
        assert "API error" in result.error

    def test_worktree_setup_failure(self, worker_config, tmp_git_repo):
        """Worker returns failure when worktree setup fails."""
        state = MagicMock(spec=LockedStateManager)
        tasks = [Task(description="Fix bug", priority=1, source="lint")]
        worker = Worker(worker_config, tasks, state, worker_id=0, main_repo_dir=tmp_git_repo)

        with patch.object(Worker, '_setup_worktree', side_effect=RuntimeError("git error")):
            result = worker.execute()

        assert result.success is False
        assert "Worktree setup failed" in result.error


class TestWorkerPrompt:
    def test_single_task_prompt(self, worker_config, tmp_git_repo):
        state = MagicMock(spec=LockedStateManager)
        tasks = [Task(description="Fix the bug", priority=1, source="lint")]
        worker = Worker(worker_config, tasks, state, worker_id=0, main_repo_dir=tmp_git_repo)

        prompt = worker._build_prompt(tasks, is_batch=False)
        assert "Fix the bug" in prompt
        assert "TASK:" in prompt
        assert "protected files" in prompt.lower()
        # Worktree absolute path should appear in the prompt
        assert str(Path(worker.worktree_dir).resolve()) in prompt
        assert "absolute paths" in prompt.lower()
        # Should NOT tell Claude to run tests
        assert "Do NOT run tests" in prompt

    def test_batch_prompt(self, worker_config, tmp_git_repo):
        state = MagicMock(spec=LockedStateManager)
        tasks = [
            Task(description="Fix bug 1", priority=1, source="lint"),
            Task(description="Fix bug 2", priority=1, source="lint"),
        ]
        worker = Worker(worker_config, tasks, state, worker_id=0, main_repo_dir=tmp_git_repo)

        prompt = worker._build_prompt(tasks, is_batch=True)
        assert "TASKS:" in prompt
        assert "Fix bug 1" in prompt
        assert "Fix bug 2" in prompt
        # Worktree absolute path should appear in the prompt
        assert str(Path(worker.worktree_dir).resolve()) in prompt
        assert "absolute paths" in prompt.lower()
        # Should NOT tell Claude to run tests
        assert "Do NOT run tests" in prompt


class TestWorkerCommitMessage:
    def test_single_task_message(self, worker_config, tmp_git_repo):
        state = MagicMock(spec=LockedStateManager)
        tasks = [Task(description="Fix the lint error", priority=1, source="lint")]
        worker = Worker(worker_config, tasks, state, worker_id=0, main_repo_dir=tmp_git_repo)

        msg = worker._build_commit_message(tasks, is_batch=False)
        assert "Fix the lint error" in msg

    def test_batch_message(self, worker_config, tmp_git_repo):
        state = MagicMock(spec=LockedStateManager)
        tasks = [
            Task(description="Fix A", priority=1, source="lint"),
            Task(description="Fix B", priority=1, source="lint"),
        ]
        worker = Worker(worker_config, tasks, state, worker_id=0, main_repo_dir=tmp_git_repo)

        msg = worker._build_commit_message(tasks, is_batch=True)
        assert "2 tasks" in msg
        assert "Fix A" in msg
        assert "Fix B" in msg


class TestWorkerMainRepoSafetyCheck:
    def test_detects_main_repo_modification(self, worker_config, tmp_git_repo):
        """Worker fails when Claude modifies files in the main repo."""
        state = MagicMock(spec=LockedStateManager)
        tasks = [Task(description="Fix bug", priority=1, source="lint")]
        worker = Worker(worker_config, tasks, state, worker_id=0, main_repo_dir=tmp_git_repo)

        with patch.object(Worker, '_setup_worktree'):
            worker.worktree_dir = tmp_git_repo
            worker._git = MagicMock()
            worker._git.get_changed_files.return_value = ["fix.py"]

            with patch('worker.ClaudeRunner') as mock_cr:
                mock_runner = MagicMock()
                mock_runner.run.return_value = ClaudeResult(
                    success=True, cost_usd=0.5, result_text="done",
                )
                mock_cr.return_value = mock_runner

                # Simulate main repo getting dirty after Claude runs
                from git_manager import GitManager
                with patch('worker.GitManager') as MockGitManager:
                    main_git_mock = MagicMock()
                    # Before Claude: clean; After Claude: dirty
                    main_git_mock.get_changed_files.side_effect = [
                        [],                    # pre-state: clean
                        ["orchestrator.py"],   # post-state: dirty
                    ]
                    MockGitManager.return_value = main_git_mock

                    with patch('worker.CycleStateWriter'):
                        result = worker.execute()

        assert result.success is False
        assert "main repo" in result.error.lower()

    def test_prompt_contains_warning(self, worker_config, tmp_git_repo):
        """Worker prompt contains warning about not modifying files outside worktree."""
        state = MagicMock(spec=LockedStateManager)
        tasks = [Task(description="Fix the bug", priority=1, source="lint")]
        worker = Worker(worker_config, tasks, state, worker_id=0, main_repo_dir=tmp_git_repo)

        # Test single-task prompt
        prompt = worker._build_prompt(tasks, is_batch=False)
        assert "Do NOT modify any files outside" in prompt

        # Test batch prompt
        prompt_batch = worker._build_prompt(tasks, is_batch=True)
        assert "Do NOT modify any files outside" in prompt_batch
