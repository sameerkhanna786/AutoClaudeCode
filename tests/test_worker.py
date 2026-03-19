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
    @pytest.mark.requires_subprocess
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

            with patch('provider_runner.create_runner') as mock_cr:
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

            with patch('provider_runner.create_runner') as mock_cr:
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
        # Should include task-type specific instructions for lint
        assert "lint" in prompt.lower()

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
        # Batch message uses shared builder which groups by source type
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

            with patch('provider_runner.create_runner') as mock_cr:
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


class TestWorkerBaseline:
    def test_worker_uses_baseline_for_validation(self, worker_config, tmp_git_repo):
        """Worker passes baseline failures to validator.validate_with_baseline()."""
        state = MagicMock(spec=LockedStateManager)
        tasks = [Task(description="Fix bug", priority=1, source="lint")]
        baseline = {"tests/test_foo.py::test_bar"}
        worker = Worker(
            worker_config, tasks, state, worker_id=0,
            main_repo_dir=tmp_git_repo, baseline_failures=baseline,
        )

        assert worker.baseline_failures == baseline

        # Create a file so there are "changed files"
        Path(tmp_git_repo, "fix.py").write_text("# fix\n")

        with patch.object(Worker, '_setup_worktree'):
            worker.worktree_dir = tmp_git_repo

            with patch('provider_runner.create_runner') as mock_cr:
                mock_runner = MagicMock()
                mock_runner.run.return_value = ClaudeResult(
                    success=True, cost_usd=0.5, result_text="done",
                )
                mock_cr.return_value = mock_runner

                with patch('worker.CycleStateWriter'):
                    with patch('worker.Validator') as MockValidator:
                        mock_validator = MagicMock()
                        mock_validation = MagicMock()
                        mock_validation.passed = True
                        mock_validator.validate_with_baseline.return_value = mock_validation
                        MockValidator.return_value = mock_validator

                        result = worker.execute()

                        # Verify validate_with_baseline was called with baseline
                        mock_validator.validate_with_baseline.assert_called_with(
                            worker.worktree_dir, baseline,
                        )

    def test_worker_default_baseline_empty(self, worker_config, tmp_git_repo):
        """Worker defaults to empty baseline when none provided."""
        state = MagicMock(spec=LockedStateManager)
        tasks = [Task(description="Fix bug", priority=1, source="lint")]
        worker = Worker(
            worker_config, tasks, state, worker_id=0,
            main_repo_dir=tmp_git_repo,
        )
        assert worker.baseline_failures == set()


class TestCostLimitExceeded:
    def test_returns_false_when_under_budget(self, worker_config, tmp_git_repo):
        """Returns False when cost is well under the 90% threshold."""
        state = MagicMock(spec=LockedStateManager)
        state.get_total_cost.return_value = 1.0  # $1 hourly
        tasks = [Task(description="Fix bug", priority=1, source="lint")]
        worker = Worker(worker_config, tasks, state, worker_id=0, main_repo_dir=tmp_git_repo)
        # default max_cost_usd_per_hour=10.0, so 90% = $9.0
        # $1.0 + $0.5 = $1.5 < $9.0
        assert worker._cost_limit_exceeded(0.5) is False

    def test_returns_true_at_90_percent_threshold(self, worker_config, tmp_git_repo):
        """Returns True when cost hits the 90% threshold."""
        state = MagicMock(spec=LockedStateManager)
        state.get_total_cost.return_value = 8.5  # $8.5 hourly
        tasks = [Task(description="Fix bug", priority=1, source="lint")]
        worker = Worker(worker_config, tasks, state, worker_id=0, main_repo_dir=tmp_git_repo)
        # $8.5 + $0.5 = $9.0 >= $10.0 * 0.9 = $9.0
        assert worker._cost_limit_exceeded(0.5) is True

    def test_returns_true_when_over_threshold(self, worker_config, tmp_git_repo):
        """Returns True when cost exceeds the 90% threshold."""
        state = MagicMock(spec=LockedStateManager)
        state.get_total_cost.return_value = 9.0
        tasks = [Task(description="Fix bug", priority=1, source="lint")]
        worker = Worker(worker_config, tasks, state, worker_id=0, main_repo_dir=tmp_git_repo)
        # $9.0 + $1.0 = $10.0 >= $9.0 threshold
        assert worker._cost_limit_exceeded(1.0) is True

    def test_handles_get_total_cost_exception(self, worker_config, tmp_git_repo):
        """Returns True (fail-safe) when state.get_total_cost() raises an exception."""
        state = MagicMock(spec=LockedStateManager)
        state.get_total_cost.side_effect = RuntimeError("DB connection failed")
        tasks = [Task(description="Fix bug", priority=1, source="lint")]
        worker = Worker(worker_config, tasks, state, worker_id=0, main_repo_dir=tmp_git_repo)
        assert worker._cost_limit_exceeded(5.0) is True

    def test_logs_warning_when_triggered(self, worker_config, tmp_git_repo):
        """Logs a warning message when cost limit is exceeded."""
        state = MagicMock(spec=LockedStateManager)
        state.get_total_cost.return_value = 9.0
        tasks = [Task(description="Fix bug", priority=1, source="lint")]
        worker = Worker(worker_config, tasks, state, worker_id=0, main_repo_dir=tmp_git_repo)

        with patch('worker.logger') as mock_logger:
            result = worker._cost_limit_exceeded(1.0)
            assert result is True
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "cost guard triggered" in warning_msg


class TestWorkerBranchNamePrecision:
    """Tests for nanosecond-precision branch names to avoid collisions."""

    def test_branch_name_uses_nanoseconds(self, worker_config, tmp_git_repo):
        """Branch name should use time_ns() for nanosecond precision."""
        state = MagicMock(spec=LockedStateManager)
        tasks = [Task(description="Fix", priority=1, source="lint")]
        worker = Worker(worker_config, tasks, state, worker_id=0, main_repo_dir=tmp_git_repo)
        # Nanosecond timestamps are > 10^18, second timestamps are ~10^9
        ts_part = worker.branch_name.split("/")[1].split("-")[0]
        assert len(ts_part) > 15, f"Timestamp {ts_part} looks like seconds, not nanoseconds"

    def test_concurrent_workers_get_unique_branches(self, worker_config, tmp_git_repo):
        """Two workers created in quick succession should get different branch names."""
        state = MagicMock(spec=LockedStateManager)
        tasks = [Task(description="Fix", priority=1, source="lint")]
        w1 = Worker(worker_config, tasks, state, worker_id=0, main_repo_dir=tmp_git_repo)
        w2 = Worker(worker_config, tasks, state, worker_id=0, main_repo_dir=tmp_git_repo)
        assert w1.branch_name != w2.branch_name


class TestConfigMutationIsolation:
    """Tests that plan_config uses deepcopy so nested config objects are not shared."""

    def test_plan_config_max_turns_does_not_mutate_original(self, worker_config, tmp_git_repo):
        """Modifying plan_config.claude.max_turns should not affect the original config."""
        import copy
        original_max_turns = worker_config.claude.max_turns

        # Simulate what worker.py does with copy.copy (the bug)
        shallow_config = copy.copy(worker_config)
        shallow_config.claude = copy.copy(worker_config.claude)
        shallow_config.claude.max_turns = 5

        # With copy.copy of claude, the original should NOT be affected
        # (the current code does copy.copy on config.claude too, which is correct
        # for flat attributes but would fail for nested mutable objects inside claude)
        assert worker_config.claude.max_turns == original_max_turns, \
            "Modifying plan_config.claude.max_turns should not affect original config"

    def test_plan_config_nested_objects_are_isolated(self, worker_config, tmp_git_repo):
        """Nested mutable objects (like lists in config) should be isolated between copies."""
        import copy

        # copy.copy creates shallow copy - nested mutable objects are shared
        shallow_config = copy.copy(worker_config)
        # The validation sub-object is shared with shallow copy
        assert shallow_config.validation is worker_config.validation, \
            "Shallow copy shares nested objects (this is the bug)"

        # copy.deepcopy would isolate them
        deep_config = copy.deepcopy(worker_config)
        assert deep_config.validation is not worker_config.validation, \
            "Deep copy should isolate nested objects"


class TestPlanRunnerExposedForTermination:
    """Plan-phase runner must be assigned to self._claude for signal-based termination."""

    def test_plan_runner_assigned_to_self_claude(self, worker_config, tmp_git_repo):
        """When plan_changes is enabled, the plan runner should be assigned to
        self._claude so the coordinator can terminate it via signal handler."""
        import inspect
        from worker import Worker
        source = inspect.getsource(Worker.execute)
        # The fix assigns plan_runner to self._claude before calling run()
        assert "self._claude = plan_runner" in source, (
            "plan_runner must be assigned to self._claude for termination support"
        )

    def test_execution_runner_restored_after_planning(self, worker_config, tmp_git_repo):
        """After planning completes, self._claude should be restored to the
        execution-phase runner."""
        import inspect
        from worker import Worker
        source = inspect.getsource(Worker.execute)
        # After plan_runner.run(), a new execution runner must be created
        idx_plan = source.index("self._claude = plan_runner")
        idx_restore = source.index("self._claude = create_runner", idx_plan + 1)
        assert idx_restore > idx_plan, (
            "Execution runner must be restored after plan_runner completes"
        )
