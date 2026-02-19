"""Tests for orchestrator module."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from claude_runner import ClaudeResult
from config_schema import Config
from git_manager import Snapshot
from orchestrator import Orchestrator
from task_discovery import Task
from validator import ValidationResult, ValidationStep


@pytest.fixture
def config(tmp_path):
    cfg = Config()
    cfg.target_dir = str(tmp_path)
    cfg.paths.history_file = str(tmp_path / "state" / "history.json")
    cfg.paths.lock_file = str(tmp_path / "state" / "lock.pid")
    cfg.paths.feedback_dir = str(tmp_path / "feedback")
    cfg.paths.feedback_done_dir = str(tmp_path / "feedback" / "done")
    cfg.paths.feedback_failed_dir = str(tmp_path / "feedback" / "failed")
    cfg.paths.backup_dir = str(tmp_path / "state" / "backups")
    # Disable validation commands to avoid subprocess calls
    cfg.validation.test_command = ""
    cfg.validation.lint_command = ""
    cfg.validation.build_command = ""
    # Disable batch mode for existing tests
    cfg.orchestrator.batch_mode = False
    return cfg


@pytest.fixture
def orch(config):
    with patch("orchestrator.GitManager") as MockGit, \
         patch("orchestrator.ClaudeRunner") as MockClaude, \
         patch("orchestrator.TaskDiscovery") as MockDisc, \
         patch("orchestrator.Validator") as MockVal, \
         patch("orchestrator.resolve_model_id", return_value=None), \
         patch("subprocess.run") as mock_sp:

        mock_sp.return_value = MagicMock(returncode=0)
        mock_git = MockGit.return_value
        mock_git.create_snapshot.return_value = Snapshot(commit_hash="a" * 40)
        mock_git.capture_worktree_state.return_value = set()
        mock_git.get_changed_files.return_value = ["fix.py"]
        mock_git.get_new_changed_files.return_value = ["fix.py"]
        mock_git.is_clean.return_value = True
        mock_git.commit.return_value = "b" * 40

        mock_claude = MockClaude.return_value
        mock_claude.run.return_value = ClaudeResult(
            success=True,
            result_text="Fixed it",
            cost_usd=0.05,
            duration_seconds=10.0,
        )

        mock_disc = MockDisc.return_value
        mock_disc.discover_all.return_value = [
            Task(description="Fix bug in foo.py", priority=2, source="test_failure"),
        ]

        mock_val = MockVal.return_value
        mock_val.validate.return_value = ValidationResult(passed=True, steps=[])

        o = Orchestrator(config)
        o.git = mock_git
        o.claude = mock_claude
        o.discovery = mock_disc
        o.validator = mock_val
        yield o


class TestOrchestrator:
    def test_successful_cycle(self, orch):
        orch._cycle()
        orch.git.commit.assert_called_once()
        orch.git.rollback.assert_not_called()

    def test_failed_validation_causes_rollback(self, orch):
        orch.config.orchestrator.max_validation_retries = 0
        orch.validator.validate.return_value = ValidationResult(
            passed=False,
            steps=[ValidationStep(name="tests", command="pytest", passed=False)],
        )
        orch._cycle()
        orch.git.rollback.assert_called_once()
        orch.git.commit.assert_not_called()

    def test_claude_failure_causes_rollback(self, orch):
        orch.claude.run.return_value = ClaudeResult(
            success=False, error="Timed out",
        )
        orch._cycle()
        orch.git.rollback.assert_called_once()

    def test_no_tasks_found(self, orch):
        orch.discovery.discover_all.return_value = []
        orch._cycle()
        orch.claude.run.assert_not_called()
        orch.git.commit.assert_not_called()

    def test_no_files_changed(self, orch):
        orch.git.get_new_changed_files.return_value = []
        orch._cycle()
        orch.git.commit.assert_not_called()

    def test_protected_file_violation_causes_rollback(self, orch):
        orch.git.get_new_changed_files.return_value = ["main.py", "fix.py"]
        orch._cycle()
        orch.git.rollback.assert_called_once()
        orch.git.commit.assert_not_called()

    def test_nested_path_not_blocked_by_protected_basename(self, orch):
        orch.git.get_new_changed_files.return_value = ["src/main.py", "fix.py"]
        orch._cycle()
        orch.git.rollback.assert_not_called()
        orch.git.commit.assert_called_once()

    def test_run_once(self, orch):
        orch.run(once=True)
        # Should have run exactly one cycle
        orch.git.create_snapshot.assert_called_once()

    def test_build_prompt(self, orch):
        task = Task(description="Fix the bug", priority=2, source="test_failure")
        prompt = orch._build_prompt(task)
        assert "Fix the bug" in prompt
        assert "main.py" in prompt
        assert "Do NOT" in prompt

    def test_feedback_takes_priority(self, orch):
        feedback_task = Task(
            description="Developer task",
            priority=1,
            source="feedback",
            source_file="/tmp/feedback/task.md",
        )
        orch.feedback.get_pending_feedback = MagicMock(return_value=[feedback_task])
        task = orch._pick_task()
        assert task.source == "feedback"
        assert task.description == "Developer task"

    def test_feedback_retry_limit_skips_failed_tasks(self, orch):
        """Feedback tasks that have failed max_feedback_retries times should be skipped."""
        orch.config.orchestrator.max_feedback_retries = 2
        feedback_task = Task(
            description="Broken task",
            priority=1,
            source="feedback",
            source_file="/tmp/feedback/broken.md",
        )
        orch.feedback.get_pending_feedback = MagicMock(return_value=[feedback_task])
        orch.state.get_task_failure_count = MagicMock(return_value=2)
        orch.feedback.mark_failed = MagicMock()

        task = orch._pick_task()

        orch.feedback.mark_failed.assert_called_once_with("/tmp/feedback/broken.md")
        # Should fall through to auto-discovered tasks
        assert task is None or task.source != "feedback"

    def test_feedback_retry_limit_allows_under_threshold(self, orch):
        """Feedback tasks under the retry limit should still be selected."""
        orch.config.orchestrator.max_feedback_retries = 3
        feedback_task = Task(
            description="Retryable task",
            priority=1,
            source="feedback",
            source_file="/tmp/feedback/retry.md",
        )
        orch.feedback.get_pending_feedback = MagicMock(return_value=[feedback_task])
        orch.state.get_task_failure_count = MagicMock(return_value=2)

        task = orch._pick_task()
        assert task.source == "feedback"
        assert task.description == "Retryable task"

    def test_feedback_retry_limit_no_source_file(self, orch):
        """Feedback tasks without source_file should still be skipped but not crash."""
        orch.config.orchestrator.max_feedback_retries = 1
        feedback_task = Task(
            description="No file task",
            priority=1,
            source="feedback",
            source_file=None,
        )
        orch.feedback.get_pending_feedback = MagicMock(return_value=[feedback_task])
        orch.state.get_task_failure_count = MagicMock(return_value=1)
        orch.feedback.mark_failed = MagicMock()

        task = orch._pick_task()
        # Should not call mark_failed since source_file is None
        orch.feedback.mark_failed.assert_not_called()

    def test_no_tasks_logs_warning_when_discovery_disabled(self, orch, caplog):
        """When no discovery methods are enabled, should log a warning."""
        import logging
        orch.discovery.discover_all.return_value = []
        orch.feedback.get_pending_feedback = MagicMock(return_value=[])
        orch.config.discovery.enable_test_failures = False
        orch.config.discovery.enable_lint_errors = False
        orch.config.discovery.enable_todos = False
        orch.config.discovery.enable_coverage = False
        orch.config.discovery.enable_claude_ideas = False
        orch.config.discovery.enable_quality_review = False
        with caplog.at_level(logging.WARNING):
            orch._cycle()
        assert any("no discovery methods enabled" in r.message for r in caplog.records)

    def test_no_tasks_logs_enabled_methods(self, orch, caplog):
        """When discovery methods are enabled but no actionable tasks, log which methods are enabled."""
        import logging
        orch.discovery.discover_all.return_value = []
        orch.feedback.get_pending_feedback = MagicMock(return_value=[])
        orch.config.discovery.enable_test_failures = True
        orch.config.discovery.enable_lint_errors = False
        orch.config.discovery.enable_todos = True
        orch.config.discovery.enable_coverage = False
        orch.config.discovery.enable_claude_ideas = True
        orch.config.discovery.enable_quality_review = False
        with caplog.at_level(logging.INFO):
            orch._cycle()
        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("test_failures" in m and "todos" in m and "claude_ideas" in m for m in info_messages)
        assert any("pending feedback: no" in m for m in info_messages)


@pytest.fixture
def batch_config(tmp_path):
    cfg = Config()
    cfg.target_dir = str(tmp_path)
    cfg.paths.history_file = str(tmp_path / "state" / "history.json")
    cfg.paths.lock_file = str(tmp_path / "state" / "lock.pid")
    cfg.paths.feedback_dir = str(tmp_path / "feedback")
    cfg.paths.feedback_done_dir = str(tmp_path / "feedback" / "done")
    cfg.paths.feedback_failed_dir = str(tmp_path / "feedback" / "failed")
    cfg.paths.backup_dir = str(tmp_path / "state" / "backups")
    cfg.validation.test_command = ""
    cfg.validation.lint_command = ""
    cfg.validation.build_command = ""
    cfg.orchestrator.batch_mode = True
    cfg.orchestrator.plan_changes = True
    cfg.orchestrator.max_tasks_per_cycle = 10
    cfg.orchestrator.initial_batch_size = 10
    cfg.orchestrator.max_batch_size = 10
    return cfg


@pytest.fixture
def orch_batch(batch_config):
    with patch("orchestrator.GitManager") as MockGit, \
         patch("orchestrator.ClaudeRunner") as MockClaude, \
         patch("orchestrator.TaskDiscovery") as MockDisc, \
         patch("orchestrator.Validator") as MockVal, \
         patch("orchestrator.resolve_model_id", return_value=None), \
         patch("subprocess.run") as mock_sp:

        mock_sp.return_value = MagicMock(returncode=0)
        mock_git = MockGit.return_value
        mock_git.create_snapshot.return_value = Snapshot(commit_hash="a" * 40)
        mock_git.capture_worktree_state.return_value = set()
        mock_git.get_changed_files.return_value = ["fix.py", "bar.py"]
        mock_git.get_new_changed_files.return_value = ["fix.py", "bar.py"]
        mock_git.is_clean.return_value = True
        mock_git.commit.return_value = "c" * 40

        mock_claude = MockClaude.return_value
        mock_claude.run.side_effect = [
            ClaudeResult(success=True, result_text="Plan: fix both files", cost_usd=0.03, duration_seconds=5.0),
            ClaudeResult(success=True, result_text="Executed plan", cost_usd=0.05, duration_seconds=10.0),
        ]

        mock_disc = MockDisc.return_value
        mock_disc.discover_all.return_value = [
            Task(description="Fix bug in foo.py", priority=2, source="test_failure"),
            Task(description="Address TODO in bar.py", priority=3, source="todo"),
            Task(description="Improve error handling", priority=4, source="claude_idea"),
        ]

        mock_val = MockVal.return_value
        mock_val.validate.return_value = ValidationResult(passed=True, steps=[])

        o = Orchestrator(batch_config)
        o.git = mock_git
        o.claude = mock_claude
        o.discovery = mock_disc
        o.validator = mock_val
        yield o


class TestBatchMode:
    def test_gather_tasks_returns_multiple(self, orch_batch):
        tasks = orch_batch._gather_tasks()
        assert len(tasks) == 3

    def test_gather_tasks_respects_max_cap(self, orch_batch):
        orch_batch.state.compute_adaptive_batch_size = MagicMock(return_value=2)
        tasks = orch_batch._gather_tasks()
        assert len(tasks) == 2

    def test_gather_tasks_excludes_recently_attempted(self, orch_batch):
        orch_batch.state.was_recently_attempted = MagicMock(
            side_effect=lambda desc, task_key="": desc == "Fix bug in foo.py"
        )
        tasks = orch_batch._gather_tasks()
        assert all(t.description != "Fix bug in foo.py" for t in tasks)
        assert len(tasks) == 2

    def test_gather_tasks_feedback_first(self, orch_batch):
        feedback_task = Task(
            description="Developer request",
            priority=1,
            source="feedback",
            source_file="/tmp/feedback/req.md",
        )
        orch_batch.feedback.get_pending_feedback = MagicMock(return_value=[feedback_task])
        tasks = orch_batch._gather_tasks()
        assert tasks[0].source == "feedback"
        assert tasks[0].description == "Developer request"

    def test_gather_tasks_single_when_batch_off(self, orch_batch):
        orch_batch.config.orchestrator.batch_mode = False
        tasks = orch_batch._gather_tasks()
        assert len(tasks) == 1

    def test_successful_batch_cycle(self, orch_batch):
        orch_batch._cycle()
        orch_batch.git.commit.assert_called_once()
        commit_msg = orch_batch.git.commit.call_args[0][0]
        assert "[auto]" not in commit_msg
        assert "batch(" not in commit_msg
        orch_batch.git.rollback.assert_called_once()  # once for plan cleanup

    def test_batch_cycle_records_all_descriptions(self, orch_batch):
        orch_batch.state.record_cycle = MagicMock()
        orch_batch._cycle()
        # Check the recorded cycle via state.record_cycle
        record_call = orch_batch.state.record_cycle.call_args[0][0]
        assert len(record_call.task_descriptions) == 3
        assert "Fix bug in foo.py" in record_call.task_descriptions
        assert "Address TODO in bar.py" in record_call.task_descriptions

    def test_batch_planning_failure_rolls_back(self, orch_batch):
        orch_batch.claude.run.side_effect = [
            ClaudeResult(success=False, error="Planning timeout"),
        ]
        orch_batch._cycle()
        orch_batch.git.rollback.assert_called()
        orch_batch.git.commit.assert_not_called()

    def test_batch_marks_all_feedback_done(self, orch_batch):
        feedback1 = Task(description="Task A", priority=1, source="feedback", source_file="/tmp/a.md")
        feedback2 = Task(description="Task B", priority=1, source="feedback", source_file="/tmp/b.md")
        orch_batch.feedback.get_pending_feedback = MagicMock(return_value=[feedback1, feedback2])
        orch_batch.feedback.mark_done = MagicMock()
        orch_batch.discovery.discover_all.return_value = []
        # Reset side_effect for 2 tasks
        orch_batch.claude.run.side_effect = [
            ClaudeResult(success=True, result_text="Plan", cost_usd=0.02, duration_seconds=3.0),
            ClaudeResult(success=True, result_text="Done", cost_usd=0.04, duration_seconds=7.0),
        ]
        orch_batch._cycle()
        assert orch_batch.feedback.mark_done.call_count == 2
        orch_batch.feedback.mark_done.assert_any_call("/tmp/a.md")
        orch_batch.feedback.mark_done.assert_any_call("/tmp/b.md")

    def test_format_task_list(self, orch_batch):
        tasks = [
            Task(description="Fix bug", priority=2, source="test_failure"),
            Task(description="Refactor foo", priority=3, source="todo"),
        ]
        result = orch_batch._format_task_list(tasks)
        assert "1. Fix bug [test_failure]" in result
        assert "2. Refactor foo [todo]" in result

    def test_batch_plan_prompt_includes_test_and_readme_checks(self, orch_batch):
        tasks = [
            Task(description="Fix bug", priority=2, source="test_failure"),
            Task(description="Refactor foo", priority=3, source="todo"),
        ]
        prompt = orch_batch._build_batch_plan_prompt(tasks)
        assert "NEW tests" in prompt
        assert "README.md" in prompt
        assert "3." in prompt  # task_count = len(tasks) + 1
        assert "4." in prompt  # task_count_plus1 = len(tasks) + 2

    def test_build_batch_commit_message(self, orch_batch):
        tasks = [
            Task(description="Fix bug in foo.py", priority=2, source="test_failure"),
            Task(description="Address TODO", priority=3, source="todo"),
        ]
        msg = orch_batch._build_batch_commit_message(tasks)
        assert "[auto]" not in msg
        assert "batch(" not in msg
        # Should contain descriptions in body
        assert "Fix bug in foo.py" in msg or "foo.py" in msg
        assert "Address TODO" in msg or "TODO" in msg

    def test_batch_mode_no_plan_uses_batch_prompt(self, orch_batch):
        """When batch_mode=True and plan_changes=False, all tasks should be
        included in a single-shot batch prompt (not just the first one)."""
        orch_batch.config.orchestrator.plan_changes = False
        # With plan_changes=False, only one Claude call should happen
        orch_batch.claude.run.side_effect = None
        orch_batch.claude.run.return_value = ClaudeResult(
            success=True, result_text="Done", cost_usd=0.05, duration_seconds=10.0,
        )
        orch_batch._cycle()
        # Should have been called exactly once (no plan phase)
        assert orch_batch.claude.run.call_count == 1
        prompt = orch_batch.claude.run.call_args[0][0]
        # All three task descriptions should appear in the prompt
        assert "Fix bug in foo.py" in prompt
        assert "Address TODO in bar.py" in prompt
        assert "Improve error handling" in prompt
        # Should use the batch prompt template, not the single-task template
        assert "batch of tasks" in prompt


class TestSyntaxCheckFiles:
    def test_syntax_check_files_reports_line_number(self, tmp_path):
        """_syntax_check_files should include file name and line number in the error."""
        cfg = Config()
        cfg.target_dir = str(tmp_path)
        cfg.paths.history_file = str(tmp_path / "state" / "history.json")
        cfg.paths.lock_file = str(tmp_path / "state" / "lock.pid")
        cfg.paths.feedback_dir = str(tmp_path / "feedback")
        cfg.paths.feedback_done_dir = str(tmp_path / "feedback" / "done")
        cfg.paths.feedback_failed_dir = str(tmp_path / "feedback" / "failed")
        cfg.paths.backup_dir = str(tmp_path / "state" / "backups")
        cfg.orchestrator.self_improve = True

        with patch("orchestrator.GitManager"), \
             patch("orchestrator.ClaudeRunner"), \
             patch("orchestrator.TaskDiscovery"), \
             patch("orchestrator.Validator"), \
             patch("orchestrator.resolve_model_id", return_value=None), \
             patch("subprocess.run") as mock_sp:
            mock_sp.return_value = MagicMock(returncode=0)
            o = Orchestrator(cfg)

        # Create a .py file with a syntax error on line 3
        bad_file = tmp_path / "broken.py"
        bad_file.write_text("x = 1\ny = 2\nz = (\n")

        result = o._syntax_check_files(["broken.py"])
        assert result is not None
        assert "broken.py" in result
        assert "line" in result.lower()
        # Line number should be present (the error is on line 3 or the EOF)
        assert "3" in result or "4" in result

    def test_syntax_check_files_logs_warning(self, tmp_path, caplog):
        """_syntax_check_files should log a warning with file, line, and offset."""
        import logging

        cfg = Config()
        cfg.target_dir = str(tmp_path)
        cfg.paths.history_file = str(tmp_path / "state" / "history.json")
        cfg.paths.lock_file = str(tmp_path / "state" / "lock.pid")
        cfg.paths.feedback_dir = str(tmp_path / "feedback")
        cfg.paths.feedback_done_dir = str(tmp_path / "feedback" / "done")
        cfg.paths.feedback_failed_dir = str(tmp_path / "feedback" / "failed")
        cfg.paths.backup_dir = str(tmp_path / "state" / "backups")
        cfg.orchestrator.self_improve = True

        with patch("orchestrator.GitManager"), \
             patch("orchestrator.ClaudeRunner"), \
             patch("orchestrator.TaskDiscovery"), \
             patch("orchestrator.Validator"), \
             patch("orchestrator.resolve_model_id", return_value=None), \
             patch("subprocess.run") as mock_sp:
            mock_sp.return_value = MagicMock(returncode=0)
            o = Orchestrator(cfg)

        bad_file = tmp_path / "broken.py"
        bad_file.write_text("def foo(\n")

        with caplog.at_level(logging.WARNING):
            o._syntax_check_files(["broken.py"])

        assert any(
            "broken.py" in r.message and "Syntax error" in r.message
            for r in caplog.records
        )


class TestCycleTimeout:
    def test_cycle_timeout_wraps_claude_call(self, tmp_path):
        """_run_claude_with_timeout returns a failed result when the timeout fires."""
        import concurrent.futures

        cfg = Config()
        cfg.target_dir = str(tmp_path)
        cfg.paths.history_file = str(tmp_path / "state" / "history.json")
        cfg.paths.lock_file = str(tmp_path / "state" / "lock.pid")
        cfg.paths.feedback_dir = str(tmp_path / "feedback")
        cfg.paths.feedback_done_dir = str(tmp_path / "feedback" / "done")
        cfg.paths.feedback_failed_dir = str(tmp_path / "feedback" / "failed")
        cfg.paths.backup_dir = str(tmp_path / "state" / "backups")
        cfg.orchestrator.cycle_timeout_seconds = 1  # very short timeout

        with patch("orchestrator.GitManager"), \
             patch("orchestrator.ClaudeRunner") as MockClaude, \
             patch("orchestrator.TaskDiscovery"), \
             patch("orchestrator.Validator"), \
             patch("orchestrator.resolve_model_id", return_value=None), \
             patch("subprocess.run") as mock_sp:
            mock_sp.return_value = MagicMock(returncode=0)
            o = Orchestrator(cfg)
            # Make claude.run hang longer than the timeout
            import threading

            def slow_run(prompt, working_dir=None):
                time.sleep(10)
                return ClaudeResult(success=True, result_text="Done")

            o.claude.run = slow_run
            result = o._run_claude_with_timeout("test prompt")
            assert result.success is False
            assert "timeout" in result.error.lower()

    def test_cycle_timeout_success(self, tmp_path):
        """_run_claude_with_timeout returns the result when the call completes within timeout."""
        cfg = Config()
        cfg.target_dir = str(tmp_path)
        cfg.paths.history_file = str(tmp_path / "state" / "history.json")
        cfg.paths.lock_file = str(tmp_path / "state" / "lock.pid")
        cfg.paths.feedback_dir = str(tmp_path / "feedback")
        cfg.paths.feedback_done_dir = str(tmp_path / "feedback" / "done")
        cfg.paths.feedback_failed_dir = str(tmp_path / "feedback" / "failed")
        cfg.paths.backup_dir = str(tmp_path / "state" / "backups")
        cfg.orchestrator.cycle_timeout_seconds = 60

        with patch("orchestrator.GitManager"), \
             patch("orchestrator.ClaudeRunner"), \
             patch("orchestrator.TaskDiscovery"), \
             patch("orchestrator.Validator"), \
             patch("orchestrator.resolve_model_id", return_value=None), \
             patch("subprocess.run") as mock_sp:
            mock_sp.return_value = MagicMock(returncode=0)
            o = Orchestrator(cfg)
            o.claude.run = MagicMock(return_value=ClaudeResult(
                success=True, result_text="Done", cost_usd=0.01, duration_seconds=2.0,
            ))
            result = o._run_claude_with_timeout("test prompt")
            assert result.success is True
            assert result.result_text == "Done"


class TestAdaptiveBatchSizing:
    def test_gather_tasks_uses_adaptive_size(self, orch_batch):
        orch_batch.state.compute_adaptive_batch_size = MagicMock(return_value=2)
        tasks = orch_batch._gather_tasks()
        assert len(tasks) == 2
        orch_batch.state.compute_adaptive_batch_size.assert_called_once()

    def test_cycle_record_includes_batch_size_and_keys(self, orch_batch):
        tasks = [
            Task(description="Fix bug in foo.py", priority=2, source="test_failure"),
            Task(description="Address TODO in bar.py", priority=3, source="todo",
                 source_file="bar.py", line_number=5),
        ]
        record = orch_batch._make_cycle_record(tasks, success=True)
        assert record.batch_size == 2
        assert len(record.task_keys) == 2
        assert record.task_keys[1] == "todo:bar.py:5"

    def test_gather_tasks_dedup_by_key(self, orch_batch):
        """Tasks matching by key should be deduped even with different descriptions."""
        # Make was_recently_attempted return True when key matches
        def mock_recently_attempted(desc, task_key=""):
            return task_key == "test_failure:Fix bug in foo.py"
        orch_batch.state.was_recently_attempted = MagicMock(side_effect=mock_recently_attempted)
        tasks = orch_batch._gather_tasks()
        assert all(t.description != "Fix bug in foo.py" for t in tasks)
        assert len(tasks) == 2


class TestValidationRetry:
    """Tests for retry-on-validation-failure behavior."""

    @pytest.fixture
    def retry_orch(self, tmp_path):
        cfg = Config()
        cfg.target_dir = str(tmp_path)
        cfg.paths.history_file = str(tmp_path / "state" / "history.json")
        cfg.paths.lock_file = str(tmp_path / "state" / "lock.pid")
        cfg.paths.feedback_dir = str(tmp_path / "feedback")
        cfg.paths.feedback_done_dir = str(tmp_path / "feedback" / "done")
        cfg.paths.feedback_failed_dir = str(tmp_path / "feedback" / "failed")
        cfg.paths.backup_dir = str(tmp_path / "state" / "backups")
        cfg.validation.test_command = ""
        cfg.validation.lint_command = ""
        cfg.validation.build_command = ""
        cfg.orchestrator.batch_mode = False
        cfg.orchestrator.max_validation_retries = 5

        with patch("orchestrator.GitManager") as MockGit, \
             patch("orchestrator.ClaudeRunner") as MockClaude, \
             patch("orchestrator.TaskDiscovery") as MockDisc, \
             patch("orchestrator.Validator") as MockVal, \
             patch("orchestrator.resolve_model_id", return_value=None), \
             patch("subprocess.run") as mock_sp:

            mock_sp.return_value = MagicMock(returncode=0)
            mock_git = MockGit.return_value
            mock_git.create_snapshot.return_value = Snapshot(commit_hash="a" * 40)
            mock_git.capture_worktree_state.return_value = set()
            mock_git.get_new_changed_files.return_value = ["fix.py"]
            mock_git.is_clean.return_value = True
            mock_git.commit.return_value = "b" * 40

            mock_claude = MockClaude.return_value
            mock_claude.run.return_value = ClaudeResult(
                success=True, result_text="Fixed it",
                cost_usd=0.05, duration_seconds=10.0,
            )

            mock_disc = MockDisc.return_value
            mock_disc.discover_all.return_value = [
                Task(description="Fix bug in foo.py", priority=2, source="test_failure"),
            ]

            mock_val = MockVal.return_value
            mock_val.validate.return_value = ValidationResult(passed=True, steps=[])

            o = Orchestrator(cfg)
            o.git = mock_git
            o.claude = mock_claude
            o.discovery = mock_disc
            o.validator = mock_val
            yield o

    def test_retry_succeeds_on_second_attempt(self, retry_orch):
        """First validation fails, retry fixes it, commit happens."""
        fail_result = ValidationResult(
            passed=False,
            steps=[ValidationStep(
                name="tests", command="pytest", passed=False,
                output="FAILED test_foo.py::test_bar", return_code=1,
            )],
        )
        pass_result = ValidationResult(passed=True, steps=[])
        retry_orch.validator.validate.side_effect = [fail_result, pass_result]

        retry_orch.state.record_cycle = MagicMock()
        retry_orch._cycle()

        retry_orch.git.commit.assert_called_once()
        retry_orch.git.rollback.assert_not_called()
        record = retry_orch.state.record_cycle.call_args[0][0]
        assert record.success is True
        assert record.validation_retry_count == 1

    def test_retry_all_attempts_exhausted(self, retry_orch):
        """All 6 attempts fail (1 initial + 5 retries), rollback at end."""
        fail_result = ValidationResult(
            passed=False,
            steps=[ValidationStep(
                name="tests", command="pytest", passed=False,
                output="FAILED", return_code=1,
            )],
        )
        # 6 validation calls: 1 initial + 5 retries
        retry_orch.validator.validate.side_effect = [fail_result] * 6

        retry_orch.state.record_cycle = MagicMock()
        retry_orch._cycle()

        retry_orch.git.rollback.assert_called_once()
        retry_orch.git.commit.assert_not_called()
        record = retry_orch.state.record_cycle.call_args[0][0]
        assert record.success is False
        assert record.validation_retry_count == 5

    def test_retry_no_rollback_between_attempts(self, retry_orch):
        """git.rollback is not called until final failure (in-place fix behavior)."""
        fail_result = ValidationResult(
            passed=False,
            steps=[ValidationStep(
                name="tests", command="pytest", passed=False,
                output="FAILED", return_code=1,
            )],
        )
        pass_result = ValidationResult(passed=True, steps=[])
        # Fail 3 times, then pass
        retry_orch.validator.validate.side_effect = [
            fail_result, fail_result, fail_result, pass_result,
        ]

        retry_orch._cycle()

        # Rollback should never be called since we eventually succeeded
        retry_orch.git.rollback.assert_not_called()
        retry_orch.git.commit.assert_called_once()

    def test_retry_disabled_when_zero(self, retry_orch):
        """Setting max_validation_retries=0 causes immediate rollback."""
        retry_orch.config.orchestrator.max_validation_retries = 0
        fail_result = ValidationResult(
            passed=False,
            steps=[ValidationStep(
                name="tests", command="pytest", passed=False,
                output="FAILED", return_code=1,
            )],
        )
        retry_orch.validator.validate.return_value = fail_result

        retry_orch.state.record_cycle = MagicMock()
        retry_orch._cycle()

        retry_orch.git.rollback.assert_called_once()
        retry_orch.git.commit.assert_not_called()
        # Claude should only be called once (the initial invocation, no retries)
        assert retry_orch.claude.run.call_count == 1

    def test_retry_cost_guard(self, retry_orch):
        """Retry aborts early if cost limit is approached."""
        fail_result = ValidationResult(
            passed=False,
            steps=[ValidationStep(
                name="tests", command="pytest", passed=False,
                output="FAILED", return_code=1,
            )],
        )
        retry_orch.validator.validate.return_value = fail_result
        # Make cost guard trigger by returning high accumulated cost
        retry_orch.state.get_total_cost = MagicMock(return_value=9.5)
        retry_orch.config.safety.max_cost_usd_per_hour = 10.0

        retry_orch.state.record_cycle = MagicMock()
        retry_orch._cycle()

        retry_orch.git.rollback.assert_called_once()
        record = retry_orch.state.record_cycle.call_args[0][0]
        assert record.success is False
        assert "cost guard" in record.error

    def test_retry_claude_failure_aborts(self, retry_orch):
        """If retry Claude invocation itself fails, rollback immediately."""
        fail_result = ValidationResult(
            passed=False,
            steps=[ValidationStep(
                name="tests", command="pytest", passed=False,
                output="FAILED", return_code=1,
            )],
        )
        retry_orch.validator.validate.return_value = fail_result

        # First Claude call succeeds (initial), second fails (retry)
        retry_orch.claude.run.side_effect = [
            ClaudeResult(success=True, result_text="Fixed it",
                         cost_usd=0.05, duration_seconds=10.0),
            ClaudeResult(success=False, error="API error",
                         cost_usd=0.01, duration_seconds=2.0),
        ]

        retry_orch.state.record_cycle = MagicMock()
        retry_orch._cycle()

        retry_orch.git.rollback.assert_called_once()
        retry_orch.git.commit.assert_not_called()
        record = retry_orch.state.record_cycle.call_args[0][0]
        assert record.success is False
        assert "Retry failed" in record.error

    def test_retry_prompt_includes_test_output(self, retry_orch):
        """The retry prompt contains the actual validation failure output."""
        fail_result = ValidationResult(
            passed=False,
            steps=[ValidationStep(
                name="tests", command="pytest -x", passed=False,
                output="FAILED test_foo.py::test_bar - AssertionError: 1 != 2",
                return_code=1,
            )],
        )
        pass_result = ValidationResult(passed=True, steps=[])
        retry_orch.validator.validate.side_effect = [fail_result, pass_result]

        retry_orch._cycle()

        # The second Claude call should be the retry with failure info
        assert retry_orch.claude.run.call_count == 2
        retry_prompt = retry_orch.claude.run.call_args_list[1][0][0]
        assert "FAILED test_foo.py::test_bar" in retry_prompt
        assert "AssertionError: 1 != 2" in retry_prompt
        assert "VALIDATION FAILURES" in retry_prompt
        assert "attempt 1 of 6" in retry_prompt

    def test_retry_aggregates_cost(self, retry_orch):
        """Single CycleRecord sums cost from all attempts."""
        fail_result = ValidationResult(
            passed=False,
            steps=[ValidationStep(
                name="tests", command="pytest", passed=False,
                output="FAILED", return_code=1,
            )],
        )
        pass_result = ValidationResult(passed=True, steps=[])
        retry_orch.validator.validate.side_effect = [fail_result, fail_result, pass_result]

        # Initial call + 2 retry calls
        retry_orch.claude.run.side_effect = [
            ClaudeResult(success=True, result_text="v1", cost_usd=0.05, duration_seconds=10.0),
            ClaudeResult(success=True, result_text="v2", cost_usd=0.03, duration_seconds=5.0),
            ClaudeResult(success=True, result_text="v3", cost_usd=0.04, duration_seconds=8.0),
        ]

        retry_orch.state.record_cycle = MagicMock()
        retry_orch._cycle()

        record = retry_orch.state.record_cycle.call_args[0][0]
        assert record.success is True
        # 0.05 (initial) + 0.03 (retry1) + 0.04 (retry2)
        assert abs(record.cost_usd - 0.12) < 0.001
        assert abs(record.duration_seconds - 23.0) < 0.1
        assert record.validation_retry_count == 2


class TestCommitMessageStyling:
    """Tests for natural commit message generation."""

    @pytest.fixture
    def msg_orch(self, tmp_path):
        cfg = Config()
        cfg.target_dir = str(tmp_path)
        cfg.paths.history_file = str(tmp_path / "state" / "history.json")
        cfg.paths.lock_file = str(tmp_path / "state" / "lock.pid")
        cfg.paths.feedback_dir = str(tmp_path / "feedback")
        cfg.paths.feedback_done_dir = str(tmp_path / "feedback" / "done")
        cfg.paths.feedback_failed_dir = str(tmp_path / "feedback" / "failed")
        cfg.paths.backup_dir = str(tmp_path / "state" / "backups")
        cfg.validation.test_command = ""
        cfg.validation.lint_command = ""
        cfg.validation.build_command = ""
        cfg.orchestrator.batch_mode = False

        with patch("orchestrator.GitManager"), \
             patch("orchestrator.ClaudeRunner"), \
             patch("orchestrator.TaskDiscovery"), \
             patch("orchestrator.Validator"), \
             patch("orchestrator.resolve_model_id", return_value=None), \
             patch("subprocess.run") as mock_sp:
            mock_sp.return_value = MagicMock(returncode=0)
            o = Orchestrator(cfg)
            yield o

    def test_single_task_commit_message_no_auto_prefix(self, msg_orch):
        task = Task(description="Fix test failure: FAILED tests/test_foo.py::test_bar",
                    priority=2, source="test_failure")
        msg = msg_orch._build_commit_message(task)
        assert "[auto]" not in msg

    def test_single_task_commit_message_test_failure(self, msg_orch):
        task = Task(
            description="Fix test failure: FAILED tests/test_foo.py::test_bar - AssertionError",
            priority=2, source="test_failure",
        )
        msg = msg_orch._build_commit_message(task)
        subject = msg.split("\n")[0]
        assert "[auto]" not in subject
        assert "Fix" in subject
        assert "test_foo.py" in subject

    def test_single_task_commit_message_todo(self, msg_orch):
        task = Task(
            description="Address TODO in config_schema.py:42: add type validation",
            priority=3, source="todo",
        )
        msg = msg_orch._build_commit_message(task)
        subject = msg.split("\n")[0]
        assert "Add type validation" in subject
        assert "config_schema.py" in subject
        assert "[auto]" not in subject

    def test_single_task_commit_message_todo_fixme(self, msg_orch):
        task = Task(
            description="Address TODO in bar.py:5: FIXME: broken edge case",
            priority=3, source="todo",
        )
        msg = msg_orch._build_commit_message(task)
        subject = msg.split("\n")[0]
        assert "Broken edge case" in subject
        assert "bar.py" in subject

    def test_single_task_commit_message_feedback(self, msg_orch):
        task = Task(
            description="Fix the login bug described in the issue",
            priority=1, source="feedback",
        )
        msg = msg_orch._build_commit_message(task)
        subject = msg.split("\n")[0]
        assert subject == "Fix the login bug described in the issue"

    def test_single_task_commit_message_truncation(self, msg_orch):
        long_desc = "Implement a comprehensive refactoring of the authentication module to support OAuth 2.0 and SAML integration with proper error handling"
        task = Task(description=long_desc, priority=4, source="claude_idea")
        msg = msg_orch._build_commit_message(task)
        subject = msg.split("\n")[0]
        assert len(subject) <= 72
        # Should have a body with the full description
        assert "\n\n" in msg

    def test_single_task_commit_message_claude_idea(self, msg_orch):
        task = Task(
            description="In `safety.py:98-105`, `check_protected_files` uses os.path.normpath comparison",
            priority=5, source="claude_idea",
        )
        msg = msg_orch._build_commit_message(task)
        subject = msg.split("\n")[0]
        assert "[auto]" not in subject
        # Backticks should be stripped
        assert "`" not in subject
        # Line numbers should be stripped
        assert ":98-105" not in subject
        assert "safety.py" in subject

    def test_batch_commit_message_same_source(self, msg_orch):
        tasks = [
            Task(description="Fix test failure in foo.py", priority=2, source="test_failure"),
            Task(description="Fix test failure in bar.py", priority=2, source="test_failure"),
        ]
        msg = msg_orch._build_batch_commit_message(tasks)
        subject = msg.split("\n")[0]
        assert "Fix test failures" in subject
        assert "foo.py" in subject
        assert "bar.py" in subject

    def test_batch_commit_message_mixed_sources(self, msg_orch):
        tasks = [
            Task(description="Fix test failure in foo.py", priority=2, source="test_failure"),
            Task(description="Address TODO in bar.py:10: add validation", priority=3, source="todo"),
            Task(description="Fix lint error in baz.py", priority=3, source="lint"),
        ]
        msg = msg_orch._build_batch_commit_message(tasks)
        subject = msg.split("\n")[0]
        assert "[auto]" not in subject
        assert "batch(" not in subject

    def test_batch_commit_message_no_batch_prefix(self, msg_orch):
        tasks = [
            Task(description="Task A", priority=2, source="claude_idea"),
            Task(description="Task B", priority=3, source="claude_idea"),
            Task(description="Task C", priority=4, source="claude_idea"),
        ]
        msg = msg_orch._build_batch_commit_message(tasks)
        assert "batch(" not in msg
        assert "[auto]" not in msg

    def test_clean_description_strips_backticks(self, msg_orch):
        result = Orchestrator._clean_description("`file.py` has a bug in `func()`")
        assert "`" not in result
        assert "File.py has a bug in func()" == result

    def test_clean_description_strips_line_numbers(self, msg_orch):
        result = Orchestrator._clean_description("In safety.py:98-105, check_protected_files")
        assert ":98-105" not in result
        assert "safety.py" in result

    def test_clean_description_strips_single_line_number(self, msg_orch):
        result = Orchestrator._clean_description("Fix bug in foo.py:42")
        assert ":42" not in result
        assert "foo.py" in result

    def test_no_pipeline_metadata_in_commit(self, msg_orch):
        """Verify that pipeline metadata is not added to commit messages."""
        task = Task(description="Fix something", priority=2, source="test_failure")
        msg = msg_orch._build_commit_message(task)
        assert "[pipeline:" not in msg
        assert "revisions=" not in msg
        assert "approved=" not in msg

    def test_no_pipeline_metadata_in_commit_via_validate(self, tmp_path):
        """Full cycle: pipeline metadata should not appear in the commit message."""
        cfg = Config()
        cfg.target_dir = str(tmp_path)
        cfg.paths.history_file = str(tmp_path / "state" / "history.json")
        cfg.paths.lock_file = str(tmp_path / "state" / "lock.pid")
        cfg.paths.feedback_dir = str(tmp_path / "feedback")
        cfg.paths.feedback_done_dir = str(tmp_path / "feedback" / "done")
        cfg.paths.feedback_failed_dir = str(tmp_path / "feedback" / "failed")
        cfg.paths.backup_dir = str(tmp_path / "state" / "backups")
        cfg.validation.test_command = ""
        cfg.validation.lint_command = ""
        cfg.validation.build_command = ""
        cfg.orchestrator.batch_mode = False
        cfg.orchestrator.max_validation_retries = 0

        with patch("orchestrator.GitManager") as MockGit, \
             patch("orchestrator.ClaudeRunner"), \
             patch("orchestrator.TaskDiscovery"), \
             patch("orchestrator.Validator") as MockVal, \
             patch("orchestrator.resolve_model_id", return_value=None), \
             patch("subprocess.run") as mock_sp:

            mock_sp.return_value = MagicMock(returncode=0)
            mock_git = MockGit.return_value
            mock_git.create_snapshot.return_value = Snapshot(commit_hash="a" * 40)
            mock_git.capture_worktree_state.return_value = set()
            mock_git.get_new_changed_files.return_value = ["fix.py"]
            mock_git.commit.return_value = "b" * 40

            mock_val = MockVal.return_value
            mock_val.validate.return_value = ValidationResult(passed=True, steps=[])

            o = Orchestrator(cfg)
            o.git = mock_git
            o.validator = mock_val

            tasks = [Task(description="Fix thing", priority=2, source="test_failure")]
            o._validate_with_retries(
                tasks=tasks,
                snapshot=Snapshot(commit_hash="a" * 40),
                pre_existing_files=set(),
                total_cost=0.05, total_duration=10.0,
                is_batch=False,
                extra_record_kwargs={
                    "pipeline_mode": "multi_agent",
                    "pipeline_revision_count": 2,
                    "pipeline_review_approved": True,
                },
            )

            commit_msg = o.git.commit.call_args[0][0]
            assert "[pipeline:" not in commit_msg
            assert "revisions=" not in commit_msg
            assert "approved=" not in commit_msg

    def test_single_task_lint_message(self, msg_orch):
        task = Task(
            description="Fix lint error in foo.py: [F401] unused import",
            priority=3, source="lint",
        )
        msg = msg_orch._build_commit_message(task)
        subject = msg.split("\n")[0]
        assert "[auto]" not in subject
        assert "Fix" in subject

    def test_single_task_coverage_message(self, msg_orch):
        task = Task(
            description="Low coverage in utils.py (45%)",
            priority=4, source="coverage",
        )
        msg = msg_orch._build_commit_message(task)
        subject = msg.split("\n")[0]
        assert "Add test coverage for" in subject

    def test_single_task_quality_message(self, msg_orch):
        task = Task(
            description="Complex function in parser.py needs simplification",
            priority=5, source="quality",
        )
        msg = msg_orch._build_commit_message(task)
        subject = msg.split("\n")[0]
        assert subject.startswith("Refactor")
