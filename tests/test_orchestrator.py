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
         patch("orchestrator.Validator") as MockVal:

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
    return cfg


@pytest.fixture
def orch_batch(batch_config):
    with patch("orchestrator.GitManager") as MockGit, \
         patch("orchestrator.ClaudeRunner") as MockClaude, \
         patch("orchestrator.TaskDiscovery") as MockDisc, \
         patch("orchestrator.Validator") as MockVal:

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
        orch_batch.config.orchestrator.max_tasks_per_cycle = 2
        tasks = orch_batch._gather_tasks()
        assert len(tasks) == 2

    def test_gather_tasks_excludes_recently_attempted(self, orch_batch):
        orch_batch.state.was_recently_attempted = MagicMock(
            side_effect=lambda desc: desc == "Fix bug in foo.py"
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
        assert "batch(" in commit_msg
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
        assert msg.startswith("[auto] batch(2):")
        assert "test_failure" in msg
        assert "todo" in msg
        assert "Fix bug in foo.py" in msg
        assert "Address TODO" in msg

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
