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
    cfg.paths.backup_dir = str(tmp_path / "state" / "backups")
    # Disable validation commands to avoid subprocess calls
    cfg.validation.test_command = ""
    cfg.validation.lint_command = ""
    cfg.validation.build_command = ""
    return cfg


@pytest.fixture
def orch(config):
    with patch("orchestrator.GitManager") as MockGit, \
         patch("orchestrator.ClaudeRunner") as MockClaude, \
         patch("orchestrator.TaskDiscovery") as MockDisc, \
         patch("orchestrator.Validator") as MockVal:

        mock_git = MockGit.return_value
        mock_git.create_snapshot.return_value = Snapshot(commit_hash="a" * 40)
        mock_git.get_changed_files.return_value = ["fix.py"]
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
        orch.git.get_changed_files.return_value = []
        orch._cycle()
        orch.git.commit.assert_not_called()

    def test_protected_file_violation_causes_rollback(self, orch):
        orch.git.get_changed_files.return_value = ["main.py", "fix.py"]
        orch._cycle()
        orch.git.rollback.assert_called_once()
        orch.git.commit.assert_not_called()

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
