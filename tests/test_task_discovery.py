"""Tests for task_discovery module."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config_schema import Config
from task_discovery import Task, TaskDiscovery


@pytest.fixture
def discovery(tmp_path, default_config):
    default_config.target_dir = str(tmp_path)
    return TaskDiscovery(default_config)


class TestTaskDiscovery:
    def test_discover_all_returns_sorted(self, discovery, tmp_path):
        # Create a file with a TODO
        (tmp_path / "app.py").write_text("# TODO: fix this\n")
        with patch.object(discovery, "_discover_test_failures", return_value=[]):
            with patch.object(discovery, "_discover_lint_errors", return_value=[]):
                tasks = discovery.discover_all()
                # Should find at least the TODO
                assert any(t.source == "todo" for t in tasks)

    @patch("task_discovery.subprocess.run")
    def test_discover_test_failures_passing(self, mock_run, discovery):
        mock_run.return_value = MagicMock(returncode=0, stdout="all passed", stderr="")
        tasks = discovery._discover_test_failures()
        assert tasks == []

    @patch("task_discovery.subprocess.run")
    def test_discover_test_failures_with_failures(self, mock_run, discovery):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="FAILED tests/test_foo.py::test_bar - AssertionError\n",
            stderr="",
        )
        tasks = discovery._discover_test_failures()
        assert len(tasks) == 1
        assert tasks[0].priority == 2
        assert tasks[0].source == "test_failure"
        assert "test_bar" in tasks[0].description

    @patch("task_discovery.subprocess.run")
    def test_discover_test_failures_generic(self, mock_run, discovery):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="some error output\n",
            stderr="",
        )
        tasks = discovery._discover_test_failures()
        assert len(tasks) == 1
        assert "exit code 1" in tasks[0].description

    @patch("task_discovery.subprocess.run")
    def test_discover_test_failures_timeout(self, mock_run, discovery):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="pytest", timeout=120)
        tasks = discovery._discover_test_failures()
        assert tasks == []

    def test_discover_todos(self, discovery, tmp_path):
        (tmp_path / "code.py").write_text("x = 1\n# TODO: refactor this\ny = 2\n# FIXME: broken\n")
        tasks = discovery._discover_todos()
        assert len(tasks) == 2
        assert all(t.priority == 3 for t in tasks)
        assert all(t.source == "todo" for t in tasks)

    def test_discover_todos_excludes_dirs(self, discovery, tmp_path):
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "cached.py").write_text("# TODO: should be ignored\n")
        tasks = discovery._discover_todos()
        assert tasks == []

    def test_discover_todos_empty(self, discovery, tmp_path):
        (tmp_path / "clean.py").write_text("x = 1\n")
        tasks = discovery._discover_todos()
        assert tasks == []

    def test_discover_lint_empty_command(self, discovery):
        tasks = discovery._discover_lint_errors()
        assert tasks == []

    @patch("task_discovery.subprocess.run")
    def test_discover_lint_errors_json(self, mock_run, discovery):
        discovery.config.validation.lint_command = "ruff check --output-format=json ."
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout='[{"filename": "foo.py", "message": "unused import", "code": "F401"}]',
            stderr="",
        )
        tasks = discovery._discover_lint_errors()
        assert len(tasks) == 1
        assert "foo.py" in tasks[0].description
        assert tasks[0].source == "lint"

    def test_discover_quality_long_file(self, discovery, tmp_path):
        (tmp_path / "big.py").write_text("\n" * 600)
        tasks = discovery._discover_quality_issues()
        assert len(tasks) == 1
        assert tasks[0].source == "quality"
        assert "big.py" in tasks[0].description
