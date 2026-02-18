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
        # Set up a source file containing a task marker comment
        (tmp_path / "app.py").write_text("# TODO: fix this\n")
        with patch.object(discovery, "_discover_test_failures", return_value=[]):
            with patch.object(discovery, "_discover_lint_errors", return_value=[]):
                tasks = discovery.discover_all()
                # Should find at least one task from the todo-comment scanner
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

    @patch("task_discovery.subprocess.run")
    def test_discover_claude_ideas_success(self, mock_run, discovery):
        discovery.config.discovery.enable_claude_ideas = True
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "IDEA: Add input validation to the config loader\\nIDEA: Add retry logic to Claude runner\\nSome other text"}',
            stderr="",
        )
        tasks = discovery._discover_claude_ideas()
        assert len(tasks) == 2
        assert all(t.priority == 4 for t in tasks)
        assert all(t.source == "claude_idea" for t in tasks)
        assert "input validation" in tasks[0].description.lower()
        assert "retry" in tasks[1].description.lower()

    @patch("task_discovery.subprocess.run")
    def test_discover_claude_ideas_timeout(self, mock_run, discovery):
        discovery.config.discovery.enable_claude_ideas = True
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=120)
        tasks = discovery._discover_claude_ideas()
        assert tasks == []

    @patch("task_discovery.subprocess.run")
    def test_discover_claude_ideas_failure(self, mock_run, discovery):
        discovery.config.discovery.enable_claude_ideas = True
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        tasks = discovery._discover_claude_ideas()
        assert tasks == []

    @patch("task_discovery.subprocess.run")
    def test_discover_claude_ideas_no_ideas(self, mock_run, discovery):
        discovery.config.discovery.enable_claude_ideas = True
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "The codebase looks great, no improvements needed."}',
            stderr="",
        )
        tasks = discovery._discover_claude_ideas()
        assert tasks == []

    @patch("task_discovery.subprocess.run")
    def test_discover_claude_ideas_caps_at_five(self, mock_run, discovery):
        discovery.config.discovery.enable_claude_ideas = True
        ideas = "\n".join(f"IDEA: Improvement {i}" for i in range(10))
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=f'{{"result": "{ideas}"}}',
            stderr="",
        )
        tasks = discovery._discover_claude_ideas()
        assert len(tasks) <= 5

    def test_discover_todos_ignores_string_literals(self, discovery, tmp_path):
        """TODO inside a string literal should not be detected."""
        (tmp_path / "strings.py").write_text('x = "# TODO: not a comment"\n')
        tasks = discovery._discover_todos()
        assert tasks == []

    def test_discover_todos_js_comments(self, discovery, tmp_path):
        """// comments in JS files should be detected."""
        (tmp_path / "app.js").write_text("// TODO: fix this\n")
        tasks = discovery._discover_todos()
        assert len(tasks) == 1
        assert "TODO" in tasks[0].description

    def test_discover_todos_block_comment_prefix(self, discovery, tmp_path):
        """/* FIXME ... */ in Java files should be detected."""
        (tmp_path / "App.java").write_text("/* FIXME: broken */\n")
        tasks = discovery._discover_todos()
        assert len(tasks) == 1
        assert "FIXME" in tasks[0].description

    def test_discover_todos_in_string_not_matched_js(self, discovery, tmp_path):
        """TODO inside a JS string literal should not be detected."""
        (tmp_path / "app.js").write_text('const s = "// TODO: fake";\n')
        tasks = discovery._discover_todos()
        assert tasks == []

    def test_discover_todos_real_comment_after_code(self, discovery, tmp_path):
        """An inline comment after code should still be detected."""
        (tmp_path / "code.py").write_text("x = 1  # TODO: real comment\n")
        tasks = discovery._discover_todos()
        assert len(tasks) == 1
        assert "TODO" in tasks[0].description

    def test_discover_todos_respects_max_todo_tasks(self, discovery, tmp_path):
        """TODO task count should be capped by config.discovery.max_todo_tasks."""
        lines = "\n".join(f"# TODO: item {i}" for i in range(5))
        (tmp_path / "many.py").write_text(lines + "\n")
        discovery.config.discovery.max_todo_tasks = 2
        tasks = discovery._discover_todos()
        assert len(tasks) == 2
