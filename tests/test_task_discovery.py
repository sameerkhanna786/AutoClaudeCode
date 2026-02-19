"""Tests for task_discovery module."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config_schema import Config
from process_utils import RunResult
from task_discovery import Task, TaskDiscovery, MAX_TASK_DESCRIPTION_LENGTH


@pytest.fixture
def discovery(tmp_path, default_config):
    default_config.target_dir = str(tmp_path)
    return TaskDiscovery(default_config)


def _run_result(returncode=0, stdout="", stderr="", timed_out=False):
    return RunResult(returncode=returncode, stdout=stdout, stderr=stderr, timed_out=timed_out)


class TestTaskDiscovery:
    def test_discover_all_returns_sorted(self, discovery, tmp_path):
        # Set up a source file containing a task marker comment
        (tmp_path / "app.py").write_text("# TODO: fix this\n")
        with patch.object(discovery, "_discover_test_failures", return_value=[]):
            with patch.object(discovery, "_discover_lint_errors", return_value=[]):
                tasks = discovery.discover_all()
                # Verify the planted marker comment in app.py is discovered
                assert any(t.source == "todo" for t in tasks)

    @patch("task_discovery.run_with_group_kill")
    def test_discover_test_failures_passing(self, mock_run, discovery):
        mock_run.return_value = _run_result(returncode=0, stdout="all passed")
        tasks = discovery._discover_test_failures()
        assert tasks == []

    @patch("task_discovery.run_with_group_kill")
    def test_discover_test_failures_with_failures(self, mock_run, discovery):
        mock_run.return_value = _run_result(
            returncode=1,
            stdout="FAILED tests/test_foo.py::test_bar - AssertionError\n",
        )
        tasks = discovery._discover_test_failures()
        assert len(tasks) == 1
        assert tasks[0].priority == 2
        assert tasks[0].source == "test_failure"
        assert "test_bar" in tasks[0].description

    @patch("task_discovery.run_with_group_kill")
    def test_discover_test_failures_generic(self, mock_run, discovery):
        mock_run.return_value = _run_result(
            returncode=1,
            stdout="some error output\n",
        )
        tasks = discovery._discover_test_failures()
        assert len(tasks) == 1
        assert "exit code 1" in tasks[0].description

    @patch("task_discovery.run_with_group_kill")
    def test_discover_test_failures_timeout(self, mock_run, discovery):
        mock_run.return_value = _run_result(timed_out=True, returncode=-1)
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

    @patch("task_discovery.run_with_group_kill")
    def test_discover_lint_errors_json(self, mock_run, discovery):
        discovery.config.validation.lint_command = "ruff check --output-format=json ."
        mock_run.return_value = _run_result(
            returncode=1,
            stdout='[{"filename": "foo.py", "message": "unused import", "code": "F401"}]',
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

    @patch("task_discovery.run_with_group_kill")
    def test_discover_claude_ideas_success(self, mock_run, discovery):
        discovery.config.discovery.enable_claude_ideas = True
        mock_run.return_value = _run_result(
            returncode=0,
            stdout='{"result": "IDEA: Add input validation to the config loader\\nIDEA: Add retry logic to Claude runner\\nSome other text"}',
        )
        tasks = discovery._discover_claude_ideas()
        assert len(tasks) == 2
        assert all(t.priority == 4 for t in tasks)
        assert all(t.source == "claude_idea" for t in tasks)
        assert "input validation" in tasks[0].description.lower()
        assert "retry" in tasks[1].description.lower()

    @patch("task_discovery.run_with_group_kill")
    def test_discover_claude_ideas_timeout(self, mock_run, discovery):
        discovery.config.discovery.enable_claude_ideas = True
        mock_run.return_value = _run_result(timed_out=True, returncode=-1)
        tasks = discovery._discover_claude_ideas()
        assert tasks == []

    @patch("task_discovery.run_with_group_kill")
    def test_discover_claude_ideas_failure(self, mock_run, discovery):
        discovery.config.discovery.enable_claude_ideas = True
        mock_run.return_value = _run_result(returncode=1, stderr="error")
        tasks = discovery._discover_claude_ideas()
        assert tasks == []

    @patch("task_discovery.run_with_group_kill")
    def test_discover_claude_ideas_no_ideas(self, mock_run, discovery):
        discovery.config.discovery.enable_claude_ideas = True
        mock_run.return_value = _run_result(
            returncode=0,
            stdout='{"result": "The codebase looks great, no improvements needed."}',
        )
        tasks = discovery._discover_claude_ideas()
        assert tasks == []

    @patch("task_discovery.run_with_group_kill")
    def test_discover_claude_ideas_caps_at_five(self, mock_run, discovery):
        discovery.config.discovery.enable_claude_ideas = True
        ideas = "\n".join(f"IDEA: Improvement {i}" for i in range(10))
        mock_run.return_value = _run_result(
            returncode=0,
            stdout=f'{{"result": "{ideas}"}}',
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

    @patch("task_discovery.run_with_group_kill")
    def test_discover_claude_ideas_uses_discovery_model(self, mock_run, discovery):
        """Claude idea discovery should use discovery_model, not claude.model."""
        discovery.config.discovery.enable_claude_ideas = True
        discovery.config.discovery.discovery_model = "haiku"
        discovery.config.claude.model = "opus"
        mock_run.return_value = _run_result(
            returncode=0,
            stdout='{"result": "IDEA: Test improvement"}',
        )
        discovery._discover_claude_ideas()
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        model_idx = cmd.index("--model")
        assert cmd[model_idx + 1] == "haiku"

    @patch("task_discovery.run_with_group_kill")
    def test_discover_claude_ideas_uses_discovery_timeout(self, mock_run, discovery):
        """Claude idea discovery should use discovery_timeout, not claude.timeout_seconds."""
        discovery.config.discovery.enable_claude_ideas = True
        discovery.config.discovery.discovery_timeout = 240
        discovery.config.claude.timeout_seconds = 300
        mock_run.return_value = _run_result(
            returncode=0,
            stdout='{"result": "IDEA: Test"}',
        )
        discovery._discover_claude_ideas()
        call_args = mock_run.call_args
        assert call_args[1]["timeout"] == 240

    @patch("task_discovery.run_with_group_kill")
    def test_discover_claude_ideas_ignores_unrelated_json(self, mock_run, discovery):
        """If a line contains valid JSON without a 'result' field, parsing
        should continue and find the correct JSON later."""
        discovery.config.discovery.enable_claude_ideas = True
        # First line is unrelated JSON, second line has the actual result
        stdout = '{"status": "ok"}\n{"result": "IDEA: Add input validation"}'
        mock_run.return_value = _run_result(returncode=0, stdout=stdout)
        tasks = discovery._discover_claude_ideas()
        assert len(tasks) == 1
        assert "input validation" in tasks[0].description.lower()

    @patch("task_discovery.run_with_group_kill")
    def test_discover_claude_ideas_non_string_result(self, mock_run, discovery):
        """If the JSON 'result' field is not a string, it should be ignored."""
        discovery.config.discovery.enable_claude_ideas = True
        mock_run.return_value = _run_result(returncode=0, stdout='{"result": 42}')
        tasks = discovery._discover_claude_ideas()
        assert tasks == []

    @patch("task_discovery.run_with_group_kill")
    def test_discover_claude_ideas_null_result(self, mock_run, discovery):
        """If the JSON 'result' field is null, it should be ignored."""
        discovery.config.discovery.enable_claude_ideas = True
        mock_run.return_value = _run_result(returncode=0, stdout='{"result": null}')
        tasks = discovery._discover_claude_ideas()
        assert tasks == []

    @patch("task_discovery.run_with_group_kill")
    def test_discover_claude_ideas_unexpected_json_structure(self, mock_run, discovery):
        """If JSON is a list instead of an object, it should be handled gracefully."""
        discovery.config.discovery.enable_claude_ideas = True
        mock_run.return_value = _run_result(
            returncode=0,
            stdout='[{"result": "IDEA: something"}]',
        )
        tasks = discovery._discover_claude_ideas()
        assert tasks == []


class TestTaskDescriptionValidation:
    def test_task_description_truncation(self):
        """Descriptions longer than MAX_TASK_DESCRIPTION_LENGTH are truncated."""
        long_desc = "x" * (MAX_TASK_DESCRIPTION_LENGTH + 500)
        task = Task(description=long_desc, priority=1, source="test")
        assert len(task.description) == MAX_TASK_DESCRIPTION_LENGTH + 3  # +3 for "..."
        assert task.description.endswith("...")

    def test_task_description_newline_replacement(self):
        """Newlines in descriptions are replaced with spaces."""
        task = Task(description="line one\nline two\nline three", priority=1, source="test")
        assert "\n" not in task.description
        assert task.description == "line one line two line three"

    def test_task_description_whitespace_stripped(self):
        """Leading/trailing whitespace is stripped."""
        task = Task(description="  hello world  ", priority=1, source="test")
        assert task.description == "hello world"

    def test_task_description_short_preserved(self):
        """Short descriptions are preserved as-is (after strip)."""
        task = Task(description="Fix the bug", priority=2, source="test_failure")
        assert task.description == "Fix the bug"


class TestClaudeIdeasMinLength:
    @patch("task_discovery.run_with_group_kill")
    def test_discover_claude_ideas_skips_short_descriptions(self, mock_run, discovery):
        """IDEA lines with descriptions shorter than 10 chars are skipped."""
        discovery.config.discovery.enable_claude_ideas = True
        ideas = "IDEA: fix\nIDEA: Add comprehensive input validation to config loader"
        mock_run.return_value = _run_result(
            returncode=0,
            stdout=f'{{"result": "{ideas}"}}',
        )
        tasks = discovery._discover_claude_ideas()
        assert len(tasks) == 1
        assert "comprehensive" in tasks[0].description.lower()

    @patch("task_discovery.run_with_group_kill")
    def test_discover_claude_ideas_skips_empty_idea_lines(self, mock_run, discovery):
        """IDEA lines with no description or only whitespace are skipped."""
        discovery.config.discovery.enable_claude_ideas = True
        ideas = "IDEA: \\nIDEA:\\nIDEA:   "
        mock_run.return_value = _run_result(
            returncode=0,
            stdout=f'{{"result": "{ideas}"}}',
        )
        tasks = discovery._discover_claude_ideas()
        assert tasks == []

    @patch("task_discovery.run_with_group_kill")
    def test_discover_claude_ideas_accepts_long_enough_descriptions(self, mock_run, discovery):
        """IDEA lines with descriptions >= 10 chars are accepted."""
        discovery.config.discovery.enable_claude_ideas = True
        ideas = "IDEA: Add retries to API calls"
        mock_run.return_value = _run_result(
            returncode=0,
            stdout=f'{{"result": "{ideas}"}}',
        )
        tasks = discovery._discover_claude_ideas()
        assert len(tasks) == 1


class TestTaskKey:
    def test_todo_task_key_with_file_and_line(self):
        task = Task(description="Address TODO in foo.py:10", priority=3, source="todo",
                    source_file="foo.py", line_number=10)
        assert task.task_key == "todo:foo.py:10"

    def test_todo_task_key_with_file_no_line(self):
        task = Task(description="Address TODO in foo.py", priority=3, source="todo",
                    source_file="foo.py")
        assert task.task_key == "todo:foo.py"

    def test_lint_task_key(self):
        task = Task(description="Fix lint error in foo.py: [F401] unused import",
                    priority=2, source="lint", source_file="foo.py")
        assert task.task_key == "lint:foo.py"

    def test_claude_idea_task_key_with_backtick_ref(self):
        task = Task(description="Improve error handling in `safety.py:98-105` for edge cases",
                    priority=4, source="claude_idea")
        assert task.task_key == "claude_idea:safety.py"

    def test_claude_idea_task_key_with_in_ref(self):
        task = Task(description="Add validation in config_schema.py for negative values",
                    priority=4, source="claude_idea")
        assert task.task_key == "claude_idea:config_schema.py"

    def test_claude_idea_task_key_no_file_ref(self):
        task = Task(description="Improve overall error handling across the codebase",
                    priority=4, source="claude_idea")
        assert task.task_key == "claude_idea:Improve overall error handling across the codebase"

    def test_claude_idea_task_key_truncates_at_60(self):
        long_desc = "A" * 100
        task = Task(description=long_desc, priority=4, source="claude_idea")
        assert task.task_key == f"claude_idea:{long_desc[:60]}"

    def test_test_failure_task_key_with_failed_pattern(self):
        task = Task(description="Fix test failure: FAILED tests/test_foo.py::test_bar - AssertionError",
                    priority=2, source="test_failure")
        assert task.task_key == "test_failure:tests/test_foo.py::test_bar"

    def test_test_failure_task_key_with_source_file(self):
        task = Task(description="Fix test failure", priority=2, source="test_failure",
                    source_file="tests/test_foo.py")
        assert task.task_key == "test_failure:tests/test_foo.py"

    def test_feedback_task_key(self):
        task = Task(description="Fix the login bug", priority=1, source="feedback",
                    source_file="/tmp/feedback/fix.md")
        assert task.task_key == "feedback:/tmp/feedback/fix.md"

    def test_coverage_task_key(self):
        task = Task(description="Improve test coverage for foo.py (currently 30%)",
                    priority=4, source="coverage", source_file="foo.py")
        assert task.task_key == "coverage:foo.py"

    def test_coverage_task_key_no_source_file(self):
        task = Task(description="Improve test coverage for bar.py (currently 20%)",
                    priority=4, source="coverage")
        assert task.task_key == "coverage:bar.py"

    def test_quality_task_key(self):
        task = Task(description="Review big.py (600 lines)", priority=5, source="quality",
                    source_file="big.py")
        assert task.task_key == "quality:big.py"

    def test_fallback_task_key(self):
        task = Task(description="Some generic task", priority=5, source="unknown_source")
        assert task.task_key == "unknown_source:Some generic task"
