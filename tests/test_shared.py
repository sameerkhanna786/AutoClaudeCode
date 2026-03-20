"""Tests for shared.py — shared utility functions."""

from __future__ import annotations

import os
import textwrap
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from task_discovery import Task
from shared import (
    format_task_list,
    format_validation_errors,
    syntax_check_files,
    gather_tasks,
    clean_description,
    build_commit_message,
    build_batch_commit_message,
    extract_file_names,
    build_task_prompt,
    build_plan_prompt,
    build_execute_prompt,
    build_retry_prompt,
    _derive_todo_subject,
    TASK_TYPE_INSTRUCTIONS,
)


# ---------------------------------------------------------------------------
# format_task_list
# ---------------------------------------------------------------------------

class TestFormatTaskList(unittest.TestCase):

    def test_single_task(self):
        task = Task(description="Fix bug in foo.py", priority=1, source="test_failure")
        result = format_task_list([task])
        self.assertIn("1. Fix bug in foo.py [test_failure]", result)

    def test_multiple_tasks(self):
        tasks = [
            Task(description="Fix A", priority=1, source="lint"),
            Task(description="Fix B", priority=2, source="todo"),
        ]
        result = format_task_list(tasks)
        self.assertIn("1. Fix A [lint]", result)
        self.assertIn("2. Fix B [todo]", result)

    def test_task_with_context(self):
        task = Task(
            description="Fix test", priority=1, source="test_failure",
            context="line 10: assert failed\nline 11: expected True",
        )
        result = format_task_list([task])
        self.assertIn("CONTEXT:", result)
        self.assertIn("line 10: assert failed", result)
        self.assertIn("line 11: expected True", result)


# ---------------------------------------------------------------------------
# syntax_check_files
# ---------------------------------------------------------------------------

class TestSyntaxCheckFiles(unittest.TestCase):

    def test_valid_python_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = Path(tmpdir) / "valid.py"
            py_file.write_text("x = 1\n")
            result = syntax_check_files(["valid.py"], tmpdir)
            self.assertIsNone(result)

    def test_syntax_error_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = Path(tmpdir) / "bad.py"
            py_file.write_text("def foo(\n")
            result = syntax_check_files(["bad.py"], tmpdir)
            self.assertIsNotNone(result)
            self.assertIn("Syntax error", result)
            self.assertIn("bad.py", result)

    def test_non_py_file_skipped(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_file = Path(tmpdir) / "readme.txt"
            txt_file.write_text("not python {{{{")
            result = syntax_check_files(["readme.txt"], tmpdir)
            self.assertIsNone(result)

    def test_utf8_file_with_non_ascii(self):
        """syntax_check_files should handle UTF-8 files with non-ASCII characters."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = Path(tmpdir) / "unicode.py"
            py_file.write_text(
                '# -*- coding: utf-8 -*-\nx = "héllo wörld"\n',
                encoding="utf-8",
            )
            result = syntax_check_files(["unicode.py"], tmpdir)
            self.assertIsNone(result)

    def test_non_utf8_file_skipped(self):
        """syntax_check_files should skip files with non-UTF-8 encoding instead of crashing."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = Path(tmpdir) / "latin1.py"
            # Write bytes that are valid Latin-1 but invalid UTF-8
            py_file.write_bytes(b"x = '\xe9'\n")
            result = syntax_check_files(["latin1.py"], tmpdir)
            # Should return None (skip the file), not crash with UnicodeDecodeError
            self.assertIsNone(result)

    def test_non_utf8_file_does_not_mask_syntax_errors(self):
        """A non-UTF-8 file should be skipped but syntax errors in other files still caught."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            latin_file = Path(tmpdir) / "latin1.py"
            latin_file.write_bytes(b"x = '\xe9'\n")
            bad_file = Path(tmpdir) / "bad.py"
            bad_file.write_text("def foo(\n", encoding="utf-8")
            result = syntax_check_files(["latin1.py", "bad.py"], tmpdir)
            self.assertIsNotNone(result)
            self.assertIn("bad.py", result)


# ---------------------------------------------------------------------------
# gather_tasks
# ---------------------------------------------------------------------------

class TestGatherTasks(unittest.TestCase):

    def _make_mocks(self, feedback_tasks=None, discovered_tasks=None,
                    recently_attempted=None, failure_counts=None):
        config = MagicMock()
        config.orchestrator.max_feedback_retries = 3
        config.discovery.adaptive_priority = False

        feedback = MagicMock()
        feedback.get_pending_feedback.return_value = feedback_tasks or []

        state = MagicMock()
        recently_attempted = recently_attempted or set()
        state.was_recently_attempted.side_effect = (
            lambda desc, task_key=None: desc in recently_attempted
        )
        failure_counts = failure_counts or {}
        state.get_task_failure_count.side_effect = (
            lambda desc, source, task_key=None: failure_counts.get(desc, 0)
        )

        discovery = MagicMock()
        discovery.discover_all.return_value = discovered_tasks or []

        return config, feedback, state, discovery

    def test_feedback_tasks_returned(self):
        fb_task = Task(
            description="Fix the login page", priority=1, source="feedback",
            source_file="login.md",
        )
        config, feedback, state, discovery = self._make_mocks(feedback_tasks=[fb_task])
        result = gather_tasks(config, feedback, state, discovery)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].description, "Fix the login page")

    def test_discovered_tasks_returned(self):
        disc_task = Task(description="Fix lint in foo.py", priority=2, source="lint")
        config, feedback, state, discovery = self._make_mocks(discovered_tasks=[disc_task])
        result = gather_tasks(config, feedback, state, discovery)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].source, "lint")

    def test_dedup_recently_attempted(self):
        task = Task(description="Already done", priority=2, source="lint")
        config, feedback, state, discovery = self._make_mocks(
            discovered_tasks=[task],
            recently_attempted={"Already done"},
        )
        result = gather_tasks(config, feedback, state, discovery)
        self.assertEqual(len(result), 0)

    def test_adaptive_priority(self):
        task = Task(description="Fix lint", priority=10, source="lint")
        config, feedback, state, discovery = self._make_mocks(discovered_tasks=[task])
        config.discovery.adaptive_priority = True
        state.get_success_rate_by_type.return_value = {"lint": 0.5}
        result = gather_tasks(config, feedback, state, discovery)
        self.assertEqual(len(result), 1)
        # priority should be reduced: 10 * max(0.1, 1.0 - 0.5) = 10 * 0.5 = 5
        self.assertEqual(result[0].priority, 5)

    def test_feedback_exceeds_max_retries(self):
        fb_task = Task(
            description="Broken task", priority=1, source="feedback",
            source_file="broken.md",
        )
        config, feedback, state, discovery = self._make_mocks(
            feedback_tasks=[fb_task],
            failure_counts={"Broken task": 5},
        )
        result = gather_tasks(config, feedback, state, discovery)
        self.assertEqual(len(result), 0)
        feedback.mark_failed.assert_called_once_with("broken.md")

    def test_dashboard_active_enqueues_discovered_tasks(self):
        """When dashboard is active, auto-discovered tasks go to approval queue."""
        disc_task = Task(description="Fix lint in foo.py", priority=2, source="lint")
        config, feedback, state, discovery = self._make_mocks(discovered_tasks=[disc_task])
        config.orchestrator.task_approval = True

        mock_queue = MagicMock()
        mock_queue.get_approved.return_value = []
        mock_queue.enqueue.return_value = "lint_foo.py"

        result = gather_tasks(config, feedback, state, discovery,
                              dashboard_active=True, task_approval_queue=mock_queue)
        # Discovered task should NOT be in result (enqueued instead)
        self.assertEqual(len(result), 0)
        mock_queue.enqueue.assert_called_once()

    def test_dashboard_active_feedback_bypasses_approval(self):
        """Feedback tasks should bypass approval gate even when dashboard is active."""
        fb_task = Task(description="Fix login", priority=1, source="feedback",
                       source_file="login.md")
        config, feedback, state, discovery = self._make_mocks(feedback_tasks=[fb_task])
        config.orchestrator.task_approval = True

        mock_queue = MagicMock()
        mock_queue.get_approved.return_value = []

        result = gather_tasks(config, feedback, state, discovery,
                              dashboard_active=True, task_approval_queue=mock_queue)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].source, "feedback")

    def test_dashboard_active_returns_approved_tasks(self):
        """Approved tasks should be included in the result."""
        config, feedback, state, discovery = self._make_mocks()
        config.orchestrator.task_approval = True

        approved_task = Task(description="Approved task", priority=3, source="lint")
        mock_queue = MagicMock()
        mock_queue.get_approved.return_value = [approved_task]

        result = gather_tasks(config, feedback, state, discovery,
                              dashboard_active=True, task_approval_queue=mock_queue)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].description, "Approved task")

    def test_dashboard_inactive_no_approval_gate(self):
        """When dashboard is not active, discovered tasks return directly."""
        disc_task = Task(description="Fix lint in foo.py", priority=2, source="lint")
        config, feedback, state, discovery = self._make_mocks(discovered_tasks=[disc_task])
        config.orchestrator.task_approval = True

        mock_queue = MagicMock()
        result = gather_tasks(config, feedback, state, discovery,
                              dashboard_active=False, task_approval_queue=mock_queue)
        self.assertEqual(len(result), 1)
        mock_queue.enqueue.assert_not_called()

    def test_task_approval_disabled(self):
        """When task_approval is False, discovered tasks return directly."""
        disc_task = Task(description="Fix lint in foo.py", priority=2, source="lint")
        config, feedback, state, discovery = self._make_mocks(discovered_tasks=[disc_task])
        config.orchestrator.task_approval = False

        mock_queue = MagicMock()
        result = gather_tasks(config, feedback, state, discovery,
                              dashboard_active=True, task_approval_queue=mock_queue)
        self.assertEqual(len(result), 1)
        mock_queue.enqueue.assert_not_called()

    def test_gather_tasks_logs_approval_gate_blocking(self):
        """Log message when tasks pending approval but none approved."""
        disc_task = Task(description="Fix lint in foo.py", priority=2, source="lint")
        config, feedback, state, discovery = self._make_mocks(discovered_tasks=[disc_task])
        config.orchestrator.task_approval = True
        config.discovery.idea_cooldown_seconds = 600

        mock_queue = MagicMock()
        mock_queue.get_approved.return_value = []
        mock_queue.enqueue.return_value = "lint_foo.py"
        mock_queue.pending_count.return_value = 1

        with patch("shared.logger") as mock_logger:
            result = gather_tasks(config, feedback, state, discovery,
                                  dashboard_active=True, task_approval_queue=mock_queue)
            self.assertEqual(len(result), 0)
            # Check that the approval gate message was logged
            mock_logger.info.assert_any_call(
                "Task approval gate active: %d task(s) pending approval in dashboard, "
                "0 approved. Approve tasks at the dashboard to proceed.",
                1,
            )

    def test_gather_tasks_auto_approves_test_failures(self):
        """test_failure tasks bypass approval gate automatically."""
        test_task = Task(description="Fix test_foo.py::test_bar", priority=1, source="test_failure")
        lint_task = Task(description="Fix lint in foo.py", priority=2, source="lint")
        config, feedback, state, discovery = self._make_mocks(
            discovered_tasks=[test_task, lint_task],
        )
        config.orchestrator.task_approval = True
        config.discovery.idea_cooldown_seconds = 600

        mock_queue = MagicMock()
        mock_queue.get_approved.return_value = []

        result = gather_tasks(config, feedback, state, discovery,
                              dashboard_active=True, task_approval_queue=mock_queue)
        # test_failure task should be auto-approved (in result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].source, "test_failure")
        # lint task should be enqueued, not returned
        mock_queue.enqueue.assert_called_once()


# ---------------------------------------------------------------------------
# clean_description
# ---------------------------------------------------------------------------

class TestCleanDescription(unittest.TestCase):

    def test_strip_fix_test_failure_prefix(self):
        result = clean_description("Fix test failure: test_foo.py fails on assert")
        self.assertEqual(result, "Test_foo.py fails on assert")

    def test_strip_idea_prefix(self):
        result = clean_description("IDEA: improve caching")
        self.assertEqual(result, "Improve caching")

    def test_strip_backticks(self):
        result = clean_description("Fix `foo.py` error")
        self.assertEqual(result, "Fix foo.py error")

    def test_strip_line_numbers(self):
        result = clean_description("Error in foo.py:42-50")
        self.assertEqual(result, "Error in foo.py")

    def test_capitalizes_first_letter(self):
        result = clean_description("Fix lint error in foo.py: unused import")
        self.assertTrue(result[0].isupper())

    def test_empty_after_strip(self):
        # After stripping prefix "Fix test failure: ", leftover is just whitespace
        result = clean_description("Fix test failure: ")
        # The input after .strip() is "Fix test failure:" which doesn't match
        # the prefix "Fix test failure: " (trailing space), so it passes through
        self.assertEqual(result, "Fix test failure:")


# ---------------------------------------------------------------------------
# build_commit_message
# ---------------------------------------------------------------------------

class TestBuildCommitMessage(unittest.TestCase):

    def test_test_failure(self):
        task = Task(description="Fix test failure: test_bar fails", priority=2, source="test_failure")
        result = build_commit_message(task)
        self.assertIn("Fix", result)
        self.assertNotIn("Fix test failure:", result)

    def test_todo_source(self):
        task = Task(
            description="Address TODO in shared.py:10: implement caching",
            priority=3, source="todo",
        )
        result = build_commit_message(task)
        self.assertIn("shared.py", result)

    def test_feedback_source(self):
        task = Task(description="Add dark mode support", priority=1, source="feedback")
        result = build_commit_message(task)
        self.assertEqual(result, "Add dark mode support")

    def test_coverage_source(self):
        task = Task(description="Improve coverage for utils.py", priority=4, source="coverage")
        result = build_commit_message(task)
        self.assertTrue(result.startswith("Add test coverage for"))

    def test_long_subject_truncation(self):
        long_desc = "Fix test failure: " + "a" * 100
        task = Task(description=long_desc, priority=2, source="test_failure")
        result = build_commit_message(task)
        first_line = result.split("\n")[0]
        self.assertLessEqual(len(first_line), 72)
        self.assertTrue(first_line.endswith("..."))


# ---------------------------------------------------------------------------
# build_batch_commit_message
# ---------------------------------------------------------------------------

class TestBuildBatchCommitMessage(unittest.TestCase):

    def test_same_source(self):
        tasks = [
            Task(description="Fix test failure in foo.py", priority=2, source="test_failure"),
            Task(description="Fix test failure in bar.py", priority=2, source="test_failure"),
        ]
        result = build_batch_commit_message(tasks)
        first_line = result.split("\n")[0]
        self.assertIn("Fix test failures", first_line)

    def test_mixed_sources(self):
        tasks = [
            Task(description="Fix test failure in foo.py", priority=2, source="test_failure"),
            Task(description="Fix lint error in bar.py", priority=2, source="lint"),
        ]
        result = build_batch_commit_message(tasks)
        first_line = result.split("\n")[0]
        self.assertIn("fix test failures", first_line.lower())

    def test_body_contains_task_descriptions(self):
        tasks = [
            Task(description="Fix test failure in foo.py", priority=2, source="test_failure"),
            Task(description="Fix test failure in bar.py", priority=2, source="test_failure"),
        ]
        result = build_batch_commit_message(tasks)
        self.assertIn("- ", result)


# ---------------------------------------------------------------------------
# extract_file_names
# ---------------------------------------------------------------------------

class TestExtractFileNames(unittest.TestCase):

    def test_various_extensions(self):
        tasks = [
            Task(description="Fix bug in foo.py", priority=2, source="test_failure"),
            Task(description="Lint error in bar.js", priority=2, source="lint"),
            Task(description="Update config.yaml", priority=3, source="todo"),
        ]
        result = extract_file_names(tasks)
        self.assertIn("foo.py", result)
        self.assertIn("bar.js", result)
        self.assertIn("config.yaml", result)

    def test_no_match(self):
        tasks = [Task(description="Do something without files", priority=2, source="feedback")]
        result = extract_file_names(tasks)
        self.assertEqual(result, [])

    def test_deduplication(self):
        tasks = [
            Task(description="Error in utils.py line 10", priority=2, source="lint"),
            Task(description="Error in utils.py line 20", priority=2, source="lint"),
        ]
        result = extract_file_names(tasks)
        self.assertEqual(result.count("utils.py"), 1)


# ---------------------------------------------------------------------------
# build_task_prompt
# ---------------------------------------------------------------------------

class TestBuildTaskPrompt(unittest.TestCase):

    def test_single_task(self):
        task = Task(description="Fix bug", priority=1, source="test_failure")
        result = build_task_prompt([task], ["main.py"])
        self.assertIn("TASK: Fix bug", result)
        self.assertIn("main.py", result)
        self.assertIn("Do NOT run git commands", result)

    def test_batch_tasks(self):
        tasks = [
            Task(description="Fix A", priority=1, source="lint"),
            Task(description="Fix B", priority=2, source="todo"),
        ]
        result = build_task_prompt(tasks, ["main.py"])
        self.assertIn("TASKS:", result)
        self.assertIn("batch", result.lower())

    def test_with_working_dir(self):
        task = Task(description="Fix bug", priority=1, source="test_failure")
        result = build_task_prompt([task], ["main.py"], working_dir="/tmp/worktree")
        self.assertIn("/tmp/worktree", result)
        self.assertIn("absolute paths", result)


# ---------------------------------------------------------------------------
# build_plan_prompt
# ---------------------------------------------------------------------------

class TestBuildPlanPrompt(unittest.TestCase):

    def test_single_task(self):
        task = Task(description="Fix bug", priority=1, source="test_failure")
        result = build_plan_prompt([task], ["main.py"])
        self.assertIn("TASK: Fix bug", result)
        self.assertIn("Do NOT make any changes", result)
        self.assertIn("plan", result.lower())

    def test_batch_tasks(self):
        tasks = [
            Task(description="Fix A", priority=1, source="lint"),
            Task(description="Fix B", priority=2, source="todo"),
        ]
        result = build_plan_prompt(tasks, ["main.py"])
        self.assertIn("TASKS:", result)
        self.assertIn("plan", result.lower())


# ---------------------------------------------------------------------------
# build_execute_prompt
# ---------------------------------------------------------------------------

class TestBuildExecutePrompt(unittest.TestCase):

    def test_single_task(self):
        task = Task(description="Fix bug", priority=1, source="test_failure")
        result = build_execute_prompt([task], "Step 1: edit foo.py", ["main.py"])
        self.assertIn("PLAN TO EXECUTE", result)
        self.assertIn("Step 1: edit foo.py", result)

    def test_batch_tasks(self):
        tasks = [
            Task(description="Fix A", priority=1, source="lint"),
            Task(description="Fix B", priority=2, source="todo"),
        ]
        result = build_execute_prompt(tasks, "Plan text here", ["main.py"])
        self.assertIn("TASKS:", result)
        self.assertIn("Plan text here", result)


# ---------------------------------------------------------------------------
# build_retry_prompt
# ---------------------------------------------------------------------------

class TestBuildRetryPrompt(unittest.TestCase):

    def test_with_attempt_info(self):
        task = Task(description="Fix bug", priority=1, source="test_failure")
        result = build_retry_prompt(
            [task], "AssertionError: expected True", ["main.py"],
            attempt=2, max_attempts=5,
        )
        self.assertIn("attempt 2 of 5", result)
        self.assertIn("VALIDATION FAILURES", result)
        self.assertIn("AssertionError", result)

    def test_long_output_truncation(self):
        task = Task(description="Fix bug", priority=1, source="test_failure")
        long_output = "x" * 10000
        result = build_retry_prompt([task], long_output, ["main.py"])
        self.assertIn("(truncated)", result)
        # The original 10000 chars should be truncated
        self.assertLess(len(result), 10000 + 2000)  # prompt text + truncated output

    def test_batch_retry(self):
        tasks = [
            Task(description="Fix A", priority=1, source="lint"),
            Task(description="Fix B", priority=2, source="todo"),
        ]
        result = build_retry_prompt(tasks, "error output", ["main.py"])
        self.assertIn("TASKS:", result)

    def test_with_task_history(self):
        task = Task(description="Fix bug", priority=1, source="test_failure")
        history = [
            {"success": False, "error": "SyntaxError in foo.py", "validation_summary": "tests: FAIL"},
            {"success": False, "error": "AssertionError: expected 5 got 3", "validation_summary": "tests: FAIL"},
        ]
        result = build_retry_prompt(
            [task], "new error", ["main.py"], task_history=history,
        )
        self.assertIn("PREVIOUS FAILED ATTEMPTS", result)
        self.assertIn("SyntaxError in foo.py", result)
        self.assertIn("AssertionError: expected 5 got 3", result)

    def test_task_history_skips_successes(self):
        task = Task(description="Fix bug", priority=1, source="test_failure")
        history = [
            {"success": True, "error": "", "validation_summary": "tests: PASS"},
        ]
        result = build_retry_prompt(
            [task], "new error", ["main.py"], task_history=history,
        )
        self.assertNotIn("PREVIOUS FAILED ATTEMPTS", result)

    def test_task_history_empty(self):
        task = Task(description="Fix bug", priority=1, source="test_failure")
        result = build_retry_prompt(
            [task], "new error", ["main.py"], task_history=[],
        )
        self.assertNotIn("PREVIOUS FAILED ATTEMPTS", result)


# ---------------------------------------------------------------------------
# format_validation_errors
# ---------------------------------------------------------------------------

class TestFormatValidationErrors(unittest.TestCase):

    def _make_step(self, name, passed, return_code=0, command="cmd", output=""):
        from types import SimpleNamespace
        return SimpleNamespace(
            name=name, passed=passed, return_code=return_code,
            command=command, output=output,
        )

    def _make_validation(self, steps, summary="all passed"):
        from types import SimpleNamespace
        return SimpleNamespace(steps=steps, summary=summary)

    def test_failed_step_included(self):
        v = self._make_validation([
            self._make_step("tests", False, 1, "pytest", "FAILED test_foo"),
        ], summary="1 failed")
        result = format_validation_errors(v)
        self.assertIn("tests FAILED", result)
        self.assertIn("FAILED test_foo", result)

    def test_include_full_false_omits_output(self):
        v = self._make_validation([
            self._make_step("tests", False, 1, "pytest", "long output"),
        ])
        result = format_validation_errors(v, include_full=False)
        self.assertIn("tests FAILED", result)
        self.assertNotIn("long output", result)

    def test_all_pass_returns_summary(self):
        v = self._make_validation([
            self._make_step("tests", True),
        ], summary="all passed")
        result = format_validation_errors(v)
        self.assertEqual(result, "all passed")

    def test_output_truncated_at_8000(self):
        long_output = "x" * 10000
        v = self._make_validation([
            self._make_step("tests", False, 1, "pytest", long_output),
        ])
        result = format_validation_errors(v)
        self.assertIn("(truncated)", result)


# ---------------------------------------------------------------------------
# _derive_todo_subject
# ---------------------------------------------------------------------------

class TestDeriveTodoSubject(unittest.TestCase):

    def test_full_regex_match(self):
        desc = "Address TODO in shared.py:10: implement caching"
        result = _derive_todo_subject(desc)
        self.assertEqual(result, "Implement caching in shared.py")

    def test_partial_match_no_action(self):
        desc = "Address TODO in shared.py:10: TODO: "
        result = _derive_todo_subject(desc)
        self.assertEqual(result, "Address TODO in shared.py")

    def test_fallback_no_match(self):
        desc = "Some random todo description"
        result = _derive_todo_subject(desc)
        self.assertEqual(result, "Some random todo description")

    def test_fixme_stripped(self):
        desc = "TODO in utils.py:5: FIXME: handle edge case"
        result = _derive_todo_subject(desc)
        self.assertEqual(result, "Handle edge case in utils.py")

    def test_empty_description(self):
        result = _derive_todo_subject("")
        self.assertEqual(result, "Address TODO")


# ---------------------------------------------------------------------------
# topological_sort_tasks
# ---------------------------------------------------------------------------

from shared import topological_sort_tasks


class TestTopologicalSortTasks(unittest.TestCase):

    def test_linear_chain(self):
        a = Task(description="A", priority=1, source="feedback", task_id="a")
        b = Task(description="B", priority=1, source="feedback", task_id="b", depends_on=["a"])
        c = Task(description="C", priority=1, source="feedback", task_id="c", depends_on=["b"])
        result = topological_sort_tasks([c, a, b])
        ids = [t.task_id for t in result]
        self.assertEqual(ids, ["a", "b", "c"])

    def test_diamond_dependency(self):
        a = Task(description="A", priority=1, source="feedback", task_id="a")
        b = Task(description="B", priority=2, source="feedback", task_id="b", depends_on=["a"])
        c = Task(description="C", priority=1, source="feedback", task_id="c", depends_on=["a"])
        d = Task(description="D", priority=1, source="feedback", task_id="d", depends_on=["b", "c"])
        result = topological_sort_tasks([d, b, c, a])
        ids = [t.task_id for t in result]
        self.assertEqual(ids[0], "a")
        self.assertEqual(ids[-1], "d")
        # c should come before b due to lower priority number
        self.assertLess(ids.index("c"), ids.index("b"))

    def test_cycle_detected(self):
        a = Task(description="A", priority=1, source="feedback", task_id="a", depends_on=["b"])
        b = Task(description="B", priority=1, source="feedback", task_id="b", depends_on=["a"])
        with self.assertRaises(ValueError):
            topological_sort_tasks([a, b])

    def test_no_deps(self):
        a = Task(description="A", priority=2, source="feedback", task_id="a")
        b = Task(description="B", priority=1, source="feedback", task_id="b")
        result = topological_sort_tasks([a, b])
        ids = [t.task_id for t in result]
        # b has lower priority number, should come first
        self.assertEqual(ids[0], "b")

    def test_empty_list(self):
        result = topological_sort_tasks([])
        self.assertEqual(result, [])

    def test_unknown_dep_ignored_with_warning(self):
        """Dependencies on task_ids not in the list should be ignored with a warning."""
        a = Task(description="A", priority=1, source="feedback", task_id="a", depends_on=["nonexistent"])
        with patch("shared.logger") as mock_logger:
            result = topological_sort_tasks([a])
        self.assertEqual(len(result), 1)
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("non-existent", warning_msg)

    def test_many_tasks_performance(self):
        """Verify heap-based sort handles larger DAGs correctly."""
        # Create a chain of 50 tasks: 0 -> 1 -> 2 -> ... -> 49
        tasks = []
        for i in range(50):
            deps = [str(i - 1)] if i > 0 else []
            tasks.append(Task(
                description=f"Task {i}", priority=1, source="feedback",
                task_id=str(i), depends_on=deps,
            ))
        # Shuffle input order
        import random
        shuffled = list(tasks)
        random.shuffle(shuffled)
        result = topological_sort_tasks(shuffled)
        ids = [t.task_id for t in result]
        # Should produce 0, 1, 2, ..., 49
        self.assertEqual(ids, [str(i) for i in range(50)])

    def test_wide_fan_out_priority_ordering(self):
        """Root with many children — children should be ordered by priority."""
        root = Task(description="Root", priority=1, source="feedback", task_id="root")
        children = [
            Task(description=f"C{i}", priority=10 - i, source="feedback",
                 task_id=f"c{i}", depends_on=["root"])
            for i in range(5)
        ]
        result = topological_sort_tasks([root] + children)
        self.assertEqual(result[0].task_id, "root")
        # Remaining should be sorted by priority (ascending)
        child_priorities = [t.priority for t in result[1:]]
        self.assertEqual(child_priorities, sorted(child_priorities))


    def test_duplicate_empty_task_ids(self):
        """Multiple tasks with empty task_id should not collide silently."""
        a = Task(description="Task A", priority=1, source="lint")
        b = Task(description="Task B", priority=2, source="lint")
        c = Task(description="Task C", priority=3, source="lint")
        # All have default empty task_id=""
        result = topological_sort_tasks([a, b, c])
        self.assertEqual(len(result), 3)
        # Should be sorted by priority
        descriptions = [t.description for t in result]
        self.assertEqual(descriptions, ["Task A", "Task B", "Task C"])

    def test_mixed_empty_and_named_task_ids(self):
        """Mix of tasks with and without task_ids should all be included."""
        a = Task(description="Named", priority=2, source="lint", task_id="named")
        b = Task(description="Anon 1", priority=1, source="lint")
        c = Task(description="Anon 2", priority=3, source="lint")
        result = topological_sort_tasks([a, b, c])
        self.assertEqual(len(result), 3)

    def test_duplicate_named_task_ids_no_data_loss(self):
        """Two tasks with the same task_id must not silently drop one."""
        a = Task(description="First", priority=1, source="lint", task_id="dup")
        b = Task(description="Second", priority=2, source="lint", task_id="dup")
        result = topological_sort_tasks([a, b])
        self.assertEqual(len(result), 2)
        descriptions = {t.description for t in result}
        self.assertEqual(descriptions, {"First", "Second"})

    def test_triple_duplicate_task_ids(self):
        """Three tasks with the same task_id should all be preserved."""
        tasks = [
            Task(description=f"Task {i}", priority=i, source="lint", task_id="same")
            for i in range(3)
        ]
        result = topological_sort_tasks(tasks)
        self.assertEqual(len(result), 3)


class TestSummarizeMixedSourcesEmptyTasks(unittest.TestCase):
    """Test that _summarize_mixed_sources handles empty task lists."""

    def test_empty_tasks_returns_apply_changes(self):
        """_summarize_mixed_sources with empty tasks should not crash."""
        from shared import _summarize_mixed_sources
        # Empty tasks and sources: should not raise TypeError from set().union()
        result = _summarize_mixed_sources(set(), [])
        self.assertEqual(result, "Apply changes")


class TestSummarizeSameSourceFileCount(unittest.TestCase):
    """Test that _summarize_same_source uses file count, not task count."""

    def test_test_failure_uses_file_count_not_task_count(self):
        """When multiple tasks reference same files, count should be unique files."""
        from shared import _summarize_same_source
        tasks = [
            Task(description="Fix test_foo in tests/test_a.py", priority=1, source="test_failure", source_file="tests/test_a.py"),
            Task(description="Fix test_bar in tests/test_a.py", priority=1, source="test_failure", source_file="tests/test_a.py"),
            Task(description="Fix test_baz in tests/test_b.py", priority=1, source="test_failure", source_file="tests/test_b.py"),
            Task(description="Fix test_qux in tests/test_c.py", priority=1, source="test_failure", source_file="tests/test_c.py"),
        ]
        result = _summarize_same_source("test_failure", tasks)
        # Should say "3 files" (unique files), not "4 files" (task count)
        self.assertIn("3 files", result)

    def test_lint_uses_file_count_not_task_count(self):
        """Lint summary should also use unique file count."""
        from shared import _summarize_same_source
        tasks = [
            Task(description="Fix lint in src/a.py", priority=1, source="lint", source_file="src/a.py"),
            Task(description="Fix lint2 in src/a.py", priority=1, source="lint", source_file="src/a.py"),
            Task(description="Fix lint in src/b.py", priority=1, source="lint", source_file="src/b.py"),
            Task(description="Fix lint in src/c.py", priority=1, source="lint", source_file="src/c.py"),
            Task(description="Fix lint in src/d.py", priority=1, source="lint", source_file="src/d.py"),
        ]
        result = _summarize_same_source("lint", tasks)
        # Should say "4 files" (unique files), not "5 files" (task count)
        self.assertIn("4 files", result)


class TestWebhookUrlSchemeValidation(unittest.TestCase):
    """Test that webhook URLs with non-HTTP schemes are rejected."""

    def test_file_scheme_rejected(self):
        """file:// URLs must be rejected to prevent SSRF."""
        from notifications import NotificationManager, WebhookConfig, NotificationsConfig, NotificationEventsConfig
        config = NotificationsConfig(
            enabled=True,
            webhooks=[WebhookConfig(url="file:///etc/passwd", type="generic")],
            events=NotificationEventsConfig(),
        )
        mgr = NotificationManager(config)
        # _send_webhook should return early without raising
        mgr._send_webhook(config.webhooks[0], "test_event", {"msg": "hi"})
        mgr.shutdown()

    def test_https_scheme_allowed(self):
        """https:// URLs should be allowed (may fail network but not scheme check)."""
        from notifications import NotificationManager, WebhookConfig, NotificationsConfig, NotificationEventsConfig
        from unittest.mock import patch, MagicMock
        config = NotificationsConfig(
            enabled=True,
            webhooks=[WebhookConfig(url="https://hooks.example.com/test", type="generic")],
            events=NotificationEventsConfig(),
        )
        mgr = NotificationManager(config)
        with patch("notifications._resolve_and_check_ip", return_value=(False, "93.184.216.34")), \
             patch("notifications.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"")))
            mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)
            mgr._send_webhook(config.webhooks[0], "test_event", {"msg": "hi"})
            mock_urlopen.assert_called_once()
        mgr.shutdown()


class TestStateLockHeldAfterClose(unittest.TestCase):
    """Test that _file_lock sets held=False after fd is closed."""

    def test_held_false_after_lock_released(self):
        """held flag should be False after exiting _file_lock context."""
        import tempfile
        from config_schema import Config
        from state_lock import LockedStateManager

        config = Config()
        with tempfile.TemporaryDirectory() as tmp:
            config.paths.state_dir = tmp
            config.paths.history_file = tmp + "/history.json"
            mgr = LockedStateManager(config)
            with mgr._file_lock():
                assert getattr(mgr._local, 'held', False) is True
            assert getattr(mgr._local, 'held', False) is False


class TestWorkerDeepCopiesConfig(unittest.TestCase):
    """Test that Worker deep-copies config to avoid shared mutation."""

    def test_worker_config_is_independent_copy(self):
        """Modifying worker.config should not affect original config."""
        from config_schema import Config, ParallelConfig
        from worker import Worker
        from unittest.mock import MagicMock

        config = Config()
        config.parallel = ParallelConfig(enabled=True, max_workers=3, worktree_base_dir=".worktrees")
        original_max_workers = config.parallel.max_workers

        state = MagicMock()
        tasks = [Task(description="test", priority=1, source="lint")]
        worker = Worker(config, tasks, state, worker_id=0, main_repo_dir="/tmp/repo")

        # Mutate worker's config
        worker.config.parallel.max_workers = 99
        # Original should be unaffected
        self.assertEqual(config.parallel.max_workers, original_max_workers)


class TestSummarizeSameSourceNoFileRefs(unittest.TestCase):
    """Test _summarize_same_source when tasks have no file references."""

    def test_test_failure_no_files_uses_task_count(self):
        """When no file names can be extracted, should use task count not '0 files'."""
        from shared import _summarize_same_source
        tasks = [
            Task(description="Test suite is broken", priority=1, source="test_failure"),
            Task(description="Another test fails", priority=1, source="test_failure"),
            Task(description="Third test issue", priority=1, source="test_failure"),
        ]
        result = _summarize_same_source("test_failure", tasks)
        # Should NOT say "0 files" — that's misleading
        self.assertNotIn("0 files", result)
        # Should mention the count of tasks instead
        self.assertIn("3", result)

    def test_lint_no_files_uses_task_count(self):
        """When no file names can be extracted for lint, should use task count."""
        from shared import _summarize_same_source
        tasks = [
            Task(description="Fix whitespace issue", priority=1, source="lint"),
            Task(description="Fix import order", priority=1, source="lint"),
        ]
        result = _summarize_same_source("lint", tasks)
        self.assertNotIn("0 files", result)
        self.assertIn("2", result)


class TestBuildCommitMessageEmptyDescription(unittest.TestCase):
    """Regression: build_commit_message with empty cleaned description should
    produce a meaningful subject, not just the verb (e.g. 'Fix')."""

    def test_empty_description_with_verb(self):
        """When clean_description returns empty, subject should include source type."""
        task = Task(description="Fix test failure: ", priority=2, source="test_failure")
        result = build_commit_message(task)
        # Should not be just "Fix" — should have meaningful content
        self.assertGreater(len(result.strip()), 5)
        self.assertIn("test", result.lower())

    def test_empty_description_lint_source(self):
        """Lint source with empty cleaned description."""
        task = Task(description="Fix lint error: ", priority=2, source="lint")
        result = build_commit_message(task)
        self.assertGreater(len(result.strip()), 5)

    def test_empty_description_coverage_source(self):
        """Coverage source with empty cleaned description."""
        task = Task(description="", priority=4, source="coverage")
        result = build_commit_message(task)
        self.assertGreater(len(result.strip()), 5)


class TestSummarizeMixedSourcesEmptyDescription(unittest.TestCase):
    """Regression: _summarize_mixed_sources should handle empty clean_description
    for claude_idea/feedback tasks without producing empty parts."""

    def test_empty_claude_idea_description(self):
        from shared import _summarize_mixed_sources
        tasks = [
            Task(description="IDEA: ", priority=4, source="claude_idea"),
            Task(description="Fix lint error in foo.py", priority=2, source="lint"),
        ]
        source_groups = {}
        for t in tasks:
            source_groups.setdefault(t.source, []).append(t)
        result = _summarize_mixed_sources(set(source_groups.keys()), tasks)
        # Should not start with a space or have empty parts
        self.assertTrue(result[0].isupper(), f"Subject should start with uppercase: {result!r}")
        self.assertNotIn("  ", result)

    def test_empty_feedback_description(self):
        from shared import _summarize_mixed_sources
        tasks = [
            Task(description="", priority=1, source="feedback"),
            Task(description="Fix test failure in bar.py", priority=2, source="test_failure"),
        ]
        source_groups = {}
        for t in tasks:
            source_groups.setdefault(t.source, []).append(t)
        result = _summarize_mixed_sources(set(source_groups.keys()), tasks)
        self.assertTrue(len(result) > 0)
        self.assertTrue(result[0].isupper())


class TestSyntaxCheckFilesPathTraversal(unittest.TestCase):
    """Test that syntax_check_files rejects path traversal attempts."""

    def test_path_traversal_skipped(self):
        """Files with ../ components outside base_dir should be skipped."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file outside base_dir
            outside_dir = Path(tmpdir) / "outside"
            outside_dir.mkdir()
            evil_file = outside_dir / "evil.py"
            evil_file.write_text("import os; os.system('rm -rf /')\n")

            base_dir = Path(tmpdir) / "project"
            base_dir.mkdir()

            # Try to access the outside file via path traversal
            result = syntax_check_files(["../outside/evil.py"], str(base_dir))
            # Should skip the traversal path, not read/parse the file
            self.assertIsNone(result)

    def test_normal_subdirectory_still_works(self):
        """Files in subdirectories should still be checked normally."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = Path(tmpdir) / "src"
            sub.mkdir()
            py_file = sub / "good.py"
            py_file.write_text("x = 1\n")
            result = syntax_check_files(["src/good.py"], tmpdir)
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
