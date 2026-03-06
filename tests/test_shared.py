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

    def test_unknown_dep_ignored(self):
        """Dependencies on task_ids not in the list should be ignored."""
        a = Task(description="A", priority=1, source="feedback", task_id="a", depends_on=["nonexistent"])
        result = topological_sort_tasks([a])
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
