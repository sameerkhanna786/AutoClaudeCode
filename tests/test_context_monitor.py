"""Tests for context_monitor.py — smart zone enforcement."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from claude_runner import ClaudeResult
from context_monitor import ContextMonitor, ContextSignals, write_split_tasks_as_feedback
from task_discovery import Task


def _make_config():
    config = MagicMock()
    config.orchestrator.max_context_pct = 80.0
    config.orchestrator.max_split_depth = 3
    return config


class TestContextSignals(unittest.TestCase):

    def test_not_exhausted(self):
        s = ContextSignals(context_window_pct=50.0, hit_max_turns=False)
        self.assertFalse(s.is_exhausted)

    def test_exhausted_by_pct(self):
        s = ContextSignals(context_window_pct=85.0, hit_max_turns=False)
        self.assertTrue(s.is_exhausted)

    def test_exhausted_by_max_turns(self):
        s = ContextSignals(context_window_pct=50.0, hit_max_turns=True)
        self.assertTrue(s.is_exhausted)

    def test_exhausted_by_both(self):
        s = ContextSignals(context_window_pct=90.0, hit_max_turns=True)
        self.assertTrue(s.is_exhausted)


class TestContextMonitor(unittest.TestCase):

    def test_extract_signals_normal(self):
        config = _make_config()
        monitor = ContextMonitor(config)
        result = ClaudeResult(
            success=True, result_text="done",
            input_tokens=5000, output_tokens=3000,
            context_window_pct=4.0,
        )
        signals = monitor.extract_signals(result)
        self.assertEqual(signals.input_tokens, 5000)
        self.assertEqual(signals.output_tokens, 3000)
        self.assertFalse(signals.hit_max_turns)
        self.assertFalse(signals.result_text_empty)

    def test_extract_signals_max_turns(self):
        config = _make_config()
        monitor = ContextMonitor(config)
        result = ClaudeResult(
            success=True, result_text="",
            error="Claude hit max_turns without producing a final result",
            raw_json={"subtype": "error_max_turns"},
        )
        signals = monitor.extract_signals(result)
        self.assertTrue(signals.hit_max_turns)
        self.assertTrue(signals.result_text_empty)

    def test_should_split_high_pct(self):
        config = _make_config()
        monitor = ContextMonitor(config)
        signals = ContextSignals(context_window_pct=85.0)
        self.assertTrue(monitor.should_split(signals))

    def test_should_not_split_low_pct(self):
        config = _make_config()
        monitor = ContextMonitor(config)
        signals = ContextSignals(context_window_pct=50.0)
        self.assertFalse(monitor.should_split(signals))

    def test_should_split_max_turns(self):
        config = _make_config()
        monitor = ContextMonitor(config)
        signals = ContextSignals(hit_max_turns=True)
        self.assertTrue(monitor.should_split(signals))

    def test_generate_split_with_todos(self):
        config = _make_config()
        monitor = ContextMonitor(config)
        task = Task(
            description="Fix all bugs", priority=2, source="feedback",
            task_id="fix-bugs",
        )
        result = ClaudeResult(
            success=True,
            result_text=(
                "I fixed the main bug.\n"
                "TODO: Fix the edge case in parser.py\n"
                "TODO: Add test for the new validation\n"
            ),
        )
        split_tasks = monitor.generate_split_tasks(task, result)
        self.assertEqual(len(split_tasks), 2)
        self.assertTrue(all("fix-bugs" in t.depends_on for t in split_tasks))
        self.assertTrue(any("parser" in t.description for t in split_tasks))

    def test_generate_split_empty_result(self):
        config = _make_config()
        monitor = ContextMonitor(config)
        task = Task(
            description="Fix all bugs", priority=2, source="feedback",
            task_id="fix-bugs",
        )
        result = ClaudeResult(success=True, result_text="")
        split_tasks = monitor.generate_split_tasks(task, result)
        self.assertEqual(len(split_tasks), 1)
        self.assertIn("Continue", split_tasks[0].description)

    def test_generate_split_max_depth(self):
        config = _make_config()
        config.orchestrator.max_split_depth = 2
        monitor = ContextMonitor(config)
        task = Task(
            description="Fix bug", priority=2, source="feedback",
            task_id="fix",
        )
        result = ClaudeResult(
            success=True,
            result_text="TODO: Something remaining\n",
        )
        # At max depth, should return empty
        split_tasks = monitor.generate_split_tasks(task, result, split_depth=2)
        self.assertEqual(len(split_tasks), 0)

    def test_generate_split_no_todos_continuation(self):
        config = _make_config()
        monitor = ContextMonitor(config)
        task = Task(
            description="Refactor module", priority=3, source="quality",
            task_id="refactor-1",
        )
        result = ClaudeResult(
            success=True,
            result_text="I made some changes but ran out of space.",
        )
        split_tasks = monitor.generate_split_tasks(task, result)
        self.assertEqual(len(split_tasks), 1)
        self.assertIn("Continue", split_tasks[0].description)


class TestWriteSplitTasks(unittest.TestCase):

    def test_write_feedback_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tasks = [
                Task(
                    description="Fix parser edge case",
                    priority=2, source="feedback",
                    task_id="fix__split_1_0",
                    depends_on=["fix"],
                ),
            ]
            count = write_split_tasks_as_feedback(tasks, tmpdir)
            self.assertEqual(count, 1)

            files = list(Path(tmpdir).glob("split-*.md"))
            self.assertEqual(len(files), 1)

            content = files[0].read_text()
            self.assertIn("task_id:", content)
            self.assertIn("depends_on:", content)
            self.assertIn("Fix parser edge case", content)

    def test_write_empty_tasks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            count = write_split_tasks_as_feedback([], tmpdir)
            self.assertEqual(count, 0)


class TestWriteSplitTasksAtomicWrites(unittest.TestCase):
    """Tests that split task writes are atomic (tempfile + os.replace)."""

    def test_write_creates_file_atomically(self):
        """Verify no .tmp files remain after successful write."""
        task = Task(
            description="Atomic test task",
            priority=1,
            source="test",
            task_id="atomic-1",
            depends_on=["parent-1"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            count = write_split_tasks_as_feedback([task], tmpdir)
            self.assertEqual(count, 1)
            # No .tmp files should remain
            tmp_files = list(Path(tmpdir).glob("*.tmp"))
            self.assertEqual(len(tmp_files), 0)
            # The actual file should exist
            md_files = list(Path(tmpdir).glob("split-*.md"))
            self.assertEqual(len(md_files), 1)

    def test_write_failure_cleans_up_tmp(self):
        """On write failure, tmp files should be cleaned up."""
        task = Task(
            description="Fail task",
            priority=1,
            source="test",
            task_id="fail-1",
            depends_on=[],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            # Make directory read-only to force write failure
            import os, stat
            os.chmod(tmpdir, stat.S_IRUSR | stat.S_IXUSR)
            try:
                count = write_split_tasks_as_feedback([task], tmpdir)
                self.assertEqual(count, 0)
            finally:
                os.chmod(tmpdir, stat.S_IRWXU)


class TestWriteSplitTasksFdCleanup(unittest.TestCase):
    """Tests that file descriptors are properly cleaned up when os.fdopen fails."""

    def test_fd_closed_when_fdopen_fails(self):
        """If os.fdopen() raises, the raw fd from mkstemp must still be closed."""
        task = Task(
            description="Test fd cleanup",
            priority=1,
            source="test",
            task_id="fd-cleanup-1",
            depends_on=["parent"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            close_calls = []
            original_close = os.close

            def tracking_close(fd):
                close_calls.append(fd)
                return original_close(fd)

            with patch("context_monitor.os.fdopen", side_effect=OSError("fdopen failed")), \
                 patch("context_monitor.os.close", side_effect=tracking_close):
                count = write_split_tasks_as_feedback([task], tmpdir)

            self.assertEqual(count, 0)
            # The fd from mkstemp should have been closed
            self.assertGreaterEqual(len(close_calls), 1)


class TestTaskIdSanitization(unittest.TestCase):
    """Tests that task IDs are sanitized in filenames to prevent path traversal."""

    def test_dotdot_in_task_id_sanitized(self):
        """Task IDs with '..' should be sanitized in the filename."""
        task = Task(
            description="Malicious task",
            priority=1,
            source="test",
            task_id="../../etc/passwd",
            depends_on=[],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            count = write_split_tasks_as_feedback([task], tmpdir)
            self.assertEqual(count, 1)
            files = list(Path(tmpdir).glob("split-*.md"))
            self.assertEqual(len(files), 1)
            # Filename should not contain '..' or '/'
            self.assertNotIn("..", files[0].name)
            self.assertNotIn("/", files[0].name)

    def test_null_byte_in_task_id_sanitized(self):
        """Task IDs with null bytes should be sanitized."""
        task = Task(
            description="Null byte task",
            priority=1,
            source="test",
            task_id="task\x00id",
            depends_on=[],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            count = write_split_tasks_as_feedback([task], tmpdir)
            self.assertEqual(count, 1)
            files = list(Path(tmpdir).glob("split-*.md"))
            self.assertEqual(len(files), 1)
            self.assertNotIn("\x00", files[0].name)


class TestEmptyTaskIdSplitTasks(unittest.TestCase):
    """Tests that split tasks get valid IDs even when parent task_id is empty."""

    def test_empty_task_id_gets_fallback_id(self):
        """When task.task_id is empty, split tasks should NOT have empty depends_on."""
        config = _make_config()
        monitor = ContextMonitor(config)
        task = Task(
            description="Fix all bugs", priority=2, source="feedback",
            task_id="",  # Empty task_id
        )
        result = ClaudeResult(success=True, result_text="")
        split_tasks = monitor.generate_split_tasks(task, result)
        self.assertEqual(len(split_tasks), 1)
        # depends_on should NOT contain an empty string
        for dep in split_tasks[0].depends_on:
            self.assertNotEqual(dep, "", "Split task should not depend on empty string")
        # task_id should NOT start with __split (which would mean parent_id was empty)
        self.assertFalse(
            split_tasks[0].task_id.startswith("__split"),
            "Split task ID should not start with __split (empty parent)",
        )

    def test_empty_task_id_with_todos_gets_fallback(self):
        """Split tasks from TODO extraction also get valid IDs with empty parent."""
        config = _make_config()
        monitor = ContextMonitor(config)
        task = Task(
            description="Refactor module", priority=3, source="quality",
            task_id="",  # Empty
        )
        result = ClaudeResult(
            success=True,
            result_text="TODO: Fix the edge case in parser.py\n",
        )
        split_tasks = monitor.generate_split_tasks(task, result)
        self.assertGreater(len(split_tasks), 0)
        for st in split_tasks:
            for dep in st.depends_on:
                self.assertNotEqual(dep, "")
            self.assertNotEqual(st.task_id, "")

    def test_nonempty_task_id_unchanged(self):
        """When task_id is set, behavior is unchanged."""
        config = _make_config()
        monitor = ContextMonitor(config)
        task = Task(
            description="Fix bug", priority=2, source="feedback",
            task_id="my-task",
        )
        result = ClaudeResult(success=True, result_text="")
        split_tasks = monitor.generate_split_tasks(task, result)
        self.assertEqual(len(split_tasks), 1)
        self.assertIn("my-task", split_tasks[0].depends_on)
        self.assertTrue(split_tasks[0].task_id.startswith("my-task__split"))


class TestExtractSignalsNoneResultText(unittest.TestCase):
    """Tests that extract_signals handles result_text=None without crashing."""

    def test_result_text_none_does_not_crash(self):
        """extract_signals should not raise AttributeError when result_text is None."""
        config = _make_config()
        monitor = ContextMonitor(config)
        result = ClaudeResult(
            success=False,
            result_text=None,  # type: ignore[arg-type]
            error="some error",
        )
        # After the fix, this should not raise AttributeError on None.strip()
        signals = monitor.extract_signals(result)
        self.assertTrue(signals.result_text_empty)

    def test_result_text_empty_string(self):
        """extract_signals should handle empty string result_text normally."""
        config = _make_config()
        monitor = ContextMonitor(config)
        result = ClaudeResult(success=True, result_text="")
        signals = monitor.extract_signals(result)
        self.assertTrue(signals.result_text_empty)


class TestSplitTaskIdDeterminism(unittest.TestCase):
    """Split task IDs must be deterministic (not depend on object id())."""

    def test_empty_task_id_produces_deterministic_parent_id(self):
        """Two calls with identical empty-id tasks should produce the same parent_id prefix."""
        config = _make_config()
        monitor = ContextMonitor(config)
        task1 = Task(description="Fix bugs", priority=2, source="feedback", task_id="")
        task2 = Task(description="Fix bugs", priority=2, source="feedback", task_id="")
        result = ClaudeResult(success=True, result_text="")

        splits1 = monitor.generate_split_tasks(task1, result)
        splits2 = monitor.generate_split_tasks(task2, result)

        # Parent IDs embedded in depends_on should be identical for identical tasks
        self.assertEqual(splits1[0].depends_on, splits2[0].depends_on)
        # Task IDs should also be identical for identical inputs
        self.assertEqual(splits1[0].task_id, splits2[0].task_id)


if __name__ == "__main__":
    unittest.main()
