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


if __name__ == "__main__":
    unittest.main()
