"""Tests for context_monitor.py — smart zone enforcement."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

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


if __name__ == "__main__":
    unittest.main()
