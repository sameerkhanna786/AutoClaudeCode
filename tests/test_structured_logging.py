"""Tests for structured_logging.py — JSON formatter and apply_json_logging."""

from __future__ import annotations

import json
import logging
import sys
import traceback
import unittest
from datetime import datetime, timezone
from io import StringIO

from structured_logging import JSONFormatter, apply_json_logging


class TestJSONFormatter(unittest.TestCase):

    def setUp(self):
        self.formatter = JSONFormatter()

    def _make_record(self, msg="test message", level=logging.INFO, name="test.logger",
                     args=None, exc_info=None, stack_info=None):
        """Create a LogRecord for testing."""
        record = logging.LogRecord(
            name=name,
            level=level,
            pathname="test.py",
            lineno=42,
            msg=msg,
            args=args or (),
            exc_info=exc_info,
        )
        if stack_info is not None:
            record.stack_info = stack_info
        return record

    def test_basic_format(self):
        record = self._make_record()
        output = self.formatter.format(record)
        parsed = json.loads(output)
        self.assertIn("timestamp", parsed)
        self.assertIn("level", parsed)
        self.assertIn("logger", parsed)
        self.assertIn("message", parsed)
        self.assertEqual(parsed["level"], "INFO")
        self.assertEqual(parsed["logger"], "test.logger")
        self.assertEqual(parsed["message"], "test message")

    def test_timestamp_is_iso(self):
        record = self._make_record()
        output = self.formatter.format(record)
        parsed = json.loads(output)
        ts = parsed["timestamp"]
        # Should be parseable as an ISO 8601 datetime
        dt = datetime.fromisoformat(ts)
        self.assertIsNotNone(dt.tzinfo)

    def test_level_names(self):
        for level, name in [(logging.DEBUG, "DEBUG"), (logging.INFO, "INFO"),
                            (logging.WARNING, "WARNING"), (logging.ERROR, "ERROR")]:
            record = self._make_record(level=level)
            output = self.formatter.format(record)
            parsed = json.loads(output)
            self.assertEqual(parsed["level"], name)

    def test_message_interpolation(self):
        record = self._make_record(msg="hello %s", args=("world",))
        output = self.formatter.format(record)
        parsed = json.loads(output)
        self.assertEqual(parsed["message"], "hello world")

    def test_with_exception(self):
        try:
            raise ValueError("test error")
        except ValueError:
            exc_info = sys.exc_info()

        record = self._make_record(exc_info=exc_info)
        output = self.formatter.format(record)
        parsed = json.loads(output)
        self.assertIn("exception", parsed)
        self.assertIn("ValueError", parsed["exception"])
        self.assertIn("test error", parsed["exception"])

    def test_without_exception(self):
        record = self._make_record()
        output = self.formatter.format(record)
        parsed = json.loads(output)
        self.assertNotIn("exception", parsed)

    def test_with_stack_info(self):
        record = self._make_record(stack_info="Stack trace here")
        output = self.formatter.format(record)
        parsed = json.loads(output)
        self.assertIn("stack_info", parsed)
        self.assertEqual(parsed["stack_info"], "Stack trace here")

    def test_single_line(self):
        record = self._make_record()
        output = self.formatter.format(record)
        self.assertNotIn("\n", output)

    def test_non_serializable_fallback(self):
        """Verify default=str handles non-JSON-serializable objects."""
        record = self._make_record(msg="object: %s", args=(object(),))
        output = self.formatter.format(record)
        # Should not crash — json.dumps(default=str) handles it
        parsed = json.loads(output)
        self.assertIn("object:", parsed["message"])


class TestApplyJsonLogging(unittest.TestCase):

    def test_apply_replaces_formatters(self):
        root = logging.getLogger()
        original_handlers = list(root.handlers)
        try:
            # Add test handlers
            handler1 = logging.StreamHandler(StringIO())
            handler2 = logging.StreamHandler(StringIO())
            root.addHandler(handler1)
            root.addHandler(handler2)

            apply_json_logging()

            self.assertIsInstance(handler1.formatter, JSONFormatter)
            self.assertIsInstance(handler2.formatter, JSONFormatter)
        finally:
            # Clean up
            root.removeHandler(handler1)
            root.removeHandler(handler2)

    def test_apply_preserves_handlers(self):
        root = logging.getLogger()
        original_handlers = list(root.handlers)
        try:
            handler1 = logging.StreamHandler(StringIO())
            handler2 = logging.StreamHandler(StringIO())
            root.addHandler(handler1)
            root.addHandler(handler2)

            count_before = len(root.handlers)
            apply_json_logging()
            count_after = len(root.handlers)

            # Same number of handlers
            self.assertEqual(count_before, count_after)
            self.assertIn(handler1, root.handlers)
            self.assertIn(handler2, root.handlers)
        finally:
            root.removeHandler(handler1)
            root.removeHandler(handler2)


if __name__ == "__main__":
    unittest.main()
