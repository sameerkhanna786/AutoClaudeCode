"""Tests for dashboard.py — data access functions and HTTP handler."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from dashboard import (
    FEEDBACK_FILENAME_RE,
    MAX_LOG_LINES,
    DashboardHandler,
    _load_config,
    _read_cycle_state,
    compute_status,
    get_feedback_files,
    get_loc_for_commits,
    is_orchestrator_running,
    load_history,
    read_log_tail,
)


class TestLoadHistory(unittest.TestCase):

    def test_valid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"timestamp": 1.0, "success": True}], f)
            path = f.name
        try:
            records = load_history(path)
            self.assertEqual(len(records), 1)
            self.assertTrue(records[0]["success"])
        finally:
            os.unlink(path)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("")
            path = f.name
        try:
            self.assertEqual(load_history(path), [])
        finally:
            os.unlink(path)

    def test_missing_file(self):
        self.assertEqual(load_history("/nonexistent/path/history.json"), [])

    def test_corrupt_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{invalid json")
            path = f.name
        try:
            self.assertEqual(load_history(path), [])
        finally:
            os.unlink(path)


class TestReadCycleState(unittest.TestCase):

    def test_valid_cycle_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "current_cycle.json"
            state_file.write_text(json.dumps({"phase": "executing"}))
            result = _read_cycle_state(tmpdir)
            self.assertEqual(result["phase"], "executing")

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _read_cycle_state(tmpdir)
            self.assertIsNone(result)


class TestIsOrchestratorRunning(unittest.TestCase):

    def test_no_lock_file(self):
        running, pid = is_orchestrator_running("/nonexistent/lock.pid")
        self.assertFalse(running)
        self.assertIsNone(pid)

    def test_stale_pid(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pid", delete=False) as f:
            f.write("999999999")  # unlikely to exist
            path = f.name
        try:
            running, pid = is_orchestrator_running(path)
            self.assertFalse(running)
            self.assertIsNone(pid)
        finally:
            os.unlink(path)

    def test_current_pid(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pid", delete=False) as f:
            f.write(str(os.getpid()))
            path = f.name
        try:
            running, pid = is_orchestrator_running(path)
            self.assertTrue(running)
            self.assertEqual(pid, os.getpid())
        finally:
            os.unlink(path)


class TestComputeStatus(unittest.TestCase):

    def _make_cfg(self, tmpdir):
        history_path = os.path.join(tmpdir, "history.json")
        lock_path = os.path.join(tmpdir, "lock.pid")
        return {
            "target_dir": tmpdir,
            "history_file": history_path,
            "state_dir": tmpdir,
            "lock_file": lock_path,
            "max_consecutive_failures": 5,
            "max_cycles_per_hour": 30,
            "max_cost_usd_per_hour": 10.0,
            "min_disk_space_mb": 500,
        }

    def test_empty_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._make_cfg(tmpdir)
            status = compute_status(cfg)
            self.assertEqual(status["consecutive_failures"], 0)
            self.assertEqual(status["cycles_per_hour"], 0)
            self.assertEqual(status["cost_per_hour"], 0)
            self.assertEqual(status["success_rate"], 0.0)
            self.assertEqual(status["total_cycles"], 0)

    def test_with_records(self):
        import time
        now = time.time()
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._make_cfg(tmpdir)
            records = [
                {"timestamp": now - 100, "success": True, "cost_usd": 0.5},
                {"timestamp": now - 50, "success": True, "cost_usd": 0.3},
                {"timestamp": now - 10, "success": False, "cost_usd": 0.2},
            ]
            Path(cfg["history_file"]).write_text(json.dumps(records))
            status = compute_status(cfg)
            self.assertEqual(status["consecutive_failures"], 1)
            self.assertEqual(status["cycles_per_hour"], 3)
            self.assertAlmostEqual(status["cost_per_hour"], 1.0, places=2)
            self.assertAlmostEqual(status["success_rate"], 66.7, places=1)


class TestGetFeedbackFiles(unittest.TestCase):

    def test_categorized_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pending_dir = Path(tmpdir) / "feedback"
            done_dir = Path(tmpdir) / "feedback" / "done"
            failed_dir = Path(tmpdir) / "feedback" / "failed"
            pending_dir.mkdir(parents=True)
            done_dir.mkdir(parents=True)
            failed_dir.mkdir(parents=True)

            (pending_dir / "task1.md").write_text("pending task")
            (done_dir / "task2.md").write_text("done task")
            (failed_dir / "task3.txt").write_text("failed task")

            cfg = {
                "feedback_dir": str(pending_dir),
                "feedback_done_dir": str(done_dir),
                "feedback_failed_dir": str(failed_dir),
            }
            result = get_feedback_files(cfg)
            self.assertEqual(len(result["pending"]), 1)
            self.assertEqual(result["pending"][0]["name"], "task1.md")
            self.assertEqual(len(result["done"]), 1)
            self.assertEqual(len(result["failed"]), 1)


class TestReadLogTail(unittest.TestCase):

    def test_read_last_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            for i in range(10):
                f.write(f"line {i}\n")
            path = f.name
        try:
            lines = read_log_tail(path, 5)
            self.assertEqual(len(lines), 5)
            self.assertEqual(lines[0], "line 5")
            self.assertEqual(lines[-1], "line 9")
        finally:
            os.unlink(path)

    def test_clamps_to_max(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            for i in range(10):
                f.write(f"line {i}\n")
            path = f.name
        try:
            lines = read_log_tail(path, MAX_LOG_LINES + 100)
            self.assertEqual(len(lines), 10)
        finally:
            os.unlink(path)

    def test_missing_file(self):
        lines = read_log_tail("/nonexistent/log.txt", 10)
        self.assertEqual(lines, [])


class TestLoadConfig(unittest.TestCase):

    def test_defaults_nonexistent_file(self):
        result = _load_config("/nonexistent/config.yaml")
        self.assertEqual(result["target_dir"], ".")
        self.assertEqual(result["max_consecutive_failures"], 5)

    def test_with_yaml(self):
        import yaml
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({
                "target_dir": "/custom",
                "safety": {"max_consecutive_failures": 10},
            }, f)
            path = f.name
        try:
            result = _load_config(path)
            self.assertEqual(result["target_dir"], "/custom")
            self.assertEqual(result["max_consecutive_failures"], 10)
        finally:
            os.unlink(path)


class TestGetLocForCommits(unittest.TestCase):

    def test_invalid_hash(self):
        cache = {}
        lock = threading.Lock()
        result = get_loc_for_commits("/tmp", ["not-a-hex-hash!"], cache, lock)
        self.assertIn("not-a-hex-hash!", result)
        self.assertEqual(result["not-a-hex-hash!"]["error"], "invalid hash")


class TestFeedbackFilenameRegex(unittest.TestCase):

    def test_valid_filenames(self):
        valid = [
            "task.md", "01-fix-bug.md", "my_task.txt",
            "feature.request.md", "a.md",
        ]
        for name in valid:
            self.assertIsNotNone(
                FEEDBACK_FILENAME_RE.match(name),
                f"Expected {name!r} to match",
            )

    def test_invalid_filenames(self):
        invalid = [
            "", ".md", "../evil.md", "file.py",
            "-starts-with-dash.md", ".hidden.md",
        ]
        for name in invalid:
            self.assertIsNone(
                FEEDBACK_FILENAME_RE.match(name),
                f"Expected {name!r} to NOT match",
            )


class TestDashboardHandlerAPI(unittest.TestCase):
    """Test DashboardHandler API methods via mock request/response objects."""

    def _make_handler(self, method="GET", path="/", body=None, headers=None):
        """Create a mock DashboardHandler with a writable response buffer."""
        handler = DashboardHandler.__new__(DashboardHandler)
        handler.wfile = BytesIO()
        handler.rfile = BytesIO(body.encode() if body else b"")
        handler.path = path
        handler.command = method
        handler.request_version = "HTTP/1.1"
        handler.headers = MagicMock()
        handler.headers.get = lambda key, default="0": (
            headers.get(key, default) if headers else default
        )
        handler.dashboard_cfg = {
            "target_dir": ".",
            "history_file": "/nonexistent/history.json",
            "state_dir": "/nonexistent",
            "lock_file": "/nonexistent/lock.pid",
            "log_file": "/nonexistent/log.txt",
            "feedback_dir": "/nonexistent/feedback",
            "feedback_done_dir": "/nonexistent/feedback/done",
            "feedback_failed_dir": "/nonexistent/feedback/failed",
            "max_consecutive_failures": 5,
            "max_cycles_per_hour": 30,
            "max_cost_usd_per_hour": 10.0,
            "min_disk_space_mb": 500,
        }
        handler.loc_cache = {}
        handler.loc_lock = threading.Lock()
        # Mock the response methods
        handler._headers_buffer = []
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        return handler

    def test_api_status_returns_json(self):
        handler = self._make_handler(path="/api/status")
        handler._api_status({})
        output = handler.wfile.getvalue().decode()
        data = json.loads(output)
        self.assertIn("running", data)
        self.assertIn("consecutive_failures", data)
        self.assertIn("success_rate", data)

    def test_api_history_with_filters(self):
        handler = self._make_handler(path="/api/history")
        handler._api_history({"limit": ["10"], "offset": ["0"]})
        output = handler.wfile.getvalue().decode()
        data = json.loads(output)
        self.assertIn("total", data)
        self.assertIn("records", data)

    def test_api_feedback_submit_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = self._make_handler(
                method="POST",
                path="/api/feedback",
                body=json.dumps({"filename": "task.md", "content": "do something"}),
                headers={"Content-Length": "100"},
            )
            handler.dashboard_cfg["feedback_dir"] = tmpdir
            handler._api_feedback_submit()
            output = handler.wfile.getvalue().decode()
            data = json.loads(output)
            self.assertTrue(data.get("ok"))
            self.assertTrue((Path(tmpdir) / "task.md").exists())

    def test_api_feedback_submit_invalid_filename(self):
        handler = self._make_handler(
            method="POST",
            path="/api/feedback",
            body=json.dumps({"filename": "../evil.md", "content": "hack"}),
            headers={"Content-Length": "100"},
        )
        handler._api_feedback_submit()
        output = handler.wfile.getvalue().decode()
        data = json.loads(output)
        self.assertIn("error", data)

    def test_api_feedback_delete_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            task_file = Path(tmpdir) / "task.md"
            task_file.write_text("content")
            handler = self._make_handler(method="DELETE", path="/api/feedback/task.md")
            handler.dashboard_cfg["feedback_dir"] = tmpdir
            handler._api_feedback_delete("task.md")
            output = handler.wfile.getvalue().decode()
            data = json.loads(output)
            self.assertTrue(data.get("ok"))
            self.assertFalse(task_file.exists())

    def test_api_feedback_delete_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = self._make_handler(method="DELETE", path="/api/feedback/nope.md")
            handler.dashboard_cfg["feedback_dir"] = tmpdir
            handler._api_feedback_delete("nope.md")
            output = handler.wfile.getvalue().decode()
            data = json.loads(output)
            self.assertIn("error", data)
            handler.send_response.assert_called_with(404)


if __name__ == "__main__":
    unittest.main()
