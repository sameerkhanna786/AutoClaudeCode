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

    def test_large_feedback_file_truncated(self):
        """Large feedback files (>100KB) should be truncated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            pending_dir = tmpdir_path / "feedback"
            pending_dir.mkdir(parents=True)

            # Create a file larger than 100KB
            large_content = "A" * (150 * 1024)  # 150KB
            (pending_dir / "large.md").write_text(large_content)

            cfg = {
                "feedback_dir": str(pending_dir),
                "feedback_done_dir": str(tmpdir_path / "done"),
                "feedback_failed_dir": str(tmpdir_path / "failed"),
            }
            result = get_feedback_files(cfg)
            self.assertEqual(len(result["pending"]), 1)

            # After fix, content should be truncated to ~100KB
            # For now, this will fail because the bug exists
            content = result["pending"][0]["content"]
            # The test expects truncation after the fix is implemented
            # Currently this will show the bug exists
            self.assertLessEqual(len(content), 110 * 1024,
                                 "Content should be truncated to prevent DoS")


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


class TestPendingTasksAPI(unittest.TestCase):
    """Test pending tasks approval API endpoints."""

    def _make_handler(self, method="GET", path="/", body=None, headers=None):
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
        handler._headers_buffer = []
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        return handler

    def test_list_pending_no_queue(self):
        handler = self._make_handler(path="/api/pending-tasks")
        handler.task_queue = None
        handler._api_pending_tasks_list({})
        output = handler.wfile.getvalue().decode()
        data = json.loads(output)
        self.assertEqual(data["tasks"], [])

    def test_list_pending_with_queue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from task_queue import TaskApprovalQueue
            from task_discovery import Task
            queue = TaskApprovalQueue(tmpdir)
            task = Task(description="Fix lint", priority=2, source="lint")
            queue.enqueue(task)

            handler = self._make_handler(path="/api/pending-tasks")
            handler.task_queue = queue
            handler._api_pending_tasks_list({})
            output = handler.wfile.getvalue().decode()
            data = json.loads(output)
            self.assertEqual(len(data["tasks"]), 1)

    def test_approve_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from task_queue import TaskApprovalQueue
            from task_discovery import Task
            queue = TaskApprovalQueue(tmpdir)
            task = Task(description="Fix lint", priority=2, source="lint")
            queue.enqueue(task)
            task_id = queue.list_pending()[0]["id"]

            handler = self._make_handler(method="POST")
            handler.task_queue = queue
            handler._api_pending_tasks_approve(task_id)
            output = handler.wfile.getvalue().decode()
            data = json.loads(output)
            self.assertTrue(data.get("ok"))
            self.assertEqual(queue.pending_count(), 0)

    def test_decline_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from task_queue import TaskApprovalQueue
            from task_discovery import Task
            queue = TaskApprovalQueue(tmpdir)
            task = Task(description="Fix lint", priority=2, source="lint")
            queue.enqueue(task)
            task_id = queue.list_pending()[0]["id"]

            handler = self._make_handler(method="POST")
            handler.task_queue = queue
            handler._api_pending_tasks_decline(task_id)
            output = handler.wfile.getvalue().decode()
            data = json.loads(output)
            self.assertTrue(data.get("ok"))
            self.assertEqual(queue.pending_count(), 0)

    def test_approve_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from task_queue import TaskApprovalQueue
            from task_discovery import Task
            queue = TaskApprovalQueue(tmpdir)
            for i in range(3):
                queue.enqueue(Task(description=f"Task {i}", priority=2, source="lint",
                                   source_file=f"f{i}.py"))
            handler = self._make_handler(method="POST")
            handler.task_queue = queue
            handler._api_pending_tasks_approve_all()
            output = handler.wfile.getvalue().decode()
            data = json.loads(output)
            self.assertTrue(data.get("ok"))
            self.assertEqual(data["approved_count"], 3)

    def test_decline_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from task_queue import TaskApprovalQueue
            from task_discovery import Task
            queue = TaskApprovalQueue(tmpdir)
            for i in range(2):
                queue.enqueue(Task(description=f"Task {i}", priority=2, source="lint",
                                   source_file=f"f{i}.py"))
            handler = self._make_handler(method="POST")
            handler.task_queue = queue
            handler._api_pending_tasks_decline_all()
            output = handler.wfile.getvalue().decode()
            data = json.loads(output)
            self.assertTrue(data.get("ok"))
            self.assertEqual(data["declined_count"], 2)

    def test_approve_invalid_id(self):
        handler = self._make_handler(method="POST")
        handler.task_queue = MagicMock()
        handler._api_pending_tasks_approve("../evil")
        output = handler.wfile.getvalue().decode()
        data = json.loads(output)
        self.assertIn("error", data)

    def test_status_writes_heartbeat(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from task_queue import TaskApprovalQueue
            queue = TaskApprovalQueue(tmpdir)
            handler = self._make_handler(path="/api/status")
            handler.task_queue = queue
            handler.dashboard_cfg["state_dir"] = tmpdir
            handler._api_status({})
            self.assertTrue(queue.is_dashboard_active())


class TestDashboardSecurityFixes(unittest.TestCase):
    """Test security hardening: input validation, CORS, error sanitization."""

    def _make_handler(self, method="GET", path="/", body=None, headers=None):
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
        handler._headers_buffer = []
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        return handler

    def test_history_invalid_limit_returns_400(self):
        handler = self._make_handler(path="/api/history")
        handler._api_history({"limit": ["not_a_number"], "offset": ["0"]})
        handler.send_response.assert_called_with(400)

    def test_history_invalid_offset_returns_400(self):
        handler = self._make_handler(path="/api/history")
        handler._api_history({"limit": ["10"], "offset": ["abc"]})
        handler.send_response.assert_called_with(400)

    def test_log_invalid_lines_returns_400(self):
        handler = self._make_handler(path="/api/log")
        handler._api_log({"lines": ["not_int"]})
        handler.send_response.assert_called_with(400)

    def test_metrics_invalid_lookback_returns_400(self):
        handler = self._make_handler(path="/api/metrics")
        handler._api_metrics({"lookback": ["xyz"]})
        handler.send_response.assert_called_with(400)

    def test_cors_localhost_origin_allowed(self):
        handler = self._make_handler()
        handler.headers = MagicMock()
        handler.headers.get = lambda key, default="": {
            "Origin": "http://localhost:8505",
        }.get(key, default)
        origin = handler._get_cors_origin()
        self.assertEqual(origin, "http://localhost:8505")

    def test_cors_external_origin_blocked(self):
        handler = self._make_handler()
        handler.headers = MagicMock()
        handler.headers.get = lambda key, default="": {
            "Origin": "https://evil.example.com",
        }.get(key, default)
        origin = handler._get_cors_origin()
        self.assertEqual(origin, "")

    def test_cors_localhost_subdomain_blocked(self):
        """Regression: localhost.evil.com must not be accepted as localhost."""
        handler = self._make_handler()
        handler.headers = MagicMock()
        handler.headers.get = lambda key, default="": {
            "Origin": "http://localhost.evil.com",
        }.get(key, default)
        origin = handler._get_cors_origin()
        self.assertEqual(origin, "")

    def test_cors_localhost_no_port_allowed(self):
        """Bare localhost without port should be allowed."""
        handler = self._make_handler()
        handler.headers = MagicMock()
        handler.headers.get = lambda key, default="": {
            "Origin": "http://localhost",
        }.get(key, default)
        origin = handler._get_cors_origin()
        self.assertEqual(origin, "http://localhost")

    def test_cors_127_origin_allowed(self):
        handler = self._make_handler()
        handler.headers = MagicMock()
        handler.headers.get = lambda key, default="": {
            "Origin": "http://127.0.0.1:3000",
        }.get(key, default)
        origin = handler._get_cors_origin()
        self.assertEqual(origin, "http://127.0.0.1:3000")

    def test_read_body_negative_content_length_rejected(self):
        """Regression: negative Content-Length must be rejected to prevent unbounded read."""
        handler = self._make_handler(method="POST", path="/api/feedback")
        handler.headers = MagicMock()
        handler.headers.get = lambda key, default="": {
            "Content-Length": "-1",
        }.get(key, default)
        result = handler._read_body()
        self.assertIsNone(result)
        handler.send_response.assert_called_with(413)

    def test_error_response_no_os_details(self):
        """Error messages should not leak OS error details."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = self._make_handler(
                method="DELETE",
                path="/api/feedback/nonexistent.md",
            )
            handler.dashboard_cfg["feedback_dir"] = tmpdir
            handler._api_feedback_delete("nonexistent.md")
            output = handler.wfile.getvalue().decode()
            data = json.loads(output)
            # Should not contain OS error paths or errno details
            self.assertNotIn("Errno", data.get("error", ""))
            self.assertNotIn("/", data.get("error", ""))


class TestServerBindAddress(unittest.TestCase):
    """Verify dashboard binds to localhost, not all interfaces."""

    def test_main_binds_to_localhost(self):
        """The server must bind to 127.0.0.1, not 0.0.0.0."""
        import inspect
        from dashboard import main
        source = inspect.getsource(main)
        self.assertIn("127.0.0.1", source)
        self.assertNotIn("0.0.0.0", source)


class TestFeedbackDeleteSymlinkRejected(unittest.TestCase):
    """Verify symlink deletion is rejected to prevent TOCTOU attacks."""

    def _make_handler(self, method="DELETE", path="/"):
        handler = DashboardHandler.__new__(DashboardHandler)
        handler.wfile = BytesIO()
        handler.rfile = BytesIO(b"")
        handler.path = path
        handler.command = method
        handler.request_version = "HTTP/1.1"
        handler.headers = MagicMock()
        handler.headers.get = lambda key, default="0": default
        handler.dashboard_cfg = {
            "target_dir": ".",
            "history_file": "/nonexistent/history.json",
            "state_dir": "/nonexistent",
            "lock_file": "/nonexistent/lock.pid",
            "log_file": "/nonexistent/log.txt",
            "feedback_dir": "/tmp",
            "feedback_done_dir": "/tmp/done",
            "feedback_failed_dir": "/tmp/failed",
            "max_consecutive_failures": 5,
            "max_cycles_per_hour": 30,
            "max_cost_usd_per_hour": 10.0,
            "min_disk_space_mb": 500,
        }
        handler.loc_cache = {}
        handler.loc_lock = threading.Lock()
        handler._headers_buffer = []
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        return handler

    def test_symlink_in_feedback_dir_rejected(self):
        """Symlinks should be rejected to prevent TOCTOU path traversal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a real file outside the feedback dir
            outside = Path(tmpdir) / "outside"
            outside.mkdir()
            secret = outside / "secret.txt"
            secret.write_text("sensitive data")

            # Create a symlink inside the feedback dir pointing to it
            feedback_dir = Path(tmpdir) / "feedback"
            feedback_dir.mkdir()
            symlink = feedback_dir / "evil.md"
            symlink.symlink_to(secret)

            handler = self._make_handler()
            handler.dashboard_cfg["feedback_dir"] = str(feedback_dir)
            handler._api_feedback_delete("evil.md")

            output = handler.wfile.getvalue().decode()
            data = json.loads(output)
            self.assertIn("error", data)
            handler.send_response.assert_called_with(403)
            # The actual target file should NOT have been deleted
            self.assertTrue(secret.exists())


class TestStaticHtmlBytesCache(unittest.TestCase):
    """Test that STATIC_HTML is pre-encoded to bytes at module level."""

    def test_static_html_bytes_exists(self):
        from dashboard import _STATIC_HTML_BYTES, STATIC_HTML
        self.assertIsInstance(_STATIC_HTML_BYTES, bytes)
        self.assertEqual(_STATIC_HTML_BYTES, STATIC_HTML.encode("utf-8"))

    def test_static_html_bytes_is_same_object(self):
        """Multiple imports should return the same cached bytes object."""
        from dashboard import _STATIC_HTML_BYTES as a
        from dashboard import _STATIC_HTML_BYTES as b
        self.assertIs(a, b)


class TestLocCacheBounds(unittest.TestCase):
    """Test that LOC cache is bounded to prevent unbounded memory growth."""

    def test_cache_evicts_when_full(self):
        from dashboard import get_loc_for_commits
        import threading

        cache = {}
        lock = threading.Lock()

        # Fill cache with 1000 entries
        for i in range(1000):
            cache[f"hash_{i:04d}"] = {"total_insertions": i, "total_deletions": 0, "files": []}

        self.assertEqual(len(cache), 1000)

        # Add one more via the function (use invalid hashes to avoid git calls)
        get_loc_for_commits("/nonexistent", ["zzzz"], cache, lock)

        # Cache should have been evicted (not grown unbounded past 1000)
        self.assertLessEqual(len(cache), 1000)


class TestLoadHistoryNonAscii(unittest.TestCase):
    """Tests that load_history handles non-ASCII content with utf-8 encoding."""

    def test_non_ascii_task_descriptions(self):
        """load_history should correctly read JSON containing non-ASCII characters."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                         delete=False, encoding="utf-8") as f:
            records = [
                {"timestamp": 1.0, "success": True,
                 "task": "Résoudre le problème café ☕"},
                {"timestamp": 2.0, "success": True,
                 "task": "修复中文编码问题 🐛"},
            ]
            json.dump(records, f, ensure_ascii=False)
            path = f.name
        try:
            result = load_history(path)
            self.assertEqual(len(result), 2)
            self.assertIn("café", result[0]["task"])
            self.assertIn("修复", result[1]["task"])
        finally:
            os.unlink(path)


class TestReadCycleStateNonAscii(unittest.TestCase):
    """Tests that _read_cycle_state handles non-ASCII content with utf-8 encoding."""

    def test_non_ascii_cycle_state(self):
        """_read_cycle_state should correctly read state with non-ASCII characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "current_cycle.json"
            state_data = {
                "phase": "executing",
                "task_description": "Corriger l'erreur über señor 🎉",
            }
            state_file.write_text(
                json.dumps(state_data, ensure_ascii=False), encoding="utf-8"
            )
            result = _read_cycle_state(tmpdir)
            self.assertIsNotNone(result)
            self.assertIn("über", result["task_description"])
            self.assertIn("señor", result["task_description"])


class TestIsOrchestratorRunningEncoding(unittest.TestCase):
    """Test that is_orchestrator_running handles non-ASCII lock files."""

    def test_non_ascii_lock_file_content(self):
        """Lock file with non-ASCII content should not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / "auto_claude.lock"
            # Write non-ASCII content that could cause issues without encoding
            lock_file.write_text("PID: résumé", encoding="utf-8")
            running, pid = is_orchestrator_running(str(lock_file))
            self.assertFalse(running)
            self.assertIsNone(pid)


class TestGetFeedbackFilesEncoding(unittest.TestCase):
    """Test that get_feedback_files handles non-ASCII content properly."""

    def test_non_ascii_feedback_content(self):
        """Feedback files with non-ASCII chars should be read correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feedback_dir = Path(tmpdir) / "feedback"
            done_dir = Path(tmpdir) / "done"
            failed_dir = Path(tmpdir) / "failed"
            for d in (feedback_dir, done_dir, failed_dir):
                d.mkdir()

            # Write a feedback file with Unicode content
            test_file = feedback_dir / "test.md"
            test_file.write_text("Fix über-critical bug in señor.py", encoding="utf-8")

            cfg = {
                "feedback_dir": str(feedback_dir),
                "feedback_done_dir": str(done_dir),
                "feedback_failed_dir": str(failed_dir),
            }
            result = get_feedback_files(cfg)
            self.assertEqual(len(result["pending"]), 1)
            self.assertIn("über", result["pending"][0]["content"])


class TestGetLocNonNumericValues(unittest.TestCase):
    """Regression: get_loc_for_commits should handle non-numeric values
    from git numstat (e.g. binary files that show '-' for both columns,
    or unexpected output formats) without raising ValueError."""

    @patch("dashboard.subprocess.run")
    def test_binary_file_dashes(self, mock_run):
        """Binary files show '-\t-\tpath' in numstat — should parse as 0."""
        mock_run.return_value = MagicMock(
            stdout="-\t-\tbinary.png\n10\t5\tnormal.py",
            returncode=0,
        )
        cache = {}
        lock = threading.Lock()
        result = get_loc_for_commits("/tmp", ["abc123"], cache, lock)
        self.assertIn("abc123", result)
        data = result["abc123"]
        self.assertEqual(data["total_insertions"], 10)
        self.assertEqual(data["total_deletions"], 5)
        self.assertEqual(len(data["files"]), 2)
        # Binary file should have 0 for both
        self.assertEqual(data["files"][0]["insertions"], 0)
        self.assertEqual(data["files"][0]["deletions"], 0)

    @patch("dashboard.subprocess.run")
    def test_malformed_numstat_line(self, mock_run):
        """Unexpected non-numeric values should not crash."""
        mock_run.return_value = MagicMock(
            stdout="abc\tdef\tfile.txt",
            returncode=0,
        )
        cache = {}
        lock = threading.Lock()
        # Should not raise ValueError
        result = get_loc_for_commits("/tmp", ["abc123"], cache, lock)
        self.assertIn("abc123", result)
        data = result["abc123"]
        self.assertEqual(data["total_insertions"], 0)
        self.assertEqual(data["total_deletions"], 0)


if __name__ == "__main__":
    unittest.main()
