"""Tests for task_queue.py — Task approval queue manager."""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from task_discovery import Task
from task_queue import TaskApprovalQueue, _sanitize_filename


class TestSanitizeFilename(unittest.TestCase):

    def test_simple_key(self):
        self.assertEqual(_sanitize_filename("lint:foo.py"), "lint_foo.py")

    def test_special_chars(self):
        result = _sanitize_filename("claude_idea:Add tests for bar/baz.py")
        self.assertNotIn("/", result)
        self.assertNotIn(" ", result)

    def test_long_key_truncated(self):
        key = "a" * 200
        result = _sanitize_filename(key)
        self.assertLessEqual(len(result), 120)


class TestHeartbeat(unittest.TestCase):

    def test_update_and_check(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            self.assertFalse(queue.is_dashboard_active())
            queue.update_heartbeat()
            self.assertTrue(queue.is_dashboard_active())

    def test_expired_heartbeat(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            queue.update_heartbeat()
            # Manually write an old timestamp
            hb_path = Path(tmpdir) / "dashboard_heartbeat.json"
            hb_path.write_text(json.dumps({"timestamp": time.time() - 60}))
            self.assertFalse(queue.is_dashboard_active(timeout=30))

    def test_missing_heartbeat_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            self.assertFalse(queue.is_dashboard_active())

    def test_corrupt_heartbeat_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            hb_path = Path(tmpdir) / "dashboard_heartbeat.json"
            hb_path.write_text("{invalid json")
            self.assertFalse(queue.is_dashboard_active())


class TestEnqueue(unittest.TestCase):

    def test_enqueue_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            task = Task(description="Fix lint in foo.py", priority=2, source="lint")
            result = queue.enqueue(task)
            self.assertIsNotNone(result)
            pending = queue.list_pending()
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0]["description"], "Fix lint in foo.py")

    def test_enqueue_duplicate_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            task = Task(description="Fix lint in foo.py", priority=2, source="lint")
            queue.enqueue(task)
            result = queue.enqueue(task)  # duplicate
            self.assertIsNone(result)
            self.assertEqual(len(queue.list_pending()), 1)

    def test_enqueue_recently_declined_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            task = Task(description="Fix lint in foo.py", priority=2, source="lint")
            queue.enqueue(task)
            queue.decline(queue.list_pending()[0]["id"])
            # Re-enqueue with cooldown should be skipped
            result = queue.enqueue(task, cooldown_seconds=3600)
            self.assertIsNone(result)

    def test_enqueue_declined_after_cooldown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            task = Task(description="Fix lint in foo.py", priority=2, source="lint")
            queue.enqueue(task)
            # Decline it
            pending = queue.list_pending()
            queue.decline(pending[0]["id"])
            # Manually set declined time to past
            queue._declined_keys[task.task_key] = time.time() - 7200
            # Re-enqueue with shorter cooldown should succeed
            result = queue.enqueue(task, cooldown_seconds=3600)
            self.assertIsNotNone(result)


class TestApproveDecline(unittest.TestCase):

    def _enqueue_task(self, queue, desc="Test task", source="lint"):
        task = Task(description=desc, priority=2, source=source)
        queue.enqueue(task)
        return queue.list_pending()[0]["id"]

    def test_approve_moves_to_approved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            task_id = self._enqueue_task(queue)
            self.assertTrue(queue.approve(task_id))
            self.assertEqual(len(queue.list_pending()), 0)
            # Check approved file exists
            approved_dir = Path(tmpdir) / "approved"
            self.assertTrue(any(approved_dir.iterdir()))

    def test_approve_nonexistent_returns_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            self.assertFalse(queue.approve("nonexistent"))

    def test_decline_deletes_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            task_id = self._enqueue_task(queue)
            self.assertTrue(queue.decline(task_id))
            self.assertEqual(len(queue.list_pending()), 0)

    def test_decline_nonexistent_returns_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            self.assertFalse(queue.decline("nonexistent"))


class TestApproveAtomicity(unittest.TestCase):
    """Tests verifying the atomic rename-based approve implementation."""

    def _enqueue_task(self, queue, desc="Test task", source="lint"):
        task = Task(description=desc, priority=2, source=source)
        queue.enqueue(task)
        return queue.list_pending()[0]["id"]

    def test_approve_sets_approved_at(self):
        """Approved file must contain an approved_at timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            task_id = self._enqueue_task(queue)
            queue.approve(task_id)
            approved_path = Path(tmpdir) / "approved" / f"{task_id}.json"
            data = json.loads(approved_path.read_text())
            self.assertIn("approved_at", data)
            self.assertIsInstance(data["approved_at"], float)

    def test_approve_no_ghost_duplicate(self):
        """After approve, only approved/ has the file — pending/ must be empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            task_id = self._enqueue_task(queue)
            pending_path = Path(tmpdir) / "pending_approval" / f"{task_id}.json"
            approved_path = Path(tmpdir) / "approved" / f"{task_id}.json"
            self.assertTrue(pending_path.exists())
            self.assertFalse(approved_path.exists())

            queue.approve(task_id)

            self.assertFalse(pending_path.exists())
            self.assertTrue(approved_path.exists())

    def test_approve_preserves_pending_on_temp_write_failure(self):
        """If the temp-file write fails, the pending file must remain intact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            task_id = self._enqueue_task(queue)
            pending_path = Path(tmpdir) / "pending_approval" / f"{task_id}.json"
            original_content = pending_path.read_text()

            with patch("task_queue.tempfile.mkstemp", side_effect=OSError("disk full")):
                result = queue.approve(task_id)

            self.assertFalse(result)
            # Pending file must still exist with original content
            self.assertTrue(pending_path.exists())
            self.assertEqual(pending_path.read_text(), original_content)
            # No approved file should have been created
            approved_path = Path(tmpdir) / "approved" / f"{task_id}.json"
            self.assertFalse(approved_path.exists())

    def test_approve_data_integrity(self):
        """Approved file must contain all original task fields plus approved_at."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            task = Task(description="Integrity check", priority=3, source="test",
                        source_file="check.py", context="ctx")
            queue.enqueue(task)
            task_id = queue.list_pending()[0]["id"]
            queue.approve(task_id)

            approved_path = Path(tmpdir) / "approved" / f"{task_id}.json"
            data = json.loads(approved_path.read_text())
            self.assertEqual(data["description"], "Integrity check")
            self.assertEqual(data["priority"], 3)
            self.assertEqual(data["source"], "test")
            self.assertEqual(data["source_file"], "check.py")
            self.assertEqual(data["context"], "ctx")
            self.assertIn("approved_at", data)


class TestBulkOperations(unittest.TestCase):

    def test_approve_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            for i in range(3):
                task = Task(description=f"Task {i}", priority=2, source="lint",
                           source_file=f"file{i}.py")
                queue.enqueue(task)
            count = queue.approve_all()
            self.assertEqual(count, 3)
            self.assertEqual(len(queue.list_pending()), 0)

    def test_decline_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            for i in range(3):
                task = Task(description=f"Task {i}", priority=2, source="lint",
                           source_file=f"file{i}.py")
                queue.enqueue(task)
            count = queue.decline_all()
            self.assertEqual(count, 3)
            self.assertEqual(len(queue.list_pending()), 0)


class TestGetApproved(unittest.TestCase):

    def test_get_approved_returns_tasks_and_consumes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            task = Task(description="Fix lint in foo.py", priority=2, source="lint")
            queue.enqueue(task)
            task_id = queue.list_pending()[0]["id"]
            queue.approve(task_id)

            approved = queue.get_approved()
            self.assertEqual(len(approved), 1)
            self.assertEqual(approved[0].description, "Fix lint in foo.py")
            self.assertEqual(approved[0].priority, 2)
            self.assertEqual(approved[0].source, "lint")

            # Consumed — second call returns empty
            self.assertEqual(len(queue.get_approved()), 0)

    def test_get_approved_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            self.assertEqual(queue.get_approved(), [])


class TestPendingCount(unittest.TestCase):

    def test_count_matches_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            self.assertEqual(queue.pending_count(), 0)
            for i in range(3):
                task = Task(description=f"Task {i}", priority=2, source="lint",
                           source_file=f"file{i}.py")
                queue.enqueue(task)
            self.assertEqual(queue.pending_count(), 3)


class TestClearStale(unittest.TestCase):

    def test_clear_stale_removes_old_tasks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            task = Task(description="Old task", priority=2, source="lint")
            queue.enqueue(task)

            # Manually set enqueued_at to past
            pending_dir = Path(tmpdir) / "pending_approval"
            for f in pending_dir.iterdir():
                if f.suffix == ".json":
                    data = json.loads(f.read_text())
                    data["enqueued_at"] = time.time() - 7200
                    f.write_text(json.dumps(data))

            removed = queue.clear_stale(max_age=3600)
            self.assertEqual(removed, 1)
            self.assertEqual(queue.pending_count(), 0)

    def test_clear_stale_keeps_recent_tasks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            task = Task(description="Recent task", priority=2, source="lint")
            queue.enqueue(task)
            removed = queue.clear_stale(max_age=3600)
            self.assertEqual(removed, 0)
            self.assertEqual(queue.pending_count(), 1)


if __name__ == "__main__":
    unittest.main()
