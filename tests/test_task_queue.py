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
        result = _sanitize_filename("lint:foo.py")
        self.assertTrue(result.startswith("lint_foo.py_"))
        # Should contain an 8-char hex hash suffix
        self.assertRegex(result, r'^lint_foo\.py_[0-9a-f]{8}$')

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

    def test_approve_handles_disappearing_pending_file(self):
        """If pending file disappears between update and move, approve returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            task_id = self._enqueue_task(queue)
            pending_path = Path(tmpdir) / "pending_approval" / f"{task_id}.json"

            original_replace = os.replace
            call_count = {"n": 0}

            def intercepting_replace(src, dst):
                call_count["n"] += 1
                if call_count["n"] == 2:
                    # Delete the pending file before the second os.replace
                    try:
                        os.unlink(src)
                    except OSError:
                        pass
                    raise FileNotFoundError(f"No such file: {src}")
                return original_replace(src, dst)

            with patch("task_queue.os.replace", side_effect=intercepting_replace):
                result = queue.approve(task_id)

            self.assertFalse(result)

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


class TestDeclinedKeysCleanup(unittest.TestCase):
    """Verify _declined_keys doesn't grow unboundedly."""

    def test_old_declined_keys_cleaned_up(self):
        """Declined keys older than 24h should be evicted on next decline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            # Simulate 50 old declined entries (>24h ago)
            old_time = time.time() - 90000  # 25 hours ago
            for i in range(50):
                queue._declined_keys[f"old_key_{i}"] = old_time

            # Now decline a new task
            task = Task(description="New task", priority=2, source="lint")
            queue.enqueue(task)
            pending = queue.list_pending()
            queue.decline(pending[0]["id"])

            # Old keys should have been cleaned up
            old_remaining = sum(
                1 for k in queue._declined_keys if k.startswith("old_key_")
            )
            self.assertEqual(old_remaining, 0)

    def test_recent_declined_keys_preserved(self):
        """Declined keys younger than 24h should be kept."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            # Add a recent declined entry
            queue._declined_keys["recent_key"] = time.time() - 3600  # 1h ago

            # Decline another task
            task = Task(description="Another task", priority=2, source="lint")
            queue.enqueue(task)
            pending = queue.list_pending()
            queue.decline(pending[0]["id"])

            # Recent key should still exist
            self.assertIn("recent_key", queue._declined_keys)


class TestDeclinedKeysCleanupOnEnqueue(unittest.TestCase):
    """Verify stale _declined_keys are cleaned up during enqueue, not only decline."""

    def test_stale_keys_evicted_on_enqueue(self):
        """Old declined keys should be evicted when enqueue() is called."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            # Add stale entries (> 24 hours old)
            old_time = time.time() - 90000
            for i in range(20):
                queue._declined_keys[f"stale_{i}"] = old_time

            # Enqueue a new task (not one of the stale keys)
            task = Task(description="New fresh task", priority=2, source="lint")
            queue.enqueue(task)

            # All stale keys should have been evicted
            stale_remaining = sum(
                1 for k in queue._declined_keys if k.startswith("stale_")
            )
            self.assertEqual(stale_remaining, 0)

    def test_recent_keys_preserved_on_enqueue(self):
        """Recent declined keys should survive enqueue eviction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            queue._declined_keys["recent"] = time.time() - 3600  # 1 hour ago

            task = Task(description="Another task", priority=2, source="lint")
            queue.enqueue(task)

            self.assertIn("recent", queue._declined_keys)


class TestDeclinedKeysThreadSafety(unittest.TestCase):
    """Verify _declined_keys is protected by a lock."""

    def test_declined_lock_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            self.assertTrue(hasattr(queue, '_declined_lock'))

    def test_concurrent_enqueue_decline(self):
        """Enqueue and decline from multiple threads without errors."""
        import threading

        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            errors = []

            def enqueue_task(i):
                try:
                    task = Task(description=f"Task {i}", priority=2, source="lint")
                    queue.enqueue(task, cooldown_seconds=1)
                except Exception as e:
                    errors.append(e)

            def decline_task(task_id):
                try:
                    queue.decline(task_id)
                except Exception as e:
                    errors.append(e)

            # Enqueue several tasks
            for i in range(10):
                task = Task(description=f"Task {i}", priority=2, source="lint")
                queue.enqueue(task)

            pending = queue.list_pending()
            threads = []
            for p in pending:
                t = threading.Thread(target=decline_task, args=(p["id"],))
                threads.append(t)
            for i in range(10, 20):
                t = threading.Thread(target=enqueue_task, args=(i,))
                threads.append(t)

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errors, [])


class TestSanitizeFilenameCollisions(unittest.TestCase):
    """Test that _sanitize_filename produces distinct names for distinct keys."""

    def test_different_punctuation_produces_different_filenames(self):
        """Keys that differ only in non-alnum chars should NOT collide."""
        name1 = _sanitize_filename("fix bug: foo")
        name2 = _sanitize_filename("fix bug; foo")
        self.assertNotEqual(name1, name2)

    def test_identical_keys_produce_same_filename(self):
        name1 = _sanitize_filename("fix_test_foo")
        name2 = _sanitize_filename("fix_test_foo")
        self.assertEqual(name1, name2)

    def test_long_key_truncated_with_hash(self):
        long_key = "a" * 200
        name = _sanitize_filename(long_key)
        self.assertLessEqual(len(name), 120)

    def test_long_key_preserves_hash_suffix(self):
        """Truncated filenames must preserve the hash suffix to prevent collisions."""
        long_key = "a" * 200
        name = _sanitize_filename(long_key)
        # The name should end with _<8-char-hex>
        self.assertRegex(name, r'_[0-9a-f]{8}$')

    def test_two_long_keys_different_hashes(self):
        """Two different long keys that truncate identically must have different hashes."""
        key1 = "a" * 200
        key2 = "a" * 200 + "b"
        name1 = _sanitize_filename(key1)
        name2 = _sanitize_filename(key2)
        self.assertNotEqual(name1, name2)
        self.assertLessEqual(len(name1), 120)
        self.assertLessEqual(len(name2), 120)


class TestPathTraversalValidation(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.queue = TaskApprovalQueue(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_approve_rejects_dotdot(self):
        self.assertFalse(self.queue.approve("../../etc/passwd"))

    def test_approve_rejects_slash(self):
        self.assertFalse(self.queue.approve("foo/bar"))

    def test_approve_rejects_backslash(self):
        self.assertFalse(self.queue.approve("foo\\bar"))

    def test_approve_rejects_empty(self):
        self.assertFalse(self.queue.approve(""))

    def test_approve_rejects_dotfile(self):
        self.assertFalse(self.queue.approve(".hidden"))

    def test_decline_rejects_dotdot(self):
        self.assertFalse(self.queue.decline("../../etc/shadow"))

    def test_decline_rejects_slash(self):
        self.assertFalse(self.queue.decline("foo/bar"))

    def test_decline_rejects_empty(self):
        self.assertFalse(self.queue.decline(""))

    def test_valid_task_id_accepted(self):
        self.assertTrue(TaskApprovalQueue._is_valid_task_id("fix_bug_abc123"))

    def test_valid_task_id_with_hyphens(self):
        self.assertTrue(TaskApprovalQueue._is_valid_task_id("lint_foo-py_abcd1234"))


class TestTaskQueueEncoding(unittest.TestCase):
    """Verify read_text calls use explicit UTF-8 encoding."""

    def test_enqueue_and_list_unicode_task(self):
        """Task descriptions with non-ASCII chars should round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            task = Task(
                description="Fix résumé parsing in café.py",
                priority=2,
                source="lint",
            )
            result = queue.enqueue(task)
            self.assertIsNotNone(result)
            pending = queue.list_pending()
            self.assertEqual(len(pending), 1)
            self.assertIn("résumé", pending[0]["description"])

    def test_approve_and_consume_unicode_task(self):
        """Approved tasks with non-ASCII chars should be consumable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = TaskApprovalQueue(tmpdir)
            task = Task(
                description="Améliorer le café.py",
                priority=1,
                source="todo",
            )
            queue.enqueue(task)
            pending = queue.list_pending()
            self.assertTrue(queue.approve(pending[0]["id"]))
            approved = queue.get_approved()
            self.assertEqual(len(approved), 1)
            self.assertIn("Améliorer", approved[0].description)


if __name__ == "__main__":
    unittest.main()
