"""Tests for session_manager.py — session recovery and resume."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from session_manager import SessionManager, SessionState


class TestSessionState(unittest.TestCase):

    def test_round_trip(self):
        state = SessionState(
            session_id="test-123",
            started_at=1000.0,
            tasks=[{"task_id": "t1", "description": "Fix bug"}],
            completed_task_ids=["t1"],
            total_cost_usd=1.5,
            phase="validating",
        )
        d = state.to_dict()
        restored = SessionState.from_dict(d)
        self.assertEqual(restored.session_id, "test-123")
        self.assertEqual(restored.started_at, 1000.0)
        self.assertEqual(len(restored.tasks), 1)
        self.assertEqual(restored.completed_task_ids, ["t1"])
        self.assertAlmostEqual(restored.total_cost_usd, 1.5)
        self.assertEqual(restored.phase, "validating")

    def test_from_empty_dict(self):
        state = SessionState.from_dict({})
        self.assertEqual(state.session_id, "")
        self.assertEqual(state.tasks, [])
        self.assertEqual(state.phase, "starting")

    def test_worker_states_keys_become_int(self):
        data = {"session_id": "s1", "started_at": 0,
                "worker_states": {"0": {"status": "idle"}, "3": {"status": "busy"}}}
        state = SessionState.from_dict(data)
        self.assertEqual(set(state.worker_states.keys()), {0, 3})


class TestSessionManager(unittest.TestCase):

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            state = SessionState(
                session_id="s1",
                started_at=100.0,
                tasks=[{"task_id": "a", "description": "Do A"}],
                phase="executing",
            )
            mgr.save_session(state)
            loaded = mgr.load_session()
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.session_id, "s1")
            self.assertEqual(loaded.phase, "executing")

    def test_clear_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            state = SessionState(session_id="s1", started_at=100.0)
            mgr.save_session(state)
            self.assertTrue(mgr.has_incomplete_session() is not None)
            mgr.clear_session()
            loaded = mgr.load_session()
            self.assertIsNone(loaded)

    def test_has_incomplete_session_true(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            state = SessionState(
                session_id="s1",
                started_at=100.0,
                tasks=[
                    {"task_id": "a", "task_key": "a"},
                    {"task_id": "b", "task_key": "b"},
                ],
                completed_task_ids=["a"],
            )
            mgr.save_session(state)
            self.assertTrue(mgr.has_incomplete_session())

    def test_has_incomplete_session_false_all_complete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            state = SessionState(
                session_id="s1",
                started_at=100.0,
                tasks=[
                    {"task_id": "a", "task_key": "a"},
                ],
                completed_task_ids=["a"],
            )
            mgr.save_session(state)
            self.assertFalse(mgr.has_incomplete_session())

    def test_has_incomplete_session_false_no_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            self.assertFalse(mgr.has_incomplete_session())

    def test_has_incomplete_session_false_no_tasks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            state = SessionState(session_id="s1", started_at=100.0)
            mgr.save_session(state)
            self.assertFalse(mgr.has_incomplete_session())

    def test_load_corrupted_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            session_file = Path(tmpdir) / "session.json"
            session_file.write_text("not valid json{{{")
            loaded = mgr.load_session()
            self.assertIsNone(loaded)

    def test_load_non_dict_json_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            session_file = Path(tmpdir) / "session.json"
            session_file.write_text("[1, 2, 3]")
            loaded = mgr.load_session()
            self.assertIsNone(loaded)

    def test_save_overwrites_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            s1 = SessionState(session_id="old", started_at=1.0)
            mgr.save_session(s1)
            s2 = SessionState(session_id="new", started_at=2.0, phase="done")
            mgr.save_session(s2)
            loaded = mgr.load_session()
            self.assertEqual(loaded.session_id, "new")
            self.assertEqual(loaded.phase, "done")

    def test_clear_no_file_is_noop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            mgr.clear_session()  # should not raise

    def test_task_key_fallback_in_incomplete_check(self):
        """Tasks without task_id fall back to task_key for completeness check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            state = SessionState(
                session_id="s1",
                started_at=0,
                tasks=[{"task_key": "k1"}],
                completed_task_ids=["k1"],
            )
            mgr.save_session(state)
            self.assertFalse(mgr.has_incomplete_session())

    def test_create_session_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            sid = mgr.create_session_id()
            self.assertTrue(sid.startswith("session-"))
            parts = sid.split("-")
            self.assertEqual(len(parts), 3)
            self.assertTrue(parts[1].isdigit())
            self.assertTrue(parts[2].isdigit())

    def test_creates_nested_state_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "a" / "b" / "c"
            mgr = SessionManager(str(nested))
            self.assertTrue(nested.exists())


class TestOrphanedWorktrees(unittest.TestCase):

    @patch("subprocess.run")
    def test_detect_orphaned(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "worktree /tmp/main\n"
                "branch refs/heads/main\n"
                "\n"
                "worktree /tmp/.worktrees/worker-0\n"
                "branch refs/heads/auto-claude/123-0\n"
                "\n"
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            orphaned = mgr.recover_orphaned_worktrees("/tmp/main")
            self.assertEqual(len(orphaned), 1)
            self.assertEqual(orphaned[0]["branch"], "auto-claude/123-0")

    @patch("subprocess.run")
    def test_no_orphaned(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "worktree /tmp/main\n"
                "branch refs/heads/main\n"
                "\n"
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            orphaned = mgr.recover_orphaned_worktrees("/tmp/main")
            self.assertEqual(len(orphaned), 0)

    @patch("subprocess.run")
    def test_git_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            orphaned = mgr.recover_orphaned_worktrees("/tmp/main")
            self.assertEqual(len(orphaned), 0)

    @patch("subprocess.run")
    def test_timeout_returns_empty(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=30)
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            orphaned = mgr.recover_orphaned_worktrees("/tmp/main")
            self.assertEqual(len(orphaned), 0)

    @patch("subprocess.run")
    def test_os_error_returns_empty(self, mock_run):
        mock_run.side_effect = OSError("git not found")
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            orphaned = mgr.recover_orphaned_worktrees("/tmp/main")
            self.assertEqual(len(orphaned), 0)

    @patch("subprocess.run")
    def test_last_entry_no_trailing_newline(self, mock_run):
        """The last worktree entry may not have a trailing blank line."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "worktree /tmp/wt1\n"
                "branch refs/heads/auto-claude/task-x"
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            orphaned = mgr.recover_orphaned_worktrees("/tmp/main")
            self.assertEqual(len(orphaned), 1)
            self.assertEqual(orphaned[0]["branch"], "auto-claude/task-x")


class TestRecoveryFlow(unittest.TestCase):
    """Test the full recovery flow: detect stale session, recover, clear."""

    def test_full_recovery_cycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            # 1. Save session simulating a crash mid-cycle
            state = SessionState(
                session_id="crash-session",
                started_at=1000.0,
                tasks=[
                    {"task_id": "t1", "description": "Task 1"},
                    {"task_id": "t2", "description": "Task 2"},
                ],
                completed_task_ids=["t1"],
                phase="executing",
            )
            mgr.save_session(state)

            # 2. On restart, detect incomplete session
            self.assertTrue(mgr.has_incomplete_session())

            # 3. Load session to resume
            recovered = mgr.load_session()
            self.assertIsNotNone(recovered)
            self.assertEqual(recovered.session_id, "crash-session")
            self.assertEqual(recovered.completed_task_ids, ["t1"])

            # 4. After recovery completes, clear the session
            mgr.clear_session()
            self.assertFalse(mgr.has_incomplete_session())
            self.assertIsNone(mgr.load_session())


if __name__ == "__main__":
    unittest.main()
