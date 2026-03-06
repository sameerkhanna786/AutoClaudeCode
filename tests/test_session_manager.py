"""Tests for session_manager.py — session recovery and resume."""

from __future__ import annotations

import json
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

    def test_create_session_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            sid = mgr.create_session_id()
            self.assertTrue(sid.startswith("session-"))


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


if __name__ == "__main__":
    unittest.main()
