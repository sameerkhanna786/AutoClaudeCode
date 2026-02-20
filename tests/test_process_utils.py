"""Tests for process_utils.py — process group management and subprocess execution."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from process_utils import RunResult, kill_process_group, run_with_group_kill


class TestRunResult(unittest.TestCase):

    def test_defaults(self):
        r = RunResult(returncode=0, stdout="out", stderr="err")
        self.assertEqual(r.returncode, 0)
        self.assertEqual(r.stdout, "out")
        self.assertEqual(r.stderr, "err")
        self.assertFalse(r.timed_out)

    def test_timed_out_flag(self):
        r = RunResult(returncode=-1, stdout="", stderr="", timed_out=True)
        self.assertTrue(r.timed_out)


class TestRunWithGroupKill(unittest.TestCase):

    def test_success(self):
        result = run_with_group_kill(["echo", "hello"])
        self.assertEqual(result.returncode, 0)
        self.assertIn("hello", result.stdout)
        self.assertFalse(result.timed_out)

    def test_failure(self):
        result = run_with_group_kill(["false"])
        self.assertEqual(result.returncode, 1)
        self.assertFalse(result.timed_out)

    def test_timeout(self):
        result = run_with_group_kill(["sleep", "60"], timeout=1)
        self.assertEqual(result.returncode, -1)
        self.assertTrue(result.timed_out)
        self.assertTrue(result.stdout.startswith("[TIMEOUT after 1s]"))

    def test_shell_mode(self):
        result = run_with_group_kill("echo hello && echo world", shell=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("hello", result.stdout)
        self.assertIn("world", result.stdout)

    def test_cwd(self):
        result = run_with_group_kill(["pwd"], cwd="/tmp")
        self.assertEqual(result.returncode, 0)
        # On macOS, /tmp may be symlinked to /private/tmp
        self.assertTrue(
            "/tmp" in result.stdout or "/private/tmp" in result.stdout,
            f"Expected /tmp or /private/tmp in stdout, got: {result.stdout!r}",
        )

    def test_stderr_captured(self):
        result = run_with_group_kill(
            [sys.executable, "-c", "import sys; sys.stderr.write('errtext\\n')"]
        )
        self.assertIn("errtext", result.stderr)

    @patch("process_utils.subprocess.Popen")
    def test_starts_new_session(self, mock_popen):
        # Set up mock to avoid actually running a process
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("out", "err")
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        run_with_group_kill(["echo", "test"])

        # Verify start_new_session=True was passed
        call_kwargs = mock_popen.call_args[1]
        self.assertTrue(call_kwargs.get("start_new_session", False))


class TestKillProcessGroup(unittest.TestCase):

    def test_already_dead_process(self):
        """Killing a process group with an invalid PID should not raise."""
        mock_proc = MagicMock()
        mock_proc.pid = 999999999  # unlikely to exist
        mock_proc.kill.side_effect = OSError("No such process")
        mock_proc.wait.side_effect = OSError("No child processes")

        # Should not raise
        kill_process_group(mock_proc)


if __name__ == "__main__":
    unittest.main()
