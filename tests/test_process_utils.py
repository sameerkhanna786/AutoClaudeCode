"""Tests for process_utils.py — process group management and subprocess execution."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pytest

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

    @pytest.mark.requires_subprocess
    def test_success(self):
        result = run_with_group_kill(["echo", "hello"])
        self.assertEqual(result.returncode, 0)
        self.assertIn("hello", result.stdout)
        self.assertFalse(result.timed_out)

    @pytest.mark.requires_subprocess
    def test_failure(self):
        result = run_with_group_kill(["false"])
        self.assertEqual(result.returncode, 1)
        self.assertFalse(result.timed_out)

    @pytest.mark.requires_subprocess
    def test_timeout(self):
        result = run_with_group_kill(["sleep", "60"], timeout=1)
        self.assertEqual(result.returncode, -1)
        self.assertTrue(result.timed_out)
        self.assertTrue(result.stdout.startswith("[TIMEOUT after 1s]"))

    @pytest.mark.requires_subprocess
    def test_timeout_stderr_has_prefix(self):
        """Both stdout and stderr should contain the timeout prefix on timeout."""
        result = run_with_group_kill(["sleep", "60"], timeout=1)
        self.assertTrue(result.timed_out)
        self.assertIn("[TIMEOUT after 1s]", result.stdout)
        self.assertIn("[TIMEOUT after 1s]", result.stderr)

    @pytest.mark.requires_subprocess
    def test_shell_mode(self):
        result = run_with_group_kill("echo hello && echo world", shell=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("hello", result.stdout)
        self.assertIn("world", result.stdout)

    @pytest.mark.requires_subprocess
    def test_cwd(self):
        result = run_with_group_kill(["pwd"], cwd="/tmp")
        self.assertEqual(result.returncode, 0)
        # On macOS, /tmp may be symlinked to /private/tmp
        self.assertTrue(
            "/tmp" in result.stdout or "/private/tmp" in result.stdout,
            f"Expected /tmp or /private/tmp in stdout, got: {result.stdout!r}",
        )

    @pytest.mark.requires_subprocess
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


class TestRunWithGroupKillBaseException(unittest.TestCase):
    """Tests that BaseException between Popen and communicate kills the process."""

    @patch("process_utils.subprocess.Popen")
    @patch("process_utils.kill_process_group")
    def test_keyboard_interrupt_kills_process(self, mock_kill, mock_popen):
        """KeyboardInterrupt during communicate() must kill the process group."""
        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = KeyboardInterrupt
        mock_popen.return_value = mock_proc

        with self.assertRaises(KeyboardInterrupt):
            run_with_group_kill(["sleep", "60"])

        mock_kill.assert_called_once_with(mock_proc)

    @patch("process_utils.subprocess.Popen")
    @patch("process_utils.kill_process_group")
    def test_system_exit_kills_process(self, mock_kill, mock_popen):
        """SystemExit during communicate() must kill the process group."""
        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = SystemExit(1)
        mock_popen.return_value = mock_proc

        with self.assertRaises(SystemExit):
            run_with_group_kill(["sleep", "60"])

        mock_kill.assert_called_once_with(mock_proc)


class TestPopenFailure(unittest.TestCase):
    """Tests that Popen failures (FileNotFoundError, OSError) are handled gracefully."""

    def test_command_not_found_returns_runresult(self):
        """FileNotFoundError from Popen should return RunResult, not raise."""
        result = run_with_group_kill(["nonexistent_command_xyz_12345"])
        self.assertEqual(result.returncode, -1)
        self.assertFalse(result.timed_out)
        self.assertIn("Failed to start process", result.stderr)

    def test_bad_cwd_returns_runresult(self):
        """OSError from invalid cwd should return RunResult, not raise."""
        result = run_with_group_kill(["echo", "hello"], cwd="/nonexistent/path/xyz")
        self.assertEqual(result.returncode, -1)
        self.assertFalse(result.timed_out)
        self.assertIn("Failed to start process", result.stderr)


class TestTimeoutPartialReadBounded(unittest.TestCase):
    """stream.read() after timeout must be bounded to prevent OOM."""

    @patch("process_utils.subprocess.Popen")
    @patch("process_utils.kill_process_group")
    def test_stream_read_called_with_size_limit(self, mock_kill, mock_popen):
        """After timeout, fallback stream.read() should pass a max size argument."""
        import subprocess as _subprocess

        mock_proc = MagicMock()
        # First communicate() raises timeout
        mock_proc.communicate.side_effect = [
            _subprocess.TimeoutExpired(cmd="test", timeout=1),
            _subprocess.TimeoutExpired(cmd="test", timeout=5),
        ]
        mock_stdout = MagicMock()
        mock_stdout.read.return_value = "partial out"
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = "partial err"
        mock_proc.stdout = mock_stdout
        mock_proc.stderr = mock_stderr
        mock_popen.return_value = mock_proc

        result = run_with_group_kill(["test"], timeout=1)

        self.assertTrue(result.timed_out)
        # Verify read was called with a size limit (1 MB)
        mock_stdout.read.assert_called_once_with(1024 * 1024)
        mock_stderr.read.assert_called_once_with(1024 * 1024)


class TestKillProcessGroup(unittest.TestCase):

    def test_already_dead_process(self):
        """Killing a process group with an invalid PID should not raise."""
        mock_proc = MagicMock()
        mock_proc.pid = 999999999  # unlikely to exist
        mock_proc.kill.side_effect = OSError("No such process")
        mock_proc.wait.side_effect = OSError("No child processes")

        # Should not raise
        kill_process_group(mock_proc)


class TestOutputTruncation(unittest.TestCase):
    """Happy-path output must be truncated to prevent OOM on huge output."""

    @patch("process_utils.subprocess.Popen")
    def test_stdout_truncated_to_1mb(self, mock_popen):
        """stdout exceeding 1 MB should be truncated in the happy path."""
        mock_proc = MagicMock()
        big_output = "x" * (2 * 1024 * 1024)  # 2 MB
        mock_proc.communicate.return_value = (big_output, "err")
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        result = run_with_group_kill(["echo", "test"])
        self.assertEqual(len(result.stdout), 1024 * 1024)
        self.assertEqual(result.stderr, "err")

    @patch("process_utils.subprocess.Popen")
    def test_stderr_truncated_to_1mb(self, mock_popen):
        """stderr exceeding 1 MB should be truncated in the happy path."""
        mock_proc = MagicMock()
        big_err = "e" * (2 * 1024 * 1024)  # 2 MB
        mock_proc.communicate.return_value = ("out", big_err)
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        result = run_with_group_kill(["echo", "test"])
        self.assertEqual(result.stdout, "out")
        self.assertEqual(len(result.stderr), 1024 * 1024)

    @patch("process_utils.subprocess.Popen")
    def test_small_output_not_truncated(self, mock_popen):
        """Output under 1 MB should not be truncated."""
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("small", "errs")
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        result = run_with_group_kill(["echo", "test"])
        self.assertEqual(result.stdout, "small")
        self.assertEqual(result.stderr, "errs")


if __name__ == "__main__":
    unittest.main()
