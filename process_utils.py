"""Shared utilities for process-group-aware subprocess management.

Provides helpers to kill an entire process group on timeout, preventing
orphaned grandchild processes when using shell=True or subprocess pipelines.
"""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class RunResult:
    """Result of run_with_group_kill(), mimicking subprocess.CompletedProcess."""
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


def kill_process_group(proc: subprocess.Popen) -> None:
    """Kill a subprocess and its entire process group.

    Sends SIGKILL to the process group, then falls back to proc.kill()
    in case the process group ID lookup failed, and finally waits briefly
    for the process to be reaped.
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=5)
    except (subprocess.TimeoutExpired, OSError):
        pass


def run_with_group_kill(
    command,
    *,
    shell: bool = False,
    cwd: Optional[str] = None,
    timeout: Optional[int] = None,
    text: bool = True,
) -> RunResult:
    """Run a command, killing its entire process group on timeout.

    Uses start_new_session=True so that all child processes are in a
    dedicated process group.  On timeout, kills the entire group via
    os.killpg() to prevent orphaned grandchildren.

    Returns a RunResult with stdout, stderr, returncode, and a timed_out flag.
    """
    proc = subprocess.Popen(
        command,
        shell=shell,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return RunResult(
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=False,
        )
    except subprocess.TimeoutExpired:
        kill_process_group(proc)
        # Capture any partial output produced before the timeout
        partial_stdout = ""
        partial_stderr = ""
        try:
            remaining_out, remaining_err = proc.communicate(timeout=5)
            partial_stdout = remaining_out or ""
            partial_stderr = remaining_err or ""
        except (subprocess.TimeoutExpired, OSError, ValueError):
            # Process may already be dead or pipes closed
            for stream_name, stream in [("stdout", proc.stdout), ("stderr", proc.stderr)]:
                if stream is not None:
                    try:
                        data = stream.read()
                        if data:
                            if stream_name == "stdout":
                                partial_stdout = data
                            else:
                                partial_stderr = data
                    except (OSError, ValueError):
                        pass
        timeout_val = timeout if timeout is not None else "unknown"
        prefix = f"[TIMEOUT after {timeout_val}s] "
        return RunResult(
            returncode=-1,
            stdout=prefix + partial_stdout if partial_stdout else prefix,
            stderr=prefix + partial_stderr if partial_stderr else "",
            timed_out=True,
        )
