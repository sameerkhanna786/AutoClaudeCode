"""Invoke the Claude CLI and parse JSON responses."""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config_schema import Config
from process_utils import kill_process_group

logger = logging.getLogger(__name__)


@dataclass
class ClaudeResult:
    success: bool
    result_text: str = ""
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    raw_json: Optional[Dict[str, Any]] = None
    error: str = ""


class ClaudeRunner:

    def __init__(self, config: Config):
        self.config = config
        self.max_retries = config.claude.max_retries
        self.retry_delays = config.claude.retry_delays
        self.rate_limit_base_delay = config.claude.rate_limit_base_delay
        self.rate_limit_multiplier = config.claude.rate_limit_multiplier
        self._current_process: subprocess.Popen | None = None
        self._process_lock = threading.Lock()

    def _build_command(self, prompt: str) -> List[str]:
        """Build the CLI command list."""
        cc = self.config.claude
        model = cc.resolved_model or cc.model
        cmd = [
            cc.command,
            "-p", prompt,
            "--model", model,
            "--max-turns", str(cc.max_turns),
            "--output-format", "json",
        ]
        return cmd

    @staticmethod
    def _kill_process(proc: subprocess.Popen) -> None:
        """Kill a subprocess and its entire process group."""
        kill_process_group(proc)

    def terminate(self) -> None:
        """Terminate any currently running Claude subprocess.

        Thread-safe: can be called from a different thread than the one
        executing run().
        """
        with self._process_lock:
            proc = self._current_process
        if proc is not None:
            logger.warning("Terminating running Claude subprocess (pid=%s)", proc.pid)
            self._kill_process(proc)

    def _parse_json_response(self, stdout: str) -> Dict[str, Any]:
        """Parse JSON from Claude CLI output.

        The CLI may print banner/info lines before the actual JSON,
        and log/warning lines after it. We extract only the first
        valid top-level JSON object.
        """
        if not stdout or not stdout.strip():
            raise ValueError("Claude CLI produced empty output (no JSON to parse)")

        # Strategy 1a: Try each line as a complete JSON object starting with '{'.
        # This handles the common case where the JSON is on its own line.
        for line in stdout.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

        # Strategy 1b: Try finding JSON mid-line (preceded by text on the same line).
        for line in stdout.splitlines():
            line = line.strip()
            brace_pos = line.find("{")
            if brace_pos <= 0:
                continue
            # Try from each '{' position after the start of the line
            while brace_pos != -1:
                candidate = line[brace_pos:]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    pass
                brace_pos = line.find("{", brace_pos + 1)

        # Strategy 2: Fall back to raw_decode for multi-line JSON.
        # Try parsing from any '{' position to handle JSON starting mid-line.
        decoder = json.JSONDecoder()
        for i, ch in enumerate(stdout):
            if ch != "{":
                continue
            try:
                obj, end = decoder.raw_decode(stdout, i)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

        raise ValueError("No JSON object found in Claude CLI output")

    def run(self, prompt: str,
            add_dirs: Optional[List[str]] = None) -> ClaudeResult:
        """Run Claude CLI with the given prompt and return parsed result."""
        cmd = self._build_command(prompt)
        # Always use the main project dir as cwd (macOS sandbox restriction:
        # sandbox_apply fails with exit 71 when cwd is outside the project).
        cwd = self.config.target_dir
        if add_dirs:
            for d in add_dirs:
                cmd.extend(["--add-dir", str(Path(d).resolve())])

        logger.info("Running Claude CLI in %s", cwd)
        logger.debug("Command: %s", " ".join(cmd))

        for attempt in range(self.max_retries + 1):
            try:
                popen_proc = subprocess.Popen(
                    cmd,
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                with self._process_lock:
                    self._current_process = popen_proc
                try:
                    stdout, stderr = popen_proc.communicate(
                        timeout=self.config.claude.timeout_seconds,
                    )
                except subprocess.TimeoutExpired:
                    self._kill_process(popen_proc)
                    raise
                finally:
                    with self._process_lock:
                        self._current_process = None

                # Build a CompletedProcess-like namespace for downstream code
                class _ProcResult:
                    pass
                proc = _ProcResult()
                proc.returncode = popen_proc.returncode
                proc.stdout = stdout
                proc.stderr = stderr
            except subprocess.TimeoutExpired:
                if attempt < self.max_retries:
                    delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    logger.warning(
                        "Claude CLI timed out (attempt %d/%d), retrying in %ds",
                        attempt + 1, self.max_retries + 1, delay,
                    )
                    time.sleep(delay)
                    continue
                return ClaudeResult(
                    success=False,
                    error=f"Claude CLI timed out after {self.config.claude.timeout_seconds}s",
                )
            except FileNotFoundError:
                return ClaudeResult(
                    success=False,
                    error=f"Claude CLI command not found: {self.config.claude.command}",
                )
            except OSError as e:
                if attempt < self.max_retries:
                    delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    logger.warning(
                        "Claude CLI OS error (attempt %d/%d): %s, retrying in %ds",
                        attempt + 1, self.max_retries + 1, e, delay,
                    )
                    time.sleep(delay)
                    continue
                return ClaudeResult(
                    success=False,
                    error=f"Failed to run Claude CLI: {e}",
                )

            if proc.returncode != 0:
                if attempt < self.max_retries:
                    stderr_lower = proc.stderr.lower()
                    if "rate limit" in stderr_lower or "429" in stderr_lower or "too many requests" in stderr_lower:
                        delay = self.rate_limit_base_delay * (self.rate_limit_multiplier ** attempt)
                        logger.warning(
                            "Rate limited (attempt %d/%d), backing off %ds",
                            attempt + 1, self.max_retries + 1, delay,
                        )
                    else:
                        delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                        logger.warning(
                            "Claude CLI exited with code %d (attempt %d/%d), retrying in %ds",
                            proc.returncode, attempt + 1, self.max_retries + 1, delay,
                        )
                    time.sleep(delay)
                    continue
                return ClaudeResult(
                    success=False,
                    error=f"Claude CLI exited with code {proc.returncode}: {proc.stderr.strip()}",
                )

            # Parse JSON before exiting the loop — retry on truncated output
            try:
                data = self._parse_json_response(proc.stdout)
            except (ValueError, json.JSONDecodeError) as e:
                stdout_stripped = proc.stdout.strip()
                looks_truncated = "{" in stdout_stripped and not stdout_stripped.endswith("}")
                if looks_truncated and attempt < self.max_retries:
                    delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    logger.warning(
                        "Output appears truncated (attempt %d/%d), retrying in %ds",
                        attempt + 1, self.max_retries + 1, delay,
                    )
                    time.sleep(delay)
                    continue
                return ClaudeResult(
                    success=False,
                    error=f"Failed to parse Claude CLI output: {e}",
                    result_text=proc.stdout,
                )

            break  # successful parse — exit retry loop

        result_text = data.get("result", "")
        cost_usd = data.get("total_cost_usd", data.get("cost_usd", 0.0))
        duration_ms = data.get("duration_ms", 0)
        duration = duration_ms / 1000.0 if duration_ms else data.get("duration_seconds", 0.0)

        return ClaudeResult(
            success=True,
            result_text=result_text,
            cost_usd=cost_usd,
            duration_seconds=duration,
            raw_json=data,
        )
