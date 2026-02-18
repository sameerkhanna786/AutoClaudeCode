"""Invoke the Claude CLI and parse JSON responses."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config_schema import Config

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

    def _build_command(self, prompt: str) -> List[str]:
        """Build the CLI command list."""
        cc = self.config.claude
        cmd = [
            cc.command,
            "-p", prompt,
            "--model", cc.model,
            "--max-turns", str(cc.max_turns),
            "--output-format", "json",
        ]
        return cmd

    def _parse_json_response(self, stdout: str) -> Dict[str, Any]:
        """Parse JSON from Claude CLI output.

        The CLI may print banner/info lines before the actual JSON,
        and log/warning lines after it. We extract only the first
        valid top-level JSON object.
        """
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

    def run(self, prompt: str, working_dir: Optional[str] = None) -> ClaudeResult:
        """Run Claude CLI with the given prompt and return parsed result."""
        cmd = self._build_command(prompt)
        cwd = working_dir or self.config.target_dir

        logger.info("Running Claude CLI in %s", cwd)
        logger.debug("Command: %s", " ".join(cmd))

        for attempt in range(self.max_retries + 1):
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=self.config.claude.timeout_seconds,
                )
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

            break

        try:
            data = self._parse_json_response(proc.stdout)
        except (ValueError, json.JSONDecodeError) as e:
            return ClaudeResult(
                success=False,
                error=f"Failed to parse Claude CLI output: {e}",
                result_text=proc.stdout,
            )

        result_text = data.get("result", "")
        cost_usd = data.get("cost_usd", 0.0)
        duration = data.get("duration_seconds", 0.0)

        return ClaudeResult(
            success=True,
            result_text=result_text,
            cost_usd=cost_usd,
            duration_seconds=duration,
            raw_json=data,
        )
