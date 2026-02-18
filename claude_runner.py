"""Invoke the Claude CLI and parse JSON responses."""

from __future__ import annotations

import json
import logging
import subprocess
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
        balanced top-level JSON object.
        """
        start = stdout.find("{")
        if start == -1:
            raise ValueError("No JSON object found in Claude CLI output")

        # Walk the string to find the matching closing brace,
        # respecting string literals so inner braces are ignored.
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(stdout)):
            ch = stdout[i]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(stdout[start:i + 1])

        raise ValueError("No complete JSON object found in Claude CLI output")

    def run(self, prompt: str, working_dir: Optional[str] = None) -> ClaudeResult:
        """Run Claude CLI with the given prompt and return parsed result."""
        cmd = self._build_command(prompt)
        cwd = working_dir or self.config.target_dir

        logger.info("Running Claude CLI in %s", cwd)
        logger.debug("Command: %s", " ".join(cmd))

        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.config.claude.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
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
            return ClaudeResult(
                success=False,
                error=f"Failed to run Claude CLI: {e}",
            )

        if proc.returncode != 0:
            return ClaudeResult(
                success=False,
                error=f"Claude CLI exited with code {proc.returncode}: {proc.stderr.strip()}",
            )

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
