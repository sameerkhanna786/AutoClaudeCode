"""Resolve Claude model aliases to actual model IDs via CLI probe."""

import json
import logging
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


def resolve_model_id(
    model_alias: str = "opus",
    claude_command: str = "claude",
    timeout: int = 30,
) -> Optional[str]:
    """Resolve a model alias to its actual model ID via a minimal CLI call.

    Runs: claude -p "x" --model <alias> --output-format json --max-turns 1 --tools ""
    Parses the modelUsage key from the JSON response.

    Returns the resolved model ID (e.g., "claude-opus-4-6") or None on failure.
    """
    cmd = [
        claude_command, "-p", "x",
        "--model", model_alias,
        "--output-format", "json",
        "--max-turns", "1",
        "--tools", "",
    ]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.warning("Model resolution failed (CLI error): %s", e)
        return None

    if proc.returncode != 0:
        logger.warning("Model resolution failed (exit code %d): %s",
                        proc.returncode, proc.stderr.strip()[:200])
        return None

    # Parse JSON from output (may have banner lines before it)
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
            if isinstance(data, dict) and "modelUsage" in data:
                model_usage = data["modelUsage"]
                if isinstance(model_usage, dict) and model_usage:
                    resolved = next(iter(model_usage))
                    logger.info("Resolved model '%s' -> '%s'", model_alias, resolved)
                    return resolved
        except json.JSONDecodeError:
            continue

    logger.warning("Model resolution failed: no modelUsage in CLI output")
    return None
