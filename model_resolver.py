"""Resolve Claude model aliases to actual model IDs via CLI probe."""

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Cache TTL: 24 hours
_CACHE_TTL_SECONDS = 86400
_CACHE_FILE = "state/model_cache.json"


def _read_cache(model_alias: str, cache_path: str = _CACHE_FILE) -> Optional[str]:
    """Read a cached model resolution if valid and within TTL."""
    path = Path(cache_path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if (
            isinstance(data, dict)
            and data.get("alias") == model_alias
            and time.time() - data.get("timestamp", 0) < _CACHE_TTL_SECONDS
        ):
            resolved = data.get("resolved")
            if resolved:
                logger.info(
                    "Using cached model resolution: '%s' -> '%s'",
                    model_alias, resolved,
                )
                return resolved
    except (json.JSONDecodeError, OSError, KeyError):
        pass
    return None


def _write_cache(model_alias: str, resolved: str, cache_path: str = _CACHE_FILE) -> None:
    """Write a successful model resolution to the cache file."""
    path = Path(cache_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "alias": model_alias,
            "resolved": resolved,
            "timestamp": time.time(),
        }
        path.write_text(json.dumps(data))
    except OSError as e:
        logger.debug("Failed to write model cache: %s", e)


def resolve_model_id(
    model_alias: str = "opus",
    claude_command: str = "claude",
    timeout: int = 30,
) -> Optional[str]:
    """Resolve a model alias to its actual model ID via a minimal CLI call.

    Checks a local cache first (state/model_cache.json, TTL 24h) to avoid
    redundant API calls on restarts.

    Runs: claude -p "x" --model <alias> --output-format json --max-turns 1 --tools ""
    Parses the modelUsage key from the JSON response.

    Returns the resolved model ID (e.g., "claude-opus-4-6") or None on failure.
    """
    # Check cache first
    cached = _read_cache(model_alias)
    if cached:
        return cached

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
                    _write_cache(model_alias, resolved)
                    return resolved
        except json.JSONDecodeError:
            continue

    logger.warning("Model resolution failed: no modelUsage in CLI output")
    return None
