#!/usr/bin/env python3
"""Auto Claude Code — Entry point and outer watchdog.

PROTECTED FILE — This file must never be modified by Claude.

Two-layer safety:
1. This file dynamically imports orchestrator modules.
2. If an import fails (e.g., Claude broke a module), we run
   `git checkout .` to restore and retry once.
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import subprocess
import sys
import time
from pathlib import Path


def setup_logging(log_file: str, level: str, max_bytes: int, backup_count: int) -> None:
    """Configure logging with both console and rotating file handler."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler (rotating)
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def restore_and_retry(repo_dir: str) -> None:
    """Run git checkout . && git clean -fd to restore broken files."""
    logger = logging.getLogger("watchdog")
    logger.warning("Attempting to restore broken files via git checkout...")
    try:
        subprocess.run(
            ["git", "checkout", "."],
            cwd=repo_dir,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "clean", "-fd"],
            cwd=repo_dir,
            capture_output=True,
            check=True,
        )
        logger.info("Files restored successfully")
    except subprocess.CalledProcessError as e:
        logger.error("Failed to restore files: %s", e)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto Claude Code — Autonomous development system"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single cycle and exit",
    )
    args = parser.parse_args()

    # Determine our own directory (for git restore if needed)
    own_dir = str(Path(__file__).resolve().parent)

    # First attempt: import and run
    for attempt in range(2):
        try:
            from config_schema import load_config
            from orchestrator import Orchestrator

            config = load_config(args.config)

            # Setup logging from config
            setup_logging(
                log_file=config.logging.file,
                level=config.logging.level,
                max_bytes=config.logging.max_bytes,
                backup_count=config.logging.backup_count,
            )

            logger = logging.getLogger("main")
            logger.info("Auto Claude Code starting (attempt %d)", attempt + 1)

            orchestrator = Orchestrator(config)
            orchestrator.run(once=args.once)
            return

        except ImportError as e:
            print(f"[watchdog] Import error (attempt {attempt + 1}): {e}", file=sys.stderr)
            if attempt == 0:
                restore_and_retry(own_dir)
                # Clear cached modules so re-import works
                mods_to_remove = [
                    m for m in sys.modules
                    if m in (
                        "config_schema", "orchestrator", "claude_runner",
                        "git_manager", "validator", "state", "safety",
                        "task_discovery", "feedback",
                    )
                ]
                for m in mods_to_remove:
                    del sys.modules[m]
            else:
                print("[watchdog] Failed after restore attempt. Exiting.", file=sys.stderr)
                sys.exit(1)

        except Exception as e:
            print(f"[watchdog] Fatal error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
