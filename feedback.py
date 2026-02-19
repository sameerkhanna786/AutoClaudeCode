"""Watch feedback/ directory for developer-submitted task files."""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from config_schema import Config
from task_discovery import Task

logger = logging.getLogger(__name__)


class FeedbackManager:
    def __init__(self, config: Config):
        self.config = config
        self.feedback_dir = Path(config.paths.feedback_dir)
        self.done_dir = Path(config.paths.feedback_done_dir)
        self.failed_dir = Path(config.paths.feedback_failed_dir)
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.done_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)

    def _atomic_move(self, src: Path, dst: Path) -> None:
        """Move src to dst atomically using write-then-rename to prevent corruption."""
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(dst.parent), suffix=".tmp"
        )
        try:
            content = src.read_text()
            try:
                f = os.fdopen(tmp_fd, "w")
            except Exception:
                os.close(tmp_fd)
                raise
            with f:
                f.write(content)
            os.replace(tmp_path, str(dst))
            src.unlink()
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def get_pending_feedback(self) -> List[Task]:
        """Read pending feedback files and return them as Tasks.

        Files are sorted by name so developers can prefix with numbers
        to control priority (e.g., "01-fix-bug.md" before "02-add-feature.md").
        """
        tasks = []

        if not self.feedback_dir.exists():
            return tasks

        files = sorted(
            f for f in self.feedback_dir.iterdir()
            if f.is_file() and f.suffix in (".md", ".txt") and f.name != ".gitkeep"
        )

        for fpath in files:
            try:
                content = fpath.read_text().strip()
            except OSError as e:
                logger.warning("Failed to read feedback file %s: %s", fpath, e)
                continue

            if not content:
                continue

            # Extract priority from filename prefix (e.g., "01-fix-bug.md" → priority 1)
            priority = self._extract_priority(fpath.name)

            tasks.append(Task(
                description=content,
                priority=priority,
                source="feedback",
                source_file=str(fpath),
            ))

        return tasks

    def _extract_priority(self, filename: str) -> int:
        """Extract priority from filename prefix number. Default is 1."""
        match = re.match(r"^(\d+)", filename)
        if match:
            return max(1, int(match.group(1)))
        return 1

    def mark_done(self, source_file: str) -> None:
        """Move a processed feedback file to the done/ directory."""
        src = Path(source_file)
        if not src.exists():
            return

        dst = self.done_dir / src.name
        # Avoid overwriting existing done files
        if dst.exists():
            stem = dst.stem
            suffix = dst.suffix
            counter = 1
            while dst.exists():
                dst = self.done_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        try:
            self._atomic_move(src, dst)
        except OSError as e:
            logger.warning("Failed to move %s to %s: %s", src, dst, e)
            return
        logger.info("Marked feedback as done: %s → %s", src.name, dst.name)

    def mark_failed(self, source_file: str) -> None:
        """Move a feedback file to the failed/ directory after exceeding retries."""
        src = Path(source_file)
        if not src.exists():
            return

        dst = self.failed_dir / src.name
        # Avoid overwriting existing failed files
        if dst.exists():
            stem = dst.stem
            suffix = dst.suffix
            counter = 1
            while dst.exists():
                dst = self.failed_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        try:
            self._atomic_move(src, dst)
        except OSError as e:
            logger.warning("Failed to move %s to %s: %s", src, dst, e)
            return
        logger.info("Marked feedback as failed: %s → %s", src.name, dst.name)
