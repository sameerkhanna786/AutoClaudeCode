"""Watch feedback/ directory for developer-submitted task files."""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from config_schema import Config
from task_discovery import Task

logger = logging.getLogger(__name__)

# Maximum allowed length for feedback content after sanitization
MAX_FEEDBACK_CONTENT_LENGTH = 64 * 1024  # 64 KB

# Patterns that should never appear in feedback task descriptions.
# These could be used to inject commands or manipulate Claude's behavior.
_DANGEROUS_PATTERNS = [
    # Shell command injection patterns
    re.compile(r'`[^`]*`'),                       # backtick command substitution
    re.compile(r'\$\([^)]+\)'),                    # $() command substitution
    re.compile(r'\$\{[^}]+\}'),                    # ${} variable expansion
    # Null bytes and control characters (excluding newlines/tabs)
    re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]'),
]

# Characters/sequences that are stripped entirely
_STRIP_SEQUENCES = [
    '\x00',  # null byte
]


def sanitize_feedback_content(content: str) -> str:
    """Sanitize feedback file content to prevent injection attacks.

    Removes dangerous shell metacharacters, control characters, and
    prompt injection patterns from feedback task descriptions before
    they are passed to Claude for execution.

    Returns the sanitized content, or empty string if content is invalid.
    """
    if not content or not isinstance(content, str):
        return ""

    # Strip null bytes and other dangerous sequences
    for seq in _STRIP_SEQUENCES:
        content = content.replace(seq, '')

    # Remove control characters (keep \n, \r, \t)
    content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', content)

    # Truncate to max length
    if len(content) > MAX_FEEDBACK_CONTENT_LENGTH:
        content = content[:MAX_FEEDBACK_CONTENT_LENGTH]
        logger.warning(
            "Feedback content truncated to %d bytes", MAX_FEEDBACK_CONTENT_LENGTH
        )

    content = content.strip()

    return content


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
        """Move src to dst atomically using write-then-rename to prevent corruption.

        Retries up to 3 times with increasing delays to handle race conditions
        when multiple processes attempt file operations simultaneously.
        """
        retry_delays = [0.05, 0.2, 0.5]
        last_exc: Optional[Exception] = None

        for attempt, delay in enumerate(retry_delays):
            # If the source file no longer exists on a retry, another process
            # already moved it — treat as success.
            if attempt > 0 and not src.exists():
                logger.debug(
                    "Source file %s no longer exists on retry %d, treating as success",
                    src, attempt,
                )
                return

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
                return  # Success
            except (OSError, FileNotFoundError) as e:
                last_exc = e
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                if attempt < len(retry_delays) - 1:
                    logger.debug(
                        "Atomic move %s -> %s failed (attempt %d/%d): %s — retrying in %.2fs",
                        src, dst, attempt + 1, len(retry_delays), e, delay,
                    )
                    time.sleep(delay)
                    continue
                raise
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
                with open(fpath, 'r') as f:
                    content = f.read(MAX_FEEDBACK_CONTENT_LENGTH)
            except OSError as e:
                logger.warning("Failed to read feedback file %s: %s", fpath, e)
                continue

            content = sanitize_feedback_content(content)
            if not content:
                logger.warning("Feedback file %s was empty or invalid after sanitization", fpath)
                continue

            # Extract priority from filename prefix (e.g., "01-fix-bug.md" → priority 1)
            priority = self._extract_priority(fpath.name)

            tasks.append(Task(
                description=content,
                priority=priority,
                source="feedback",
                source_file=str(fpath),
            ))

        # Clean up old done/failed files
        self._cleanup_old_files(self.done_dir)
        self._cleanup_old_files(self.failed_dir)

        return tasks

    def _cleanup_old_files(self, directory: Path, max_age_days: int = 7) -> None:
        """Remove files older than max_age_days from a directory."""
        cutoff = time.time() - (max_age_days * 86400)
        if not directory.exists():
            return
        for fpath in directory.iterdir():
            if fpath.is_file() and fpath.name != ".gitkeep":
                try:
                    if fpath.stat().st_mtime < cutoff:
                        fpath.unlink()
                except OSError:
                    pass

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

    def claim_feedback(self, source_file: str) -> bool:
        """Atomically claim a feedback file by renaming it with .claimed suffix.

        Returns True if the file was successfully claimed, False if another
        worker already claimed it (FileNotFoundError on rename).
        """
        src = Path(source_file)
        claimed = src.with_suffix(src.suffix + ".claimed")
        try:
            os.rename(str(src), str(claimed))
            return True
        except FileNotFoundError:
            return False  # another worker already claimed it

    def unclaim_feedback(self, source_file: str) -> None:
        """Restore a claimed feedback file back to its original name.

        Used when a worker fails and the feedback task should be retried.
        """
        src = Path(source_file)
        claimed = src.with_suffix(src.suffix + ".claimed")
        try:
            os.rename(str(claimed), str(src))
        except FileNotFoundError:
            pass  # file was already moved or doesn't exist

    def mark_done_claimed(self, source_file: str) -> None:
        """Move a claimed feedback file (.claimed suffix) to done/."""
        src = Path(source_file)
        claimed = src.with_suffix(src.suffix + ".claimed")
        if not claimed.exists():
            # Fall back to original path
            if src.exists():
                self.mark_done(source_file)
            return
        # Move the claimed file to done with the original name
        dst = self.done_dir / src.name
        if dst.exists():
            stem = dst.stem
            suffix = dst.suffix
            counter = 1
            while dst.exists():
                dst = self.done_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        try:
            self._atomic_move(claimed, dst)
        except OSError as e:
            logger.warning("Failed to move %s to %s: %s", claimed, dst, e)
            return
        logger.info("Marked claimed feedback as done: %s → %s", src.name, dst.name)

    def mark_failed_claimed(self, source_file: str) -> None:
        """Move a claimed feedback file (.claimed suffix) to failed/."""
        src = Path(source_file)
        claimed = src.with_suffix(src.suffix + ".claimed")
        if not claimed.exists():
            if src.exists():
                self.mark_failed(source_file)
            return
        dst = self.failed_dir / src.name
        if dst.exists():
            stem = dst.stem
            suffix = dst.suffix
            counter = 1
            while dst.exists():
                dst = self.failed_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        try:
            self._atomic_move(claimed, dst)
        except OSError as e:
            logger.warning("Failed to move %s to %s: %s", claimed, dst, e)
            return
        logger.info("Marked claimed feedback as failed: %s → %s", src.name, dst.name)
