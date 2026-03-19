"""Watch feedback/ directory for developer-submitted task files."""

from __future__ import annotations

import errno
import logging
import os
import random
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from config_schema import Config
from task_discovery import Task

logger = logging.getLogger(__name__)

# Regex to strip control characters (keep \n, \r, \t)
_CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

# Maximum allowed length for feedback content after sanitization
MAX_FEEDBACK_CONTENT_LENGTH = 64 * 1024  # 64 KB

# Patterns that should never appear in feedback task descriptions.
# These could be used to inject commands or manipulate Claude's behavior.
_DANGEROUS_PATTERNS = [
    # Shell command injection patterns
    re.compile(r'\$\([^)]+\)'),                    # $() command substitution
    re.compile(r'\$\{[^}]+\}'),                    # ${} variable expansion
    re.compile(r'`[^`]+`'),                        # backtick command substitution
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
    content = _CONTROL_CHAR_RE.sub('', content)

    # Check for dangerous patterns (shell injection, command substitution).
    # Loop until no patterns match — nested patterns like $($(cmd)) survive
    # a single pass because stripping the outer $() reveals the inner one.
    max_passes = 10  # safety limit to avoid infinite loop on pathological input
    for _pass in range(max_passes):
        found = False
        for i, pattern in enumerate(_DANGEROUS_PATTERNS):
            if pattern.search(content):
                logger.warning(
                    "Dangerous pattern detected in feedback content (pattern %d, pass %d)",
                    i, _pass,
                )
                content = pattern.sub('', content)
                found = True
        if not found:
            break

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
        self._last_cleanup_time = 0.0
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.done_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)

    def _atomic_move(self, src: Path, dst: Path) -> None:
        """Move src to dst atomically using write-then-rename to prevent corruption.

        Uses progressive exponential backoff with jitter to reduce contention
        when parallel workers compete for the same feedback files.
        Retries up to 5 times with delays: ~0.05s, ~0.15s, ~0.45s, ~1.35s, ~4.05s.
        """
        max_retries = 5
        base_delay = 0.05
        backoff_multiplier = 3.0
        jitter_factor = 0.25  # ±25% randomized jitter
        last_exc: Optional[Exception] = None

        # Read source content once before the retry loop — the content
        # doesn't change between retries, so re-reading is wasted I/O.
        # Use O_NOFOLLOW to refuse symlinks (TOCTOU defense).
        try:
            fd = os.open(str(src), os.O_RDONLY | os.O_NOFOLLOW)
            try:
                raw = os.read(fd, MAX_FEEDBACK_CONTENT_LENGTH + 1)
            finally:
                os.close(fd)
            content = raw.decode('utf-8', errors='replace')
        except FileNotFoundError:
            logger.debug("Source file %s already moved by another process", src)
            return
        except OSError as e:
            # O_NOFOLLOW raises ELOOP (errno 40 on macOS, 62 on Linux) for symlinks
            if e.errno in (errno.ELOOP, getattr(errno, 'EMLINK', -1)):
                logger.warning("Refusing to follow symlink: %s", src)
                return
            raise

        for attempt in range(max_retries):
            # If the source file no longer exists on a retry, another process
            # already moved it — treat as success.
            if attempt > 0 and not src.exists():
                logger.debug(
                    "Source file %s no longer exists on retry %d, treating as success",
                    src, attempt,
                )
                return

            # Calculate delay with exponential backoff + jitter
            if attempt > 0:
                delay = base_delay * (backoff_multiplier ** (attempt - 1))
                jitter = delay * jitter_factor * (2 * random.random() - 1)
                actual_delay = max(0, delay + jitter)
                logger.debug(
                    "Atomic move %s -> %s failed (attempt %d/%d): %s — retrying in %.3fs",
                    src, dst, attempt, max_retries, last_exc, actual_delay,
                )
                time.sleep(actual_delay)

            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(dst.parent), suffix=".tmp"
            )
            try:
                os.fchmod(tmp_fd, 0o600)
                try:
                    f = os.fdopen(tmp_fd, "w", encoding="utf-8")
                except Exception:
                    os.close(tmp_fd)
                    raise
                with f:
                    f.write(content)
                os.replace(tmp_path, str(dst))
                try:
                    src.unlink()
                except FileNotFoundError:
                    pass  # src already deleted by another process; move succeeded
                return  # Success
            except (OSError, FileNotFoundError) as e:
                last_exc = e
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                if attempt < max_retries - 1:
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

        # Single pass over the directory to partition into feedback and PRD files
        files = []
        prd_files = []
        try:
            all_entries = sorted(self.feedback_dir.iterdir(), key=lambda f: f.name)
        except OSError:
            return tasks
        for f in all_entries:
            if not f.is_file() or f.name == ".gitkeep":
                continue
            if f.is_symlink():
                logger.warning("Skipping symlink in feedback directory: %s", f)
                continue
            if (f.name.endswith(".prd.yaml") or f.name.endswith(".prd.json")
                    or f.name.endswith(".prd.yml")):
                prd_files.append(f)
            elif f.suffix in (".md", ".txt"):
                files.append(f)
        for prd_file in prd_files:
            try:
                from prd_generator import import_prd
                prd_tasks = import_prd(str(prd_file))
                tasks.extend(prd_tasks)
                # Move processed PRD to done
                self.mark_done(str(prd_file))
            except Exception as e:
                logger.warning("Failed to import PRD %s: %s", prd_file, e)

        for fpath in files:
            try:
                # Use O_NOFOLLOW to atomically reject symlinks at the kernel
                # level, preventing TOCTOU races between the is_symlink()
                # check above and the actual file open.
                fd = os.open(str(fpath), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(fd, 'r', encoding='utf-8') as f:
                    content = f.read(MAX_FEEDBACK_CONTENT_LENGTH)
            except UnicodeDecodeError:
                logger.warning(
                    "Feedback file %s is not valid UTF-8, attempting lossy decode",
                    fpath,
                )
                try:
                    raw = fpath.read_bytes()[:MAX_FEEDBACK_CONTENT_LENGTH]
                    content = raw.decode('utf-8', errors='replace')
                except OSError as e:
                    logger.warning("Failed to read feedback file %s: %s", fpath, e)
                    continue
            except OSError as e:
                logger.warning("Failed to read feedback file %s: %s", fpath, e)
                continue

            # Warn if file was truncated
            try:
                file_size = fpath.stat().st_size
                if file_size > MAX_FEEDBACK_CONTENT_LENGTH:
                    logger.warning(
                        "Feedback file %s truncated: %d bytes on disk, "
                        "read limit is %d bytes",
                        fpath, file_size, MAX_FEEDBACK_CONTENT_LENGTH,
                    )
            except OSError:
                pass

            content = sanitize_feedback_content(content)
            if not content:
                logger.warning("Feedback file %s was empty or invalid after sanitization", fpath)
                continue

            # Parse YAML frontmatter for task_id and depends_on
            task_id = ""
            depends_on: List[str] = []
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = parts[1].strip()
                    content_body = parts[2].strip()
                    for fm_line in frontmatter.split("\n"):
                        fm_line = fm_line.strip()
                        if fm_line.startswith("task_id:"):
                            task_id = fm_line[len("task_id:"):].strip().strip('"').strip("'")
                        elif fm_line.startswith("depends_on:"):
                            deps_str = fm_line[len("depends_on:"):].strip()
                            if deps_str.startswith("[") and deps_str.endswith("]"):
                                deps_str = deps_str[1:-1]
                            depends_on = [
                                d.strip().strip('"').strip("'")
                                for d in deps_str.split(",") if d.strip()
                            ]
                    if content_body:
                        content = content_body

            # Extract priority from filename prefix (e.g., "01-fix-bug.md" → priority 1)
            priority = self._extract_priority(fpath.name)

            task = Task(
                description=content,
                priority=priority,
                source="feedback",
                source_file=str(fpath),
            )
            if task_id:
                task.task_id = task_id
            if depends_on:
                task.depends_on = depends_on

            tasks.append(task)

        # Clean up old done/failed files and stale claims (at most once per hour)
        now = time.time()
        if now - self._last_cleanup_time > 3600:
            self._cleanup_old_files(self.done_dir)
            self._cleanup_old_files(self.failed_dir)
            self._cleanup_stale_claims()
            self._last_cleanup_time = now

        return tasks

    def _cleanup_old_files(self, directory: Path, max_age_days: int = 7) -> None:
        """Remove files older than max_age_days from a directory."""
        cutoff = time.time() - (max_age_days * 86400)
        if not directory.exists():
            return
        try:
            entries = list(directory.iterdir())
        except OSError:
            return
        for fpath in entries:
            if fpath.is_file() and fpath.name != ".gitkeep":
                try:
                    if fpath.stat().st_mtime < cutoff:
                        fpath.unlink()
                except OSError:
                    pass

    def _cleanup_stale_claims(self, max_age_seconds: int = 3600) -> None:
        """Remove .claimed files older than max_age_seconds.

        When a worker crashes after claiming a feedback file, the .claimed
        file remains forever, preventing the task from being retried.
        """
        if not self.feedback_dir.exists():
            return
        cutoff = time.time() - max_age_seconds
        try:
            entries = list(self.feedback_dir.iterdir())
        except OSError:
            return
        for fpath in entries:
            if fpath.is_file() and fpath.name.endswith(".claimed"):
                try:
                    if fpath.stat().st_mtime < cutoff:
                        logger.warning("Removing stale claimed file: %s", fpath.name)
                        fpath.unlink()
                except OSError:
                    pass

    def _extract_priority(self, filename: str) -> int:
        """Extract priority from filename prefix number. Default is 1."""
        match = re.match(r"^(\d+)", filename)
        if match:
            return max(1, int(match.group(1)))
        return 1

    def _is_within_feedback_dir(self, path: Path) -> bool:
        """Check that a path is within the feedback directory tree.

        Rejects symlinks to prevent path traversal via symlink targets
        that resolve outside the feedback directory.

        Uses Path.relative_to() instead of str().startswith() to prevent
        bypasses where a sibling directory name shares a common prefix
        (e.g. /a/feedback vs /a/feedback_evil).
        """
        try:
            # Reject symlinks before resolving to prevent following malicious
            # targets. Checking is_symlink() first avoids the TOCTOU window
            # that exists when resolve() is called before the symlink check.
            if path.is_symlink():
                logger.warning("Rejecting symlink in feedback directory: %s", path)
                return False
            resolved = path.resolve()
            feedback_resolved = self.feedback_dir.resolve()
            resolved.relative_to(feedback_resolved)
            return True
        except (OSError, ValueError):
            return False

    def _unique_dst(self, directory: Path, name: str) -> Path:
        """Generate a unique destination path, avoiding overwrites."""
        dst = directory / name
        if not dst.exists():
            return dst
        stem = dst.stem
        suffix = dst.suffix
        counter = 1
        while dst.exists() and counter < 1000:
            dst = directory / f"{stem}_{counter}{suffix}"
            counter += 1
        if dst.exists():
            # All numbered slots exhausted — use a timestamp suffix to
            # guarantee uniqueness and prevent silent file overwrites.
            ts = int(time.time() * 1000)
            dst = directory / f"{stem}_{ts}{suffix}"
            if dst.exists():
                # Timestamp collision — append a random suffix to guarantee uniqueness.
                import random
                rand_suffix = random.randint(0, 999999)
                dst = directory / f"{stem}_{ts}_{rand_suffix}{suffix}"
            logger.warning(
                "All 1000 filename slots exhausted for %s in %s, using timestamp suffix",
                name, directory,
            )
        return dst

    def mark_done(self, source_file: str) -> None:
        """Move a processed feedback file to the done/ directory."""
        src = Path(source_file)
        if not src.exists():
            return

        if not self._is_within_feedback_dir(src):
            logger.warning("Rejecting mark_done for path outside feedback dir: %s", src)
            return

        dst = self._unique_dst(self.done_dir, src.name)

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

        if not self._is_within_feedback_dir(src):
            logger.warning("Rejecting mark_failed for path outside feedback dir: %s", src)
            return

        dst = self._unique_dst(self.failed_dir, src.name)

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
        except OSError as e:
            logger.warning("Failed to claim feedback %s: %s", source_file, e)
            return False

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

        if not self._is_within_feedback_dir(claimed):
            logger.warning("Rejecting mark_done_claimed for path outside feedback dir: %s", claimed)
            return
        # Move the claimed file to done with the original name
        dst = self._unique_dst(self.done_dir, src.name)
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

        if not self._is_within_feedback_dir(claimed):
            logger.warning("Rejecting mark_failed_claimed for path outside feedback dir: %s", claimed)
            return
        dst = self._unique_dst(self.failed_dir, src.name)
        try:
            self._atomic_move(claimed, dst)
        except OSError as e:
            logger.warning("Failed to move %s to %s: %s", claimed, dst, e)
            return
        logger.info("Marked claimed feedback as failed: %s → %s", src.name, dst.name)
