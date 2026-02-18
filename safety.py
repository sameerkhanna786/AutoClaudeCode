"""Safety guards: lock file, disk space, rate limits, protected files, failure counter."""

from __future__ import annotations

import fcntl
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

from config_schema import Config
from state import StateManager

logger = logging.getLogger(__name__)


class SafetyError(Exception):
    """Raised when a safety check fails."""
    pass


class SafetyGuard:
    def __init__(self, config: Config, state_manager: StateManager):
        self.config = config
        self.state = state_manager
        self._lock_fd: Optional[int] = None
        self.lock_path = Path(config.paths.lock_file)

    def acquire_lock(self) -> None:
        """Acquire an exclusive file lock to prevent concurrent runs."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # Check if the PID in the lock file is still alive
            stale = False
            existing_pid_str = ""
            try:
                os.lseek(self._lock_fd, 0, os.SEEK_SET)
                existing_pid_bytes = os.read(self._lock_fd, 64)
                existing_pid_str = existing_pid_bytes.decode(errors="replace").strip()
                existing_pid = int(existing_pid_str)
                os.kill(existing_pid, 0)
            except (ValueError, ProcessLookupError):
                stale = True
            except PermissionError:
                # Process exists but we can't signal it
                stale = False

            os.close(self._lock_fd)
            self._lock_fd = None

            if stale:
                logger.warning(
                    "Cleaning up stale lock file from dead process (PID %s)",
                    existing_pid_str,
                )
                self.lock_path.unlink(missing_ok=True)
                # Retry acquisition
                self._lock_fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR)
                try:
                    fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    os.close(self._lock_fd)
                    self._lock_fd = None
                    raise SafetyError("Another instance is already running (lock file held)")
            else:
                raise SafetyError("Another instance is already running (lock file held)")

        # Write our PID
        os.ftruncate(self._lock_fd, 0)
        os.lseek(self._lock_fd, 0, os.SEEK_SET)
        os.write(self._lock_fd, str(os.getpid()).encode())

    def release_lock(self) -> None:
        """Release the file lock."""
        if self._lock_fd is not None:
            try:
                os.close(self._lock_fd)
            except OSError:
                pass
            self._lock_fd = None

    def check_disk_space(self) -> None:
        """Ensure sufficient disk space is available."""
        target = self.config.target_dir
        usage = shutil.disk_usage(target)
        free_mb = usage.free / (1024 * 1024)
        if free_mb < self.config.safety.min_disk_space_mb:
            raise SafetyError(
                f"Low disk space: {free_mb:.0f} MB free, "
                f"minimum {self.config.safety.min_disk_space_mb} MB required"
            )

    def check_rate_limit(self) -> None:
        """Ensure we haven't exceeded the cycles-per-hour limit."""
        count = self.state.get_cycle_count_last_hour()
        limit = self.config.safety.max_cycles_per_hour
        if count >= limit:
            raise SafetyError(
                f"Rate limit reached: {count} cycles in the last hour (limit: {limit})"
            )

    def check_cost_limit(self) -> None:
        """Ensure we haven't exceeded the cost-per-hour limit."""
        cost = self.state.get_total_cost(lookback_seconds=3600)
        limit = self.config.safety.max_cost_usd_per_hour
        if cost >= limit:
            raise SafetyError(
                f"Cost limit reached: ${cost:.2f} in the last hour (limit: ${limit:.2f})"
            )

    def check_consecutive_failures(self) -> None:
        """Pause if too many consecutive failures."""
        failures = self.state.get_consecutive_failures()
        limit = self.config.safety.max_consecutive_failures
        if failures >= limit:
            raise SafetyError(
                f"Too many consecutive failures: {failures} (limit: {limit}). "
                "Pausing until a successful cycle or manual intervention."
            )

    def check_protected_files(self, changed_files: List[str]) -> None:
        """Ensure no protected files have been modified."""
        target_dir = self.config.target_dir
        violations = []
        for f in changed_files:
            changed_path = os.path.join(target_dir, f)
            for p in self.config.safety.protected_files:
                protected_path = os.path.join(target_dir, p)
                # Use samefile when both paths exist (avoids unnecessary exceptions)
                if os.path.exists(changed_path) and os.path.exists(protected_path):
                    try:
                        if os.path.samefile(changed_path, protected_path):
                            violations.append(f)
                            break
                    except OSError:
                        pass
                    else:
                        # samefile returned False — paths are definitively different
                        continue
                # Fall back to realpath + normpath comparison (e.g. when file doesn't exist yet)
                if os.path.normpath(os.path.realpath(changed_path)) == os.path.normpath(os.path.realpath(protected_path)):
                    violations.append(f)
                    break
        if violations:
            raise SafetyError(
                f"Protected files modified: {', '.join(violations)}"
            )

    def check_file_count(self, changed_files: List[str]) -> None:
        """Ensure number of changed files is within limit."""
        limit = self.config.orchestrator.max_changed_files
        if limit <= 0:
            raise SafetyError(
                f"max_changed_files must be positive (got {limit})"
            )
        count = len(changed_files)
        if count > limit:
            raise SafetyError(
                f"Too many files changed: {count} (limit: {limit})"
            )
        if count > limit * 0.8:
            logger.warning("Changed file count (%d) approaching limit (%d)", count, limit)

    def pre_flight_checks(self) -> None:
        """Run all pre-cycle safety checks."""
        self.check_disk_space()
        self.check_rate_limit()
        self.check_cost_limit()
        self.check_consecutive_failures()

    def post_claude_checks(self, changed_files: List[str]) -> None:
        """Run safety checks after Claude has made changes."""
        self.check_protected_files(changed_files)
        self.check_file_count(changed_files)
