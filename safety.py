"""Safety guards: lock file, disk space, rate limits, protected files, failure counter."""

from __future__ import annotations

import atexit
import fcntl
import logging
import os
import shutil
import time
from pathlib import Path
from typing import List, Optional

from config_schema import Config
from state import StateManager

logger = logging.getLogger(__name__)

# Module-level list of SafetyGuard instances that hold locks, for atexit cleanup.
_active_guards: List[SafetyGuard] = []


def _atexit_release_locks() -> None:
    """Release all held locks on normal interpreter exit."""
    for guard in list(_active_guards):
        try:
            guard.release_lock()
        except Exception:
            pass


atexit.register(_atexit_release_locks)


class SafetyError(Exception):
    """Raised when a safety check fails."""
    pass


class GracefulDegradation:
    """Manages graceful degradation when approaching rate or cost limits.

    Instead of hard-stopping, this reduces batch size and increases sleep
    intervals to allow the system to continue operating at reduced capacity.
    """

    def __init__(self, config: Config):
        self.config = config
        self._degraded = False
        self._degradation_level = 0  # 0=normal, 1=mild, 2=moderate, 3=severe

    @property
    def is_degraded(self) -> bool:
        return self._degraded

    @property
    def degradation_level(self) -> int:
        return self._degradation_level

    def check_and_adjust(
        self,
        cycles_per_hour: int,
        cost_per_hour: float,
    ) -> dict:
        """Check proximity to limits and compute adjustments.

        Returns a dict with:
            - degraded: whether we're in degraded mode
            - level: 0-3 severity level
            - batch_size_factor: multiplier for batch size (0.25-1.0)
            - sleep_multiplier: multiplier for sleep interval (1.0-4.0)
            - reason: human-readable reason for degradation
        """
        rate_limit = self.config.safety.max_cycles_per_hour
        cost_limit = self.config.safety.max_cost_usd_per_hour

        rate_pct = (cycles_per_hour / rate_limit * 100) if rate_limit > 0 else 0
        cost_pct = (cost_per_hour / cost_limit * 100) if cost_limit > 0 else 0

        max_pct = max(rate_pct, cost_pct)

        reasons = []
        if rate_pct >= 70:
            reasons.append(f"rate at {rate_pct:.0f}% of limit ({cycles_per_hour}/{rate_limit})")
        if cost_pct >= 70:
            reasons.append(f"cost at {cost_pct:.0f}% of limit (${cost_per_hour:.2f}/${cost_limit:.2f})")

        if max_pct >= 95:
            self._degradation_level = 3
            self._degraded = True
            return {
                "degraded": True,
                "level": 3,
                "batch_size_factor": 0.25,
                "sleep_multiplier": 4.0,
                "reason": "Severe: " + "; ".join(reasons),
            }
        elif max_pct >= 85:
            self._degradation_level = 2
            self._degraded = True
            return {
                "degraded": True,
                "level": 2,
                "batch_size_factor": 0.5,
                "sleep_multiplier": 3.0,
                "reason": "Moderate: " + "; ".join(reasons),
            }
        elif max_pct >= 70:
            self._degradation_level = 1
            self._degraded = True
            return {
                "degraded": True,
                "level": 1,
                "batch_size_factor": 0.75,
                "sleep_multiplier": 2.0,
                "reason": "Mild: " + "; ".join(reasons),
            }
        else:
            self._degradation_level = 0
            self._degraded = False
            return {
                "degraded": False,
                "level": 0,
                "batch_size_factor": 1.0,
                "sleep_multiplier": 1.0,
                "reason": "",
            }


class SafetyGuard:
    def __init__(self, config: Config, state_manager: StateManager):
        self.config = config
        self.state = state_manager
        self._lock_fd: Optional[int] = None
        self.lock_path = Path(config.paths.lock_file)

    def acquire_lock(self) -> None:
        """Acquire an exclusive file lock to prevent concurrent runs.

        Uses flock() on the existing lock file inode, avoiding the race
        condition of unlink-then-recreate where two processes could each
        hold exclusive flocks on different inodes.
        """
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # Check if the PID in the lock file is still alive
            stale = False
            existing_pid_str = ""
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                existing_pid_bytes = os.read(fd, 64)
                existing_pid_str = existing_pid_bytes.decode(errors="replace").strip()
                existing_pid = int(existing_pid_str)
                os.kill(existing_pid, 0)
            except (ValueError, ProcessLookupError):
                stale = True
            except PermissionError:
                # Process exists but we can't signal it
                stale = False

            if not stale:
                os.close(fd)
                raise SafetyError("Another instance is already running (lock file held)")

            # Stale lock: the PID is dead. Retry flock on the SAME file
            # descriptor (same inode) — don't unlink and recreate.
            logger.warning(
                "Cleaning up stale lock file from dead process (PID %s)",
                existing_pid_str,
            )
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                raise SafetyError("Another instance is already running (lock file held)")

        # Lock acquired — store the fd and write our PID
        self._lock_fd = fd
        _active_guards.append(self)
        os.ftruncate(self._lock_fd, 0)
        os.lseek(self._lock_fd, 0, os.SEEK_SET)
        os.write(self._lock_fd, str(os.getpid()).encode())

    def release_lock(self) -> None:
        """Release the file lock."""
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(self._lock_fd)
            except OSError:
                pass
            self._lock_fd = None
            try:
                _active_guards.remove(self)
            except ValueError:
                pass

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

    def check_memory(self) -> None:
        """Ensure sufficient RAM is available.

        Uses /proc/meminfo on Linux and vm_stat on macOS.
        Skips the check gracefully on unsupported platforms.
        """
        import platform
        min_mb = self.config.safety.min_memory_mb
        if min_mb <= 0:
            return

        available_mb: Optional[float] = None

        system = platform.system()
        if system == "Linux":
            try:
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            parts = line.split()
                            available_mb = int(parts[1]) / 1024  # kB -> MB
                            break
            except (OSError, ValueError, IndexError):
                logger.debug("Could not read /proc/meminfo, skipping memory check")
                return
        elif system == "Darwin":
            try:
                import subprocess
                result = subprocess.run(
                    ["vm_stat"], capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    page_size = 4096  # default macOS page size
                    free_pages = 0
                    for line in result.stdout.splitlines():
                        if "page size of" in line:
                            try:
                                page_size = int(line.split()[-2])
                            except (ValueError, IndexError):
                                pass
                        elif "Pages free:" in line:
                            try:
                                free_pages += int(line.split()[-1].rstrip("."))
                            except (ValueError, IndexError):
                                pass
                        elif "Pages speculative:" in line or "Pages purgeable:" in line:
                            try:
                                free_pages += int(line.split()[-1].rstrip("."))
                            except (ValueError, IndexError):
                                pass
                    available_mb = (free_pages * page_size) / (1024 * 1024)
            except (OSError, subprocess.TimeoutExpired):
                logger.debug("Could not run vm_stat, skipping memory check")
                return
        else:
            logger.debug("Memory check not supported on %s", system)
            return

        if available_mb is None:
            logger.debug("Could not determine available memory, skipping check")
            return

        if available_mb < min_mb:
            raise SafetyError(
                f"Low memory: {available_mb:.0f} MB available, "
                f"minimum {min_mb} MB required"
            )
        elif available_mb < min_mb * 1.5:
            logger.warning(
                "Memory approaching minimum: %.0f MB available (minimum: %d MB)",
                available_mb, min_mb,
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
        """Pause if too many consecutive failures, with reset support."""
        failures = self.state.get_consecutive_failures()
        limit = self.config.safety.max_consecutive_failures
        if failures < limit:
            return

        # Check for file-based reset trigger
        reset_file = Path(self.config.paths.state_dir) / "reset_failures"
        if reset_file.exists():
            logger.info("Found reset_failures file, resetting consecutive failures")
            self.state.reset_consecutive_failures("file-based trigger (state/reset_failures)")
            reset_file.unlink(missing_ok=True)
            return

        # Check for time-based auto-reset (idle for 1+ hour)
        if self.state.should_auto_reset_failures(min_idle_seconds=3600):
            logger.info("System idle for 1+ hour, auto-resetting consecutive failures")
            self.state.reset_consecutive_failures("auto-reset after idle period")
            return

        raise SafetyError(
            f"Too many consecutive failures: {failures} (limit: {limit}). "
            "Pausing until a successful cycle or manual intervention. "
            "To reset: touch state/reset_failures"
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
            if limit == 0:
                raise SafetyError(
                    "max_changed_files is zero; no files can be changed"
                )
            raise SafetyError(
                f"max_changed_files must be positive (got {limit})"
            )
        count = len(changed_files)
        if count > limit:
            raise SafetyError(
                f"Too many files changed: {count} (limit: {limit})"
            )
        warning_threshold = int(limit * 0.8)
        if warning_threshold > 0 and count > warning_threshold:
            logger.warning("Changed file count (%d) approaching limit (%d)", count, limit)

    def check_backup_dir_size(self) -> None:
        """Monitor backup directory size and clean up old files when threshold is exceeded.

        Removes oldest backup files first until directory size is below threshold.
        This prevents unbounded growth from syntax-check backups and corruption backups.
        """
        backup_dir = Path(self.config.paths.backup_dir)
        if not backup_dir.exists():
            return

        max_mb = self.config.safety.max_backup_dir_mb
        if max_mb <= 0:
            return

        max_bytes = max_mb * 1024 * 1024

        # Calculate total size and collect files with their sizes and mtimes
        files_info = []
        total_size = 0
        try:
            for entry in backup_dir.iterdir():
                if entry.is_file():
                    try:
                        stat = entry.stat()
                        files_info.append((entry, stat.st_size, stat.st_mtime))
                        total_size += stat.st_size
                    except OSError:
                        continue
        except OSError as e:
            logger.debug("Could not scan backup directory %s: %s", backup_dir, e)
            return

        if total_size <= max_bytes:
            return

        logger.warning(
            "Backup directory %s size %.1f MB exceeds threshold %d MB, cleaning up",
            backup_dir, total_size / (1024 * 1024), max_mb,
        )

        # Sort by mtime ascending (oldest first)
        files_info.sort(key=lambda x: x[2])

        cleaned_count = 0
        cleaned_bytes = 0
        for fpath, fsize, _ in files_info:
            if total_size <= max_bytes:
                break
            try:
                fpath.unlink()
                total_size -= fsize
                cleaned_count += 1
                cleaned_bytes += fsize
            except OSError:
                continue

        if cleaned_count > 0:
            logger.info(
                "Cleaned %d backup files (%.1f MB freed), "
                "directory now %.1f MB",
                cleaned_count, cleaned_bytes / (1024 * 1024),
                total_size / (1024 * 1024),
            )

        # Also clean up old .corrupt files in state directory
        state_dir = Path(self.config.paths.state_dir)
        if state_dir.exists():
            cutoff = time.time() - (7 * 86400)  # 7 days
            for entry in state_dir.iterdir():
                if entry.is_file() and ".corrupt" in entry.name:
                    try:
                        if entry.stat().st_mtime < cutoff:
                            entry.unlink()
                            logger.debug("Removed old corrupt backup: %s", entry.name)
                    except OSError:
                        continue

    def check_git_object_growth(self) -> None:
        """Monitor .git/objects directory size to detect repository bloat.

        Repeated commits and rollbacks can cause unbounded growth in the git
        object store. This check warns when the objects directory exceeds
        the configured threshold and attempts a 'git gc' to reclaim space.
        Raises SafetyError if objects exceed 2x the threshold after gc.
        """
        target = self.config.target_dir
        max_mb = self.config.safety.max_git_objects_mb
        if max_mb <= 0:
            return

        git_objects_dir = Path(target) / ".git" / "objects"
        if not git_objects_dir.exists():
            return

        total_size = 0
        try:
            for dirpath, _dirnames, filenames in os.walk(str(git_objects_dir)):
                for fname in filenames:
                    try:
                        total_size += os.path.getsize(os.path.join(dirpath, fname))
                    except OSError:
                        continue
        except OSError as e:
            logger.debug("Could not scan .git/objects: %s", e)
            return

        size_mb = total_size / (1024 * 1024)

        if size_mb >= max_mb:
            logger.warning(
                "Git objects directory %.1f MB exceeds threshold %d MB, running git gc",
                size_mb, max_mb,
            )
            import subprocess
            try:
                subprocess.run(
                    ["git", "gc", "--auto"],
                    cwd=target,
                    capture_output=True,
                    timeout=120,
                )
            except (subprocess.TimeoutExpired, OSError) as e:
                logger.warning("git gc failed: %s", e)

            # Re-measure after gc
            post_gc_size = 0
            try:
                for dirpath, _dirnames, filenames in os.walk(str(git_objects_dir)):
                    for fname in filenames:
                        try:
                            post_gc_size += os.path.getsize(
                                os.path.join(dirpath, fname)
                            )
                        except OSError:
                            continue
            except OSError:
                return

            post_gc_mb = post_gc_size / (1024 * 1024)
            if post_gc_mb >= max_mb * 2:
                raise SafetyError(
                    f"Git objects directory still {post_gc_mb:.0f} MB after gc "
                    f"(limit: {max_mb} MB). Repository may need manual cleanup."
                )
            elif post_gc_mb >= max_mb:
                logger.warning(
                    "Git objects still %.1f MB after gc (threshold: %d MB)",
                    post_gc_mb, max_mb,
                )

    def pre_flight_checks(self) -> None:
        """Run all pre-cycle safety checks."""
        self.check_disk_space()
        self.check_memory()
        self.check_rate_limit()
        self.check_cost_limit()
        self.check_consecutive_failures()
        self.check_backup_dir_size()
        self.check_git_object_growth()

    def post_claude_checks(self, changed_files: List[str]) -> None:
        """Run safety checks after Claude has made changes."""
        self.check_protected_files(changed_files)
        self.check_file_count(changed_files)
