"""Thread-safe StateManager using file-level locking for concurrent access."""

from __future__ import annotations

import fcntl
import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from config_schema import Config
from state import CycleRecord, StateManager

logger = logging.getLogger(__name__)


class LockedStateManager(StateManager):
    """Thread-safe StateManager using fcntl.flock on a lock file.

    Wraps read-modify-write operations (record_cycle, was_recently_attempted)
    with an exclusive file lock so multiple worker threads can safely share
    a single history.json file.

    The lock is re-entrant: if the current thread already holds the lock,
    nested acquisitions are no-ops. This prevents deadlocks when locked
    methods call other locked methods internally (e.g.,
    should_auto_reset_failures -> get_consecutive_failures,
    reset_consecutive_failures -> record_cycle).
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self._lock_path = Path(config.paths.state_dir) / "history.lock"
        self._local = threading.local()

    @contextmanager
    def _file_lock(self):
        """Acquire exclusive lock on history.lock for read-modify-write safety.

        Re-entrant: if the current thread already holds the lock, yields
        immediately without re-acquiring (avoiding deadlock on the same inode).
        """
        # Check if current thread already holds the lock
        if getattr(self._local, 'held', False):
            yield
            return

        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self._lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            self._local.held = True
            yield
        finally:
            os.close(fd)
            self._local.held = False

    def record_cycle(self, record: CycleRecord) -> None:
        with self._file_lock():
            # Invalidate cache to force re-read from disk
            self._cache = None
            super().record_cycle(record)

    def was_recently_attempted(self, task_description: str, lookback_seconds: int = 3600, task_key: str = "") -> bool:
        with self._file_lock():
            return super().was_recently_attempted(task_description, lookback_seconds, task_key)

    def get_cycle_count_last_hour(self) -> int:
        with self._file_lock():
            return super().get_cycle_count_last_hour()

    def get_total_cost(self, lookback_seconds: int = 3600) -> float:
        with self._file_lock():
            return super().get_total_cost(lookback_seconds)

    def get_consecutive_failures(self) -> int:
        with self._file_lock():
            return super().get_consecutive_failures()

    def get_task_failure_count(self, task_description: str, task_type: str = "",
                              task_key: str = "", lookback_seconds: int = 86400) -> int:
        with self._file_lock():
            return super().get_task_failure_count(task_description, task_type, task_key, lookback_seconds)

    def compute_adaptive_batch_size(self) -> int:
        with self._file_lock():
            return super().compute_adaptive_batch_size()

    def get_recent_task_summaries(self, lookback_seconds: int = 86400, max_items: int = 20) -> List[str]:
        with self._file_lock():
            return super().get_recent_task_summaries(lookback_seconds, max_items)

    def should_auto_reset_failures(self, min_idle_seconds: int = 3600) -> bool:
        with self._file_lock():
            return super().should_auto_reset_failures(min_idle_seconds)

    def reset_consecutive_failures(self, reason: str = "manual reset") -> None:
        with self._file_lock():
            # Invalidate cache — this method writes to disk
            self._cache = None
            super().reset_consecutive_failures(reason)

    def get_success_rate_by_type(self, lookback_seconds: int = 86400) -> Dict[str, float]:
        with self._file_lock():
            return super().get_success_rate_by_type(lookback_seconds)

    def get_strategy_performance(self, lookback_seconds: int = 86400) -> Dict[str, Dict[str, Any]]:
        with self._file_lock():
            return super().get_strategy_performance(lookback_seconds)

    def get_productive_files(self, lookback_seconds: int = 86400) -> List[str]:
        with self._file_lock():
            return super().get_productive_files(lookback_seconds)

    def get_task_success_history(
        self,
        task_description: str,
        task_key: str = "",
        max_attempts: int = 5,
    ) -> List[Dict[str, Any]]:
        with self._file_lock():
            return super().get_task_success_history(task_description, task_key, max_attempts)

    def get_strategy_performance_report(self, lookback_seconds: int = 86400) -> str:
        with self._file_lock():
            return super().get_strategy_performance_report(lookback_seconds)

    def load_history(self) -> List[Dict[str, Any]]:
        with self._file_lock():
            # Return a shallow copy so callers don't mutate the internal cache
            return list(super().load_history())
