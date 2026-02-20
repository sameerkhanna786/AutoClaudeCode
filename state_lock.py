"""Thread-safe StateManager using file-level locking for concurrent access."""

from __future__ import annotations

import fcntl
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

from config_schema import Config
from state import CycleRecord, StateManager

logger = logging.getLogger(__name__)


class LockedStateManager(StateManager):
    """Thread-safe StateManager using fcntl.flock on a lock file.

    Wraps read-modify-write operations (record_cycle, was_recently_attempted)
    with an exclusive file lock so multiple worker threads can safely share
    a single history.json file.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self._lock_path = Path(config.paths.state_dir) / "history.lock"

    @contextmanager
    def _file_lock(self):
        """Acquire exclusive lock on history.lock for read-modify-write safety."""
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self._lock_path), os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def record_cycle(self, record: CycleRecord) -> None:
        with self._file_lock():
            # Invalidate cache to force re-read from disk
            self._cache = None
            super().record_cycle(record)

    def was_recently_attempted(self, task_description: str, lookback_seconds: int = 3600, task_key: str = "") -> bool:
        with self._file_lock():
            self._cache = None
            return super().was_recently_attempted(task_description, lookback_seconds, task_key)

    def get_cycle_count_last_hour(self) -> int:
        with self._file_lock():
            self._cache = None
            return super().get_cycle_count_last_hour()

    def get_total_cost(self, lookback_seconds: int = 3600) -> float:
        with self._file_lock():
            self._cache = None
            return super().get_total_cost(lookback_seconds)

    def get_consecutive_failures(self) -> int:
        with self._file_lock():
            self._cache = None
            return super().get_consecutive_failures()

    def get_task_failure_count(self, task_description: str, task_type: str = "", task_key: str = "") -> int:
        with self._file_lock():
            self._cache = None
            return super().get_task_failure_count(task_description, task_type, task_key)
