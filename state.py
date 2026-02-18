"""Persist cycle history to state/history.json."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config_schema import Config

logger = logging.getLogger(__name__)


@dataclass
class CycleRecord:
    timestamp: float
    task_description: str
    task_type: str = "unknown"
    success: bool = False
    commit_hash: str = ""
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    validation_summary: str = ""
    error: str = ""
    task_descriptions: List[str] = field(default_factory=list)
    task_types: List[str] = field(default_factory=list)


class StateManager:
    def __init__(self, config: Config):
        self.config = config
        self.history_file = Path(config.paths.history_file)
        self._cache: Optional[List[Dict[str, Any]]] = None
        self._cache_mtime: float = 0.0
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_history(self) -> List[Dict[str, Any]]:
        if not self.history_file.exists():
            self._cache = []
            self._cache_mtime = 0.0
            return []
        try:
            current_mtime = self.history_file.stat().st_mtime
            if self._cache is not None and current_mtime == self._cache_mtime:
                return self._cache
            text = self.history_file.read_text().strip()
            if not text:
                self._cache = []
                self._cache_mtime = current_mtime
                return []
            records = json.loads(text)
            self._cache = records
            self._cache_mtime = current_mtime
            return records
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read history: %s", e)
            return []

    def _save_history(self, records: List[Dict[str, Any]]) -> None:
        """Atomic write: write to temp file, then rename."""
        self._ensure_dir()
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self.history_file.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(records, f, indent=2)
            # os.replace can fail on Windows if target is open; retry with backoff
            retry_delays = [0.1, 0.3, 0.9, 2.7, 8.1]
            replaced = False
            last_err: Optional[OSError] = None
            for attempt, delay in enumerate(retry_delays):
                try:
                    os.replace(tmp_path, str(self.history_file))
                    replaced = True
                    break
                except OSError as e:
                    last_err = e
                    if attempt < len(retry_delays) - 1:
                        time.sleep(delay)
            if not replaced:
                assert last_err is not None  # loop ran at least once and every iteration set last_err
                raise last_err
            self._cache = records
            self._cache_mtime = self.history_file.stat().st_mtime
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _prune_history(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prune history to the most recent max_history_records entries."""
        max_records = self.config.safety.max_history_records
        if len(records) > max_records:
            return records[-max_records:]
        return records

    def record_cycle(self, record: CycleRecord) -> None:
        """Append a cycle record to history."""
        records = self._load_history()
        records.append(asdict(record))
        records = self._prune_history(records)
        self._save_history(records)
        logger.info(
            "Recorded cycle: %s (success=%s)", record.task_description, record.success
        )

    def was_recently_attempted(self, task_description: str, lookback_seconds: int = 3600) -> bool:
        """Check if a task was attempted in the last lookback_seconds."""
        cutoff = time.time() - lookback_seconds
        records = self._load_history()
        for r in records:
            if r.get("timestamp", 0) >= cutoff:
                if r.get("task_description") == task_description:
                    return True
                if task_description in r.get("task_descriptions", []):
                    return True
        return False

    def get_cycle_count_last_hour(self) -> int:
        """Return number of cycles in the last hour."""
        cutoff = time.time() - 3600
        records = self._load_history()
        return sum(1 for r in records if r.get("timestamp", 0) >= cutoff)

    def get_total_cost(self, lookback_seconds: int = 3600) -> float:
        """Return total cost in USD over the lookback period."""
        cutoff = time.time() - lookback_seconds
        records = self._load_history()
        return sum(
            r.get("cost_usd", 0.0)
            for r in records
            if r.get("timestamp", 0) >= cutoff
        )

    def get_consecutive_failures(self) -> int:
        """Return the number of consecutive failures at the end of history."""
        records = self._load_history()
        count = 0
        for r in reversed(records):
            if r.get("success", False):
                break
            count += 1
        return count

    def get_task_failure_count(self, task_description: str, task_type: str = "") -> int:
        """Return the number of failed attempts for a specific task."""
        records = self._load_history()
        count = 0
        for r in records:
            if r.get("success", False):
                continue
            match = (r.get("task_description") == task_description
                     or task_description in r.get("task_descriptions", []))
            if match and (not task_type
                          or r.get("task_type") == task_type
                          or task_type in r.get("task_types", [])):
                count += 1
        return count
