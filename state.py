"""Persist cycle history to state/history.json."""

from __future__ import annotations

import errno
import json
import logging
import os
import shutil
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
    batch_size: int = 1
    task_keys: List[str] = field(default_factory=list)
    pipeline_mode: str = ""
    pipeline_revision_count: int = 0
    pipeline_review_approved: bool = True
    validation_retry_count: int = 0
    push_succeeded: Optional[bool] = None
    task_source_files: List[str] = field(default_factory=list)
    task_line_numbers: List[Optional[int]] = field(default_factory=list)


class StateManager:
    def __init__(self, config: Config):
        self.config = config
        self.history_file = Path(config.paths.history_file)
        self._cache: Optional[List[Dict[str, Any]]] = None
        self._cache_mtime: float = 0.0
        self._history_corrupt: bool = False
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

    def _try_restore_from_backups(self) -> Optional[List[Dict[str, Any]]]:
        """Attempt to restore history from .corrupt backup files.

        Returns the parsed records from the most recent readable backup,
        or None if no backup can be parsed.
        """
        parent = self.history_file.parent
        base_name = self.history_file.name
        backups = sorted(
            parent.glob(f"{base_name}.corrupt*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for backup in backups:
            try:
                text = backup.read_text().strip()
                if text:
                    records = json.loads(text)
                    if isinstance(records, list):
                        logger.info(
                            "Restored %d history records from backup %s",
                            len(records), backup,
                        )
                        return records
            except (json.JSONDecodeError, OSError):
                continue
        return None

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
            self._history_corrupt = False
            self._cache = records
            self._cache_mtime = current_mtime
            return records
        except json.JSONDecodeError as e:
            logger.error("History file is corrupt: %s", e)
            # Back up corrupted file so record_cycle won't overwrite it
            backup = str(self.history_file) + ".corrupt"
            try:
                import shutil
                shutil.copy2(str(self.history_file), backup)
                logger.warning("Backed up corrupted history to %s", backup)
            except OSError as backup_err:
                logger.warning("Could not back up corrupted history: %s", backup_err)

            # Attempt to restore from a previous backup
            restored = self._try_restore_from_backups()
            if restored is not None:
                self._cache = restored
                self._history_corrupt = False
            else:
                logger.warning(
                    "No valid backup found. Corrupted history backed up. "
                    "Starting fresh — new cycles will write to a clean history file."
                )
                self._cache = []
                self._history_corrupt = False
            self._cache_mtime = self.history_file.stat().st_mtime
            return self._cache
        except OSError as e:
            logger.warning("Failed to read history: %s", e)
            return []

    def _save_history(self, records: List[Dict[str, Any]]) -> None:
        """Atomic write: write to temp file, then rename.

        Refuses to overwrite the history file if it was flagged as corrupt
        and unrecoverable, to prevent data loss.

        Pre-checks available disk space (10 MB threshold) before attempting
        to write, allowing graceful degradation when disk is near full.
        """
        if self._history_corrupt:
            logger.error(
                "Refusing to save history: file is corrupt and unrecoverable. "
                "Manually fix or remove %s to resume.", self.history_file,
            )
            return

        # Disk space pre-check: require at least 10 MB free
        _MIN_FREE_BYTES = 10 * 1024 * 1024
        try:
            usage = shutil.disk_usage(str(self.history_file.parent))
            if usage.free < _MIN_FREE_BYTES:
                logger.warning(
                    "Low disk space: only %.1f MB free (need %.1f MB). "
                    "Skipping history save to prevent silent write failure.",
                    usage.free / (1024 * 1024),
                    _MIN_FREE_BYTES / (1024 * 1024),
                )
                return
        except OSError as e:
            # If the disk check itself fails (e.g., path not mounted),
            # log and continue — don't block writes over a check failure
            logger.debug("Disk space check failed (continuing anyway): %s", e)

        self._ensure_dir()
        # Pre-check: verify records are JSON-serializable before writing.
        # This catches circular references, non-serializable types (e.g.,
        # unconverted dataclass instances with self-references), etc.
        try:
            json.dumps(records)
        except (TypeError, ValueError) as e:
            logger.error(
                "Refusing to save history: records are not JSON-serializable: %s", e,
            )
            return
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self.history_file.parent), suffix=".tmp"
        )
        try:
            try:
                f = os.fdopen(tmp_fd, "w")
            except Exception:
                os.close(tmp_fd)
                raise
            with f:
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
                        logger.debug(
                            "os.replace failed (attempt %d/%d): %s — retrying in %.1fs",
                            attempt + 1, len(retry_delays), e, delay,
                        )
                        time.sleep(delay)
            if not replaced:
                if last_err is not None:
                    raise last_err
                raise OSError("os.replace failed: no retries were attempted")
            self._cache = records
            self._cache_mtime = self.history_file.stat().st_mtime
        except OSError as e:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            # Graceful degradation: if disk is full, log a warning and
            # continue instead of crashing the orchestrator.
            if e.errno == errno.ENOSPC:
                logger.warning(
                    "Disk full: unable to save cycle history. "
                    "Cycle data will be lost. Free disk space to resume normal operation."
                )
                return
            raise
        except Exception:
            # Clean up temp file on failure (non-OSError cases)
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
        records = list(self._load_history())
        records.append(asdict(record))
        records = self._prune_history(records)
        self._save_history(records)
        logger.info(
            "Recorded cycle: %s (success=%s)", record.task_description, record.success
        )

    def was_recently_attempted(self, task_description: str, lookback_seconds: int = 3600, task_key: str = "") -> bool:
        """Check if a task was attempted in the last lookback_seconds."""
        cutoff = time.time() - lookback_seconds
        records = self._load_history()
        for r in records:
            if r.get("timestamp", 0) >= cutoff:
                if r.get("task_description") == task_description:
                    return True
                if task_description in r.get("task_descriptions", []):
                    return True
                if task_key and task_key in r.get("task_keys", []):
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

    def compute_adaptive_batch_size(self) -> int:
        """Replay recent history to compute adaptive batch size."""
        orch = self.config.orchestrator
        size = orch.initial_batch_size
        records = self._load_history()
        recent = records[-orch.adaptive_batch_window:]

        for r in recent:
            if r.get("success", False):
                size += orch.batch_grow_step
            else:
                size -= orch.batch_shrink_step
            size = max(orch.min_batch_size, min(orch.max_batch_size, size))

        return size

    def get_task_failure_count(self, task_description: str, task_type: str = "", task_key: str = "") -> int:
        """Return the number of failed attempts for a specific task."""
        records = self._load_history()
        count = 0
        for r in records:
            if r.get("success", False):
                continue
            match = (r.get("task_description") == task_description
                     or task_description in r.get("task_descriptions", []))
            if not match and task_key:
                match = task_key in r.get("task_keys", [])
            if match and (not task_type
                          or r.get("task_type") == task_type
                          or task_type in r.get("task_types", [])):
                count += 1
        return count

    def reset_consecutive_failures(self, reason: str = "manual reset") -> None:
        """Inject a synthetic success record to break the consecutive failure chain."""
        record = CycleRecord(
            timestamp=time.time(),
            task_description=f"System reset: {reason}",
            task_type="system_reset",
            success=True,
        )
        self.record_cycle(record)
        logger.info("Reset consecutive failures: %s", reason)

    def should_auto_reset_failures(self, min_idle_seconds: int = 3600) -> bool:
        """Return True if consecutive failures >= max and system has been idle long enough."""
        failures = self.get_consecutive_failures()
        limit = self.config.safety.max_consecutive_failures
        if failures < limit:
            return False
        records = self._load_history()
        if not records:
            return False
        last_timestamp = records[-1].get("timestamp", 0)
        idle_seconds = time.time() - last_timestamp
        return idle_seconds >= min_idle_seconds
