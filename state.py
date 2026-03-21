"""Persist cycle history to state/history.json."""

from __future__ import annotations

import errno
import json
import logging
import os
import random
import re
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config_schema import Config

logger = logging.getLogger(__name__)

# Pre-compiled regex for extracting file paths from task descriptions
_FILE_PATTERN_RE = re.compile(
    r'([a-zA-Z0-9_/.\-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|rb|sh|yaml|yml|json))'
)


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
        backups_with_mtime = []
        for p in parent.glob(f"{base_name}.corrupt*"):
            try:
                backups_with_mtime.append((p, p.stat().st_mtime))
            except OSError:
                continue  # file deleted between glob and stat
        backups_with_mtime.sort(key=lambda x: x[1], reverse=True)
        backups = [p for p, _ in backups_with_mtime]
        for backup in backups:
            try:
                text = backup.read_text(encoding="utf-8").strip()
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

    # Maximum history file size to load (50 MB). Prevents OOM on
    # corrupted or externally modified files that bypass pruning.
    _MAX_HISTORY_FILE_BYTES = 50 * 1024 * 1024

    def _load_history(self) -> List[Dict[str, Any]]:
        if not self.history_file.exists():
            self._cache = []
            self._cache_mtime = 0.0
            return []
        try:
            st = self.history_file.stat()
            current_mtime = st.st_mtime
            if self._cache is not None and current_mtime == self._cache_mtime:
                return list(self._cache)
            file_size = st.st_size
            if file_size > self._MAX_HISTORY_FILE_BYTES:
                logger.error(
                    "History file too large (%d bytes, limit %d). "
                    "Refusing to load to prevent OOM. Manual cleanup required.",
                    file_size, self._MAX_HISTORY_FILE_BYTES,
                )
                if self._cache is not None:
                    return list(self._cache)
                return []
            text = self.history_file.read_text(encoding="utf-8").strip()
            if not text:
                self._cache = []
                self._cache_mtime = current_mtime
                return []
            records = json.loads(text)
            if not isinstance(records, list):
                logger.error(
                    "History file contains %s instead of a JSON array",
                    type(records).__name__,
                )
                records = []

            self._cache = records
            self._cache_mtime = current_mtime
            return list(records)
        except json.JSONDecodeError as e:
            logger.error("History file is corrupt: %s", e)
            # Back up corrupted file so record_cycle won't overwrite it
            backup = str(self.history_file) + f".corrupt.{int(time.time())}"
            try:
                shutil.copy2(str(self.history_file), backup)
                logger.warning("Backed up corrupted history to %s", backup)
            except OSError as backup_err:
                logger.warning("Could not back up corrupted history: %s", backup_err)

            # Attempt to restore from a previous backup
            restored = self._try_restore_from_backups()
            if restored is not None:
                self._cache = restored
            else:
                logger.warning(
                    "No valid backup found. Corrupted history backed up. "
                    "Starting fresh — new cycles will write to a clean history file."
                )
                self._cache = []
            try:
                self._cache_mtime = self.history_file.stat().st_mtime
            except FileNotFoundError:
                self._cache_mtime = 0.0
            return list(self._cache)
        except OSError as e:
            logger.warning("Failed to read history: %s", e)
            # Return stale cache if available, otherwise empty list.
            # Never return [] when we have cached data — that would cause
            # record_cycle to overwrite the history file with a single entry.
            if self._cache is not None:
                return list(self._cache)
            return []

    def _save_history(self, records: List[Dict[str, Any]]) -> None:
        """Atomic write: write to temp file, then rename.

        Pre-checks available disk space (10 MB threshold) before attempting
        to write, allowing graceful degradation when disk is near full.
        """
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
        # Check for external modifications between read and write
        if self._cache_mtime > 0 and self.history_file.exists():
            try:
                current_mtime = self.history_file.stat().st_mtime
                if current_mtime != self._cache_mtime:
                    logger.warning(
                        "History file was modified externally since last read "
                        "(expected mtime %.6f, got %.6f). Data from the external "
                        "modification may be lost.",
                        self._cache_mtime, current_mtime,
                    )
            except OSError:
                pass  # File may have been deleted; proceed with write
        # Pre-serialize to validate JSON-serializability and avoid writing
        # a partial/corrupt file. This single serialization replaces both the
        # old validation-only json.dumps() and the file-writing json.dump().
        try:
            serialized = json.dumps(records, indent=2)
        except (TypeError, ValueError, RecursionError) as e:
            logger.error(
                "Refusing to save history: records are not JSON-serializable: %s", e,
            )
            return
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self.history_file.parent), suffix=".tmp"
        )
        try:
            # Restrict temp file permissions to owner-only (0o600) before
            # writing potentially sensitive data (error messages, cost info).
            os.fchmod(tmp_fd, 0o600)
            try:
                f = os.fdopen(tmp_fd, "w", encoding="utf-8")
            except Exception:
                os.close(tmp_fd)
                raise
            with f:
                f.write(serialized)
            # os.replace can fail on Windows if target is open; retry with
            # exponential backoff and jitter for persistent filesystem contention
            _REPLACE_BASE_DELAY = 0.1
            _REPLACE_MULTIPLIER = 3
            _REPLACE_MAX_RETRIES = 7
            _REPLACE_MAX_DELAY = 30.0
            replaced = False
            last_err: Optional[OSError] = None
            for attempt in range(_REPLACE_MAX_RETRIES):
                try:
                    os.replace(tmp_path, str(self.history_file))
                    replaced = True
                    break
                except OSError as e:
                    last_err = e
                    if attempt < _REPLACE_MAX_RETRIES - 1:
                        delay = min(
                            _REPLACE_BASE_DELAY * (_REPLACE_MULTIPLIER ** attempt),
                            _REPLACE_MAX_DELAY,
                        )
                        delay *= 0.5 + random.random() * 0.5  # jitter
                        logger.debug(
                            "os.replace failed (attempt %d/%d): %s — retrying in %.2fs",
                            attempt + 1, _REPLACE_MAX_RETRIES, e, delay,
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
                # Keep the existing cache intact: since the write failed
                # before os.replace, the on-disk file is unchanged and the
                # cache still matches it.  Invalidating would force a
                # re-read; if that also fails the fallback returns [],
                # which could cause the next successful record_cycle to
                # overwrite the entire history with a single entry.
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
        """Append a cycle record to history.

        Uses list copy to avoid mutating the in-memory cache before
        _save_history succeeds — otherwise a failed save would leave
        the cache in an inconsistent state.
        """
        records = list(self._load_history())
        records.append(asdict(record))
        records = self._prune_history(records)
        self._save_history(records)
        logger.info(
            "Recorded cycle: %s (success=%s)", record.task_description, record.success
        )

    def was_recently_attempted(self, task_description: str, lookback_seconds: int = 3600, task_key: str = "") -> bool:
        """Check if a task was attempted in the last lookback_seconds.

        Iterates in reverse since recent records are at the end.
        Stops early after seeing several consecutive old records,
        tolerating minor timestamp disorder from concurrent writes.
        """
        cutoff = time.time() - lookback_seconds
        records = self._load_history()
        consecutive_old = 0
        for r in reversed(records):
            if r.get("timestamp", 0) < cutoff:
                consecutive_old += 1
                if consecutive_old >= 5:
                    break
                continue
            consecutive_old = 0
            if r.get("task_description") == task_description:
                return True
            if task_description in r.get("task_descriptions", []):
                return True
            if task_key and task_key in r.get("task_keys", []):
                return True
        return False

    def get_cycle_count_last_hour(self) -> int:
        """Return number of cycles in the last hour.

        Iterates in reverse since recent records are at the end.
        Stops early after seeing several consecutive old records,
        tolerating minor timestamp disorder from concurrent writes.
        """
        cutoff = time.time() - 3600
        records = self._load_history()
        count = 0
        consecutive_old = 0
        for r in reversed(records):
            if r.get("timestamp", 0) >= cutoff:
                count += 1
                consecutive_old = 0
            else:
                consecutive_old += 1
                if consecutive_old >= 5:
                    break
        return count

    def get_total_cost(self, lookback_seconds: int = 3600) -> float:
        """Return total cost in USD over the lookback period.

        Iterates in reverse since recent records are at the end.
        Stops early after seeing several consecutive old records,
        tolerating minor timestamp disorder from concurrent writes.
        """
        cutoff = time.time() - lookback_seconds
        records = self._load_history()
        total = 0.0
        consecutive_old = 0
        for r in reversed(records):
            if r.get("timestamp", 0) >= cutoff:
                total += r.get("cost_usd", 0.0)
                consecutive_old = 0
            else:
                consecutive_old += 1
                if consecutive_old >= 5:
                    break
        return total

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
        """Replay recent history to compute adaptive batch size.

        Cost-aware: successful cycles that exceed batch_cost_ceiling
        do not grow the batch size, preventing runaway cost growth.
        """
        orch = self.config.orchestrator
        size = orch.initial_batch_size
        records = self._load_history()
        recent = records[-orch.adaptive_batch_window:]

        for r in recent:
            if r.get("success", False):
                # Cost-aware: don't grow if this cycle was expensive
                cost = r.get("cost_usd", 0.0)
                if cost < orch.batch_cost_ceiling:
                    size += orch.batch_grow_step
                # else: success but expensive — hold steady
            else:
                size -= orch.batch_shrink_step
            size = max(orch.min_batch_size, min(orch.max_batch_size, size))

        return size

    def get_task_failure_count(self, task_description: str, task_type: str = "",
                              task_key: str = "", lookback_seconds: int = 86400) -> int:
        """Return the number of failed attempts for a specific task.

        Only considers records within the lookback window (default 24h).
        Iterates in reverse since recent records are at the end.
        Stops early after seeing several consecutive old records,
        tolerating minor timestamp disorder from concurrent writes.
        """
        cutoff = time.time() - lookback_seconds
        records = self._load_history()
        count = 0
        consecutive_old = 0
        for r in reversed(records):
            if r.get("timestamp", 0) < cutoff:
                consecutive_old += 1
                if consecutive_old >= 5:
                    break
                continue
            consecutive_old = 0
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
        records = self._load_history()
        if not records:
            return False
        # Count consecutive failures from the end (inline to avoid double load)
        failures = 0
        for r in reversed(records):
            if r.get("success", False):
                break
            failures += 1
        limit = self.config.safety.max_consecutive_failures
        if failures < limit:
            return False
        last_timestamp = records[-1].get("timestamp", 0)
        idle_seconds = time.time() - last_timestamp
        return idle_seconds >= min_idle_seconds

    def get_recent_task_summaries(self, lookback_seconds: int = 86400, max_items: int = 20) -> List[str]:
        """Return human-readable summaries of recent tasks for prompt injection.

        Iterates in reverse since recent records are at the end.
        Stops early after seeing several consecutive old records,
        tolerating minor timestamp disorder from concurrent writes.
        """
        cutoff = time.time() - lookback_seconds
        records = self._load_history()
        summaries = []
        consecutive_old = 0
        for r in reversed(records):
            if r.get("timestamp", 0) >= cutoff:
                consecutive_old = 0
                desc = r.get("task_description", "")
                if len(desc) > 100:
                    desc = desc[:97] + "..."
                status = "succeeded" if r.get("success") else "failed"
                summaries.append(f"- {desc} ({status})")
            else:
                consecutive_old += 1
                if consecutive_old >= 5:
                    break
        summaries.reverse()
        return summaries[-max_items:]

    def get_success_rate_by_type(self, lookback_seconds: int = 86400) -> Dict[str, float]:
        """Compute success rates per task_type over the lookback window.

        Returns a dict mapping task_type -> success_rate (0.0-1.0).
        Types with fewer than 2 attempts are not included.

        Iterates in reverse since recent records are at the end.
        Stops early after seeing several consecutive old records,
        tolerating minor timestamp disorder from concurrent writes.
        """
        cutoff = time.time() - lookback_seconds
        records = self._load_history()

        type_counts: Dict[str, Dict[str, int]] = {}  # type -> {total, successes}
        consecutive_old = 0
        for r in reversed(records):
            if r.get("timestamp", 0) < cutoff:
                consecutive_old += 1
                if consecutive_old >= 5:
                    break
                continue
            consecutive_old = 0
            task_type = r.get("task_type", "unknown")
            if task_type not in type_counts:
                type_counts[task_type] = {"total": 0, "successes": 0}
            type_counts[task_type]["total"] += 1
            if r.get("success", False):
                type_counts[task_type]["successes"] += 1

        rates: Dict[str, float] = {}
        for task_type, counts in type_counts.items():
            if counts["total"] >= 2:
                rates[task_type] = counts["successes"] / counts["total"]

        return rates

    def get_strategy_performance(self, lookback_seconds: int = 86400) -> Dict[str, Dict[str, Any]]:
        """Return per-source performance metrics: source -> {total, successes, success_rate, avg_cost, avg_duration}

        Iterates in reverse since recent records are at the end.
        Stops early after seeing several consecutive old records,
        tolerating minor timestamp disorder from concurrent writes.
        """
        cutoff = time.time() - lookback_seconds
        records = self._load_history()
        performance: Dict[str, Dict[str, Any]] = {}
        consecutive_old = 0
        for r in reversed(records):
            if r.get("timestamp", 0) < cutoff:
                consecutive_old += 1
                if consecutive_old >= 5:
                    break
                continue
            consecutive_old = 0
            source = r.get("task_type", "unknown")
            if source not in performance:
                performance[source] = {"total": 0, "successes": 0, "total_cost": 0.0, "total_duration": 0.0}
            perf = performance[source]
            perf["total"] += 1
            if r.get("success", False):
                perf["successes"] += 1
            perf["total_cost"] += r.get("cost_usd", 0.0)
            perf["total_duration"] += r.get("duration_seconds", 0.0)
        for source, perf in performance.items():
            total = perf["total"]
            perf["success_rate"] = perf["successes"] / total if total > 0 else 0.0
            perf["avg_cost"] = perf["total_cost"] / total if total > 0 else 0.0
            perf["avg_duration"] = perf["total_duration"] / total if total > 0 else 0.0
        return performance

    def get_productive_files(self, lookback_seconds: int = 86400) -> List[str]:
        """Return files successfully modified most often, sorted by frequency.

        Iterates in reverse since recent records are at the end.
        Stops early after seeing several consecutive old records,
        tolerating minor timestamp disorder from concurrent writes.
        """
        cutoff = time.time() - lookback_seconds
        records = self._load_history()
        file_counts: Dict[str, int] = {}
        consecutive_old = 0
        for r in reversed(records):
            if r.get("timestamp", 0) < cutoff:
                consecutive_old += 1
                if consecutive_old >= 5:
                    break
                continue
            consecutive_old = 0
            if not r.get("success", False):
                continue
            for task_desc in (r.get("task_descriptions") or [r.get("task_description", "")]):
                matches = _FILE_PATTERN_RE.findall(task_desc)
                for m in matches:
                    file_counts[m] = file_counts.get(m, 0) + 1
        return sorted(file_counts.keys(), key=lambda f: file_counts[f], reverse=True)

    def get_task_success_history(
        self,
        task_description: str,
        task_key: str = "",
        max_attempts: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return the last N attempts for a given task, including errors.

        Each entry contains: {"success": bool, "error": str,
        "validation_summary": str, "timestamp": float}.
        Matches by task_description or task_key (if provided).
        """
        records = self._load_history()
        matches: List[Dict[str, Any]] = []
        for r in reversed(records):
            match = r.get("task_description") == task_description
            if not match:
                match = task_description in r.get("task_descriptions", [])
            if not match and task_key:
                match = task_key in r.get("task_keys", [])
            if match:
                matches.append({
                    "success": r.get("success", False),
                    "error": r.get("error", ""),
                    "validation_summary": r.get("validation_summary", ""),
                    "timestamp": r.get("timestamp", 0),
                })
                if len(matches) >= max_attempts:
                    break
        matches.reverse()
        return matches

    def get_strategy_performance_report(self, lookback_seconds: int = 86400) -> str:
        """Return a formatted string showing performance per task type over the lookback period.

        Shows success rate, average cost, and average duration for each task type.
        """
        perf = self.get_strategy_performance(lookback_seconds=lookback_seconds)
        if not perf:
            return "No task history in the last 24 hours."

        lines = ["Strategy Performance (last 24h):"]
        # Sort by success rate descending, then by total attempts descending
        for source in sorted(perf, key=lambda s: (-perf[s]["success_rate"], -perf[s]["total"])):
            p = perf[source]
            lines.append(
                f"  {source}: {p['successes']}/{p['total']} succeeded "
                f"({p['success_rate']:.0%}), "
                f"avg cost ${p['avg_cost']:.4f}, "
                f"avg duration {p['avg_duration']:.1f}s"
            )
        return "\n".join(lines)

    def load_history(self) -> List[Dict[str, Any]]:
        """Public API for loading history (safe for external callers).

        Returns a shallow copy so callers cannot mutate the internal cache.
        """
        return list(self._load_history())
