"""Task approval queue manager — file-based queue for dashboard task approval.

When the dashboard is connected (heartbeat active), auto-discovered tasks are
queued for user approval instead of executing immediately. Feedback tasks
bypass the queue and execute directly.

File locations:
    state/pending_approval/*.json — tasks awaiting user decision
    state/approved/*.json — tasks the user accepted (consumed by orchestrator)
    state/dashboard_heartbeat.json — {"timestamp": <unix_time>}
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from task_discovery import Task

logger = logging.getLogger(__name__)

# Sanitize task keys for safe filenames
_SAFE_FILENAME_RE = re.compile(r'[^a-zA-Z0-9_.-]')
MAX_FILENAME_LENGTH = 120


def _sanitize_filename(key: str) -> str:
    """Convert a task key to a safe filename."""
    safe = _SAFE_FILENAME_RE.sub('_', key)
    if len(safe) > MAX_FILENAME_LENGTH:
        safe = safe[:MAX_FILENAME_LENGTH]
    return safe


class TaskApprovalQueue:
    """File-based task approval queue for dashboard integration.

    Follows the same file-based patterns used by feedback.py and state.py.
    """

    def __init__(self, state_dir: str):
        self._state_dir = Path(state_dir)
        self._pending_dir = self._state_dir / "pending_approval"
        self._approved_dir = self._state_dir / "approved"
        self._heartbeat_path = self._state_dir / "dashboard_heartbeat.json"
        self._declined_keys: Dict[str, float] = {}  # task_key -> decline timestamp
        self._declined_lock = threading.Lock()
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self._pending_dir.mkdir(parents=True, exist_ok=True)
        self._approved_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Dashboard heartbeat
    # ------------------------------------------------------------------

    def update_heartbeat(self) -> None:
        """Write dashboard heartbeat file. Called by dashboard on each poll."""
        data = {"timestamp": time.time()}
        tmp_path = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(self._state_dir), suffix=".tmp",
            )
            try:
                f = os.fdopen(tmp_fd, "w")
            except Exception:
                os.close(tmp_fd)
                raise
            with f:
                json.dump(data, f)
            os.replace(tmp_path, str(self._heartbeat_path))
        except OSError as e:
            logger.warning("Failed to write dashboard heartbeat: %s", e)
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def is_dashboard_active(self, timeout: int = 30) -> bool:
        """Check if dashboard heartbeat is recent enough."""
        if not self._heartbeat_path.exists():
            return False
        try:
            text = self._heartbeat_path.read_text().strip()
            if not text:
                return False
            data = json.loads(text)
            ts = data.get("timestamp", 0)
            return (time.time() - ts) < timeout
        except (json.JSONDecodeError, OSError, TypeError):
            return False

    # ------------------------------------------------------------------
    # Queue operations
    # ------------------------------------------------------------------

    def enqueue(self, task: Task, cooldown_seconds: int = 0) -> Optional[str]:
        """Write a task to pending_approval/ as a JSON file.

        Returns the task_id if enqueued, None if skipped (duplicate or
        recently declined).
        """
        task_key = task.task_key
        safe_name = _sanitize_filename(task_key)
        pending_path = self._pending_dir / f"{safe_name}.json"
        approved_path = self._approved_dir / f"{safe_name}.json"

        # Skip if already pending or approved
        if pending_path.exists() or approved_path.exists():
            return None

        # Skip if recently declined
        with self._declined_lock:
            if task_key in self._declined_keys:
                declined_at = self._declined_keys[task_key]
                if cooldown_seconds > 0 and (time.time() - declined_at) < cooldown_seconds:
                    return None
                # Cooldown expired, allow re-enqueue
                del self._declined_keys[task_key]

        task_data = {
            "task_key": task_key,
            "description": task.description,
            "priority": task.priority,
            "source": task.source,
            "source_file": task.source_file,
            "line_number": task.line_number,
            "context": task.context,
            "task_id": task.task_id,
            "depends_on": task.depends_on,
            "enqueued_at": time.time(),
        }

        tmp_path = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(self._pending_dir), suffix=".tmp",
            )
            try:
                f = os.fdopen(tmp_fd, "w")
            except Exception:
                os.close(tmp_fd)
                raise
            with f:
                json.dump(task_data, f, indent=2)
            os.replace(tmp_path, str(pending_path))
            return task_key
        except OSError as e:
            logger.warning("Failed to enqueue task %s: %s", task_key, e)
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            return None

    def list_pending(self) -> List[Dict[str, Any]]:
        """List all pending approval tasks."""
        tasks = []
        if not self._pending_dir.exists():
            return tasks
        for f in sorted(self._pending_dir.iterdir()):
            if not f.is_file() or f.suffix != ".json":
                continue
            try:
                data = json.loads(f.read_text())
                data["id"] = f.stem
                tasks.append(data)
            except (json.JSONDecodeError, OSError):
                continue
        return tasks

    def approve(self, task_id: str) -> bool:
        """Move a task from pending_approval/ to approved/.

        Uses a rename-based atomic swap to prevent ghost duplicates: the
        pending file is updated in-place (via temp+replace), then atomically
        moved to the approved directory with a single ``os.replace`` call.
        This eliminates the crash window where both files could exist.
        """
        pending_path = self._pending_dir / f"{task_id}.json"
        approved_path = self._approved_dir / f"{task_id}.json"

        if not pending_path.exists():
            return False

        try:
            # Read pending task and add approval timestamp
            content = pending_path.read_text()
            data = json.loads(content)
            data["approved_at"] = time.time()

            # Step 1: Atomically update the pending file with approved_at
            # (write to temp file in the same dir, then os.replace)
            tmp_path = None
            try:
                tmp_fd, tmp_path = tempfile.mkstemp(
                    dir=str(self._pending_dir), suffix=".tmp",
                )
                try:
                    f = os.fdopen(tmp_fd, "w")
                except Exception:
                    os.close(tmp_fd)
                    raise
                with f:
                    json.dump(data, f, indent=2)
                os.replace(tmp_path, str(pending_path))
            except OSError as e:
                logger.warning("Failed to approve task %s: %s", task_id, e)
                if tmp_path is not None:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                return False

            # Step 2: Atomically move from pending to approved — single
            # operation that removes the source and creates the destination,
            # so there is no window where both files exist.
            try:
                os.replace(str(pending_path), str(approved_path))
            except FileNotFoundError:
                logger.warning("Pending file %s disappeared before approval move", task_id)
                return False
            except OSError as e:
                logger.warning("Failed to move pending task %s to approved: %s", task_id, e)
                return False
            return True
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read pending task %s: %s", task_id, e)
            return False

    def decline(self, task_id: str) -> bool:
        """Delete a task from pending_approval/ (declined by user)."""
        pending_path = self._pending_dir / f"{task_id}.json"
        if not pending_path.exists():
            return False

        try:
            # Read the task key before deleting for cooldown tracking
            data = json.loads(pending_path.read_text())
            task_key = data.get("task_key", task_id)
            with self._declined_lock:
                self._declined_keys[task_key] = time.time()
            pending_path.unlink()
            return True
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to decline task %s: %s", task_id, e)
            return False

    def approve_all(self) -> int:
        """Approve all pending tasks. Returns count of approved tasks."""
        count = 0
        for task in self.list_pending():
            if self.approve(task["id"]):
                count += 1
        return count

    def decline_all(self) -> int:
        """Decline all pending tasks. Returns count of declined tasks."""
        count = 0
        for task in self.list_pending():
            if self.decline(task["id"]):
                count += 1
        return count

    def get_approved(self) -> List[Task]:
        """Read and consume approved tasks (delete after reading)."""
        tasks = []
        if not self._approved_dir.exists():
            return tasks

        for f in sorted(self._approved_dir.iterdir()):
            if not f.is_file() or f.suffix != ".json":
                continue
            try:
                data = json.loads(f.read_text())
                task = Task(
                    description=data.get("description", ""),
                    priority=data.get("priority", 5),
                    source=data.get("source", "claude_idea"),
                    source_file=data.get("source_file"),
                    line_number=data.get("line_number"),
                    context=data.get("context", ""),
                    task_id=data.get("task_id", ""),
                    depends_on=data.get("depends_on", []),
                )
                tasks.append(task)
                # Consume: delete the approved file
                f.unlink()
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read approved task %s: %s", f.name, e)
                continue

        return tasks

    def pending_count(self) -> int:
        """Return the number of tasks awaiting approval."""
        if not self._pending_dir.exists():
            return 0
        return sum(
            1 for f in self._pending_dir.iterdir()
            if f.is_file() and f.suffix == ".json"
        )

    def clear_stale(self, max_age: int = 3600) -> int:
        """Clean up old unapproved tasks. Returns count of removed tasks."""
        cutoff = time.time() - max_age
        count = 0
        if not self._pending_dir.exists():
            return 0

        for f in self._pending_dir.iterdir():
            if not f.is_file() or f.suffix != ".json":
                continue
            try:
                data = json.loads(f.read_text())
                enqueued_at = data.get("enqueued_at", 0)
                if enqueued_at > 0 and enqueued_at < cutoff:
                    f.unlink()
                    count += 1
            except (json.JSONDecodeError, OSError):
                continue

        return count
