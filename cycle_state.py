"""Live cycle state for dashboard visibility.

The orchestrator writes state/current_cycle.json at phase transitions;
the dashboard polls it to show what's happening right now.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class CycleState:
    phase: str = ""                    # e.g. "task_selected", "planning", "executing", "validating", "retrying"
    task_description: str = ""
    task_type: str = ""
    task_descriptions: list = field(default_factory=list)
    started_at: float = 0.0
    pipeline_agent: str = ""           # e.g. "planner", "coder", "tester", "reviewer"
    pipeline_revision: int = 0
    accumulated_cost: float = 0.0
    batch_size: int = 1
    retry_count: int = 0


class CycleStateWriter:
    """Writes current cycle state atomically to a JSON file."""

    def __init__(self, state_dir: str, worker_id: Optional[int] = None):
        if worker_id is not None:
            filename = f"current_cycle_worker_{worker_id}.json"
        else:
            filename = "current_cycle.json"
        self._path = Path(state_dir) / filename
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> str:
        return str(self._path)

    def write(self, state: CycleState) -> None:
        """Atomically write cycle state via tempfile + os.replace."""
        data = asdict(state)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(self._path.parent), suffix=".tmp"
            )
            try:
                f = os.fdopen(tmp_fd, "w")
            except Exception:
                os.close(tmp_fd)
                raise
            with f:
                json.dump(data, f)
            os.replace(tmp_path, str(self._path))
        except OSError as e:
            logger.warning("Failed to write cycle state: %s", e)
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def clear(self) -> None:
        """Remove the cycle state file (cycle completed)."""
        try:
            self._path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("Failed to clear cycle state: %s", e)

    def update(self, **kwargs: Any) -> None:
        """Read current state, merge kwargs, and write back."""
        current = read_cycle_state(str(self._path.parent))
        if current is None:
            current = CycleState()
        for k, v in kwargs.items():
            if hasattr(current, k):
                setattr(current, k, v)
        self.write(current)


def read_cycle_state(state_dir: str) -> Optional[CycleState]:
    """Read current cycle state from disk. Returns None if no active cycle."""
    path = Path(state_dir) / "current_cycle.json"
    if not path.exists():
        return None
    try:
        text = path.read_text().strip()
        if not text:
            return None
        data = json.loads(text)
        return CycleState(
            phase=data.get("phase", ""),
            task_description=data.get("task_description", ""),
            task_type=data.get("task_type", ""),
            task_descriptions=data.get("task_descriptions", []),
            started_at=data.get("started_at", 0.0),
            pipeline_agent=data.get("pipeline_agent", ""),
            pipeline_revision=data.get("pipeline_revision", 0),
            accumulated_cost=data.get("accumulated_cost", 0.0),
            batch_size=data.get("batch_size", 1),
            retry_count=data.get("retry_count", 0),
        )
    except (json.JSONDecodeError, OSError):
        return None
