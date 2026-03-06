"""Session recovery: save/restore orchestrator state across crashes."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SESSION_FILE = "session.json"


@dataclass
class SessionState:
    """Persistent session state for crash recovery."""
    session_id: str
    started_at: float
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    completed_task_ids: List[str] = field(default_factory=list)
    worker_states: Dict[int, Dict] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    phase: str = "starting"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SessionState:
        return cls(
            session_id=data.get("session_id", ""),
            started_at=data.get("started_at", 0.0),
            tasks=data.get("tasks", []),
            completed_task_ids=data.get("completed_task_ids", []),
            worker_states={int(k): v for k, v in data.get("worker_states", {}).items()},
            total_cost_usd=data.get("total_cost_usd", 0.0),
            phase=data.get("phase", "starting"),
        )


class SessionManager:
    """Manages session persistence for crash recovery.

    Writes session state atomically to disk at key checkpoints so that
    the orchestrator can resume after a crash without losing progress.
    """

    def __init__(self, state_dir: str):
        self._state_dir = Path(state_dir)
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._session_path = self._state_dir / SESSION_FILE

    def save_session(self, state: SessionState) -> None:
        """Atomically save session state to disk."""
        data = state.to_dict()
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._state_dir), suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp_path, str(self._session_path))
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as e:
            logger.warning("Failed to save session state: %s", e)

    def load_session(self) -> Optional[SessionState]:
        """Load session state from disk. Returns None if no session exists."""
        if not self._session_path.exists():
            return None
        try:
            with open(self._session_path, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return None
            return SessionState.from_dict(data)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load session state: %s", e)
            return None

    def clear_session(self) -> None:
        """Remove the session file (called on clean completion)."""
        try:
            if self._session_path.exists():
                self._session_path.unlink()
        except OSError as e:
            logger.warning("Failed to clear session state: %s", e)

    def has_incomplete_session(self) -> bool:
        """Check if there's an incomplete session from a previous run."""
        state = self.load_session()
        if state is None:
            return False
        # A session is incomplete if it has tasks but not all are completed
        if not state.tasks:
            return False
        task_ids = {t.get("task_id", t.get("task_key", "")) for t in state.tasks}
        completed = set(state.completed_task_ids)
        return not task_ids.issubset(completed)

    def recover_orphaned_worktrees(self, repo_dir: str) -> List[Dict]:
        """Detect and list orphaned worktrees from a crashed session.

        Returns a list of dicts with worktree info (path, branch).
        """
        orphaned = []
        try:
            result = subprocess.run(
                ["git", "worktree", "list", "--porcelain"],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return orphaned

            current_worktree: Dict[str, str] = {}
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line:
                    if current_worktree and current_worktree.get("branch", "").startswith("auto-claude/"):
                        orphaned.append(current_worktree)
                    current_worktree = {}
                    continue
                if line.startswith("worktree "):
                    current_worktree["path"] = line[len("worktree "):]
                elif line.startswith("branch "):
                    branch = line[len("branch "):]
                    # Strip refs/heads/ prefix
                    if branch.startswith("refs/heads/"):
                        branch = branch[len("refs/heads/"):]
                    current_worktree["branch"] = branch

            # Don't forget the last entry
            if current_worktree and current_worktree.get("branch", "").startswith("auto-claude/"):
                orphaned.append(current_worktree)

        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("Failed to list worktrees: %s", e)

        return orphaned

    def create_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"session-{int(time.time())}-{os.getpid()}"
