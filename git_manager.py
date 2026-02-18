"""Git operations: snapshot, rollback, commit, changed files."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Snapshot:
    commit_hash: str


class GitManager:
    def __init__(self, repo_dir: str):
        self.repo_dir = repo_dir

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run a git command in the repo directory."""
        cmd = ["git"] + list(args)
        return subprocess.run(
            cmd,
            cwd=self.repo_dir,
            capture_output=True,
            text=True,
            check=check,
        )

    def create_snapshot(self) -> Snapshot:
        """Record current HEAD hash as a snapshot for potential rollback."""
        result = self._run("rev-parse", "HEAD")
        commit_hash = result.stdout.strip()
        logger.info("Snapshot created: %s", commit_hash[:8])
        return Snapshot(commit_hash=commit_hash)

    def rollback(self, snapshot: Optional[Snapshot] = None) -> None:
        """Discard all working tree changes and untracked files.

        If a snapshot is provided and HEAD has moved, reset to that commit.
        Otherwise just clean the working tree.
        """
        if snapshot:
            current = self._run("rev-parse", "HEAD").stdout.strip()
            if current != snapshot.commit_hash:
                self._run("reset", "--hard", snapshot.commit_hash)
                logger.info("Reset HEAD to snapshot %s", snapshot.commit_hash[:8])

        self._run("checkout", ".")
        self._run("clean", "-fd")
        logger.info("Working tree cleaned")

    def commit(self, message: str) -> str:
        """Stage all changes and commit. Returns the new commit hash."""
        self._run("add", "-A")
        self._run("commit", "-m", message)
        result = self._run("rev-parse", "HEAD")
        commit_hash = result.stdout.strip()
        logger.info("Committed: %s — %s", commit_hash[:8], message)
        return commit_hash

    def get_changed_files(self) -> List[str]:
        """Return list of changed/untracked files relative to repo root."""
        # Staged changes
        staged = self._run("diff", "--cached", "--name-only", check=False)
        # Unstaged changes
        unstaged = self._run("diff", "--name-only", check=False)
        # Untracked files
        untracked = self._run("ls-files", "--others", "--exclude-standard", check=False)

        files = set()
        for output in [staged.stdout, unstaged.stdout, untracked.stdout]:
            for line in output.strip().split("\n"):
                if line.strip():
                    files.add(line.strip())

        return sorted(files)

    def is_clean(self) -> bool:
        """Check if the working tree is clean (no changes or untracked files)."""
        status = self._run("status", "--porcelain", check=False)
        return status.stdout.strip() == ""
