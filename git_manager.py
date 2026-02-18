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
        self._repo_validated = False

    def _validate_repo(self) -> None:
        """Validate that repo_dir is a git repository (cached after first success)."""
        if self._repo_validated:
            return
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=self.repo_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Not a git repository: {self.repo_dir}")
        self._repo_validated = True

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run a git command in the repo directory."""
        self._validate_repo()
        cmd = ["git"] + list(args)
        return subprocess.run(
            cmd,
            cwd=self.repo_dir,
            capture_output=True,
            text=True,
            check=check,
        )

    def capture_worktree_state(self) -> set:
        """Return the set of currently modified/untracked files (before Claude runs)."""
        return set(self.get_changed_files())

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

    def commit(self, message: str, files: Optional[List[str]] = None) -> str:
        """Stage specified files (or all if none given) and commit. Returns the new commit hash."""
        if files:
            self._run("add", "--", *files)
        else:
            self._run("add", "-A")
        self._run("commit", "-m", message)
        result = self._run("rev-parse", "HEAD")
        commit_hash = result.stdout.strip()
        logger.info("Committed: %s — %s", commit_hash[:8], message)
        return commit_hash

    def push(self) -> bool:
        """Push current branch to origin. Returns True on success."""
        result = self._run("push", check=False)
        if result.returncode == 0:
            logger.info("Pushed to remote")
            return True
        logger.warning("Push failed: %s", result.stderr.strip())
        return False

    def get_changed_files(self) -> List[str]:
        """Return list of changed/untracked files relative to repo root."""
        commands = [
            ("diff", "--cached", "--name-only"),
            ("diff", "--name-only"),
            ("ls-files", "--others", "--exclude-standard"),
        ]
        results = []
        any_succeeded = False
        for cmd_args in commands:
            result = self._run(*cmd_args, check=False)
            if result.returncode != 0:
                logger.warning(
                    "git %s failed (exit %d): %s",
                    cmd_args[0], result.returncode, result.stderr.strip(),
                )
            else:
                any_succeeded = True
                results.append(result.stdout)

        if not any_succeeded:
            raise RuntimeError(
                "All git commands failed in get_changed_files; "
                "cannot determine working tree state"
            )

        files = set()
        for output in results:
            for line in output.strip().split("\n"):
                if line.strip():
                    files.add(line.strip())

        return sorted(files)

    def get_new_changed_files(self, pre_existing: set) -> List[str]:
        """Return files changed since the snapshot, excluding pre-existing dirty files."""
        current = set(self.get_changed_files())
        return sorted(current - pre_existing)

    def is_clean(self) -> bool:
        """Check if the working tree is clean (no changes or untracked files)."""
        status = self._run("status", "--porcelain", check=False)
        return status.stdout.strip() == ""
