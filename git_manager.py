"""Git operations: snapshot, rollback, commit, changed files."""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

from process_utils import kill_process_group, run_with_group_kill

logger = logging.getLogger(__name__)

# Default timeout for git operations (seconds)
GIT_DEFAULT_TIMEOUT = 120
# Longer timeout for push operations (seconds)
GIT_PUSH_TIMEOUT = 300
# Longer timeout for commit operations (pre-commit hooks may be slow)
GIT_COMMIT_TIMEOUT = 300


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
        cmd = ["git", "rev-parse", "--git-dir"]
        result = run_with_group_kill(cmd, cwd=self.repo_dir, timeout=GIT_DEFAULT_TIMEOUT)
        if result.timed_out or result.returncode != 0:
            raise RuntimeError(f"Not a git repository: {self.repo_dir}")
        self._repo_validated = True

    def _run(self, *args: str, check: bool = True, timeout: int = GIT_DEFAULT_TIMEOUT) -> subprocess.CompletedProcess:
        """Run a git command in the repo directory.

        Uses run_with_group_kill() so that git hooks and subprocesses are
        properly killed on timeout, returning a failed result instead of
        raising subprocess.TimeoutExpired.
        """
        self._validate_repo()
        cmd = ["git"] + list(args)
        result = run_with_group_kill(cmd, cwd=self.repo_dir, timeout=timeout)
        if result.timed_out:
            logger.warning(
                "git %s timed out after %ds: %s",
                args[0] if args else "?", timeout, result.stderr.strip(),
            )
            # Return a CompletedProcess-like object with failure code
            return subprocess.CompletedProcess(
                args=cmd, returncode=-1,
                stdout=result.stdout, stderr=result.stderr,
            )
        completed = subprocess.CompletedProcess(
            args=cmd, returncode=result.returncode,
            stdout=result.stdout, stderr=result.stderr,
        )
        if check and completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode, cmd,
                output=completed.stdout, stderr=completed.stderr,
            )
        return completed

    def capture_worktree_state(self) -> set:
        """Return the set of currently modified/untracked files (before Claude runs)."""
        return set(self.get_changed_files())

    def create_snapshot(self) -> Snapshot:
        """Record current HEAD hash as a snapshot for potential rollback."""
        result = self._run("rev-parse", "HEAD")
        commit_hash = result.stdout.strip()
        logger.info("Snapshot created: %s", commit_hash[:8])
        return Snapshot(commit_hash=commit_hash)

    def rollback(self, snapshot: Optional[Snapshot] = None, allowed_dirty: Optional[Set[str]] = None) -> None:
        """Discard working tree changes, optionally targeting only specific files.

        If a snapshot is provided and HEAD has moved, reset to that commit.

        If allowed_dirty is provided:
          - Only revert files that are in the allowed set (Claude's changes).
          - If unexpected dirty files exist beyond what Claude changed,
            log a warning and refuse to clean them, preventing data loss.
        If allowed_dirty is None: blanket clean (legacy behavior).
        """
        if allowed_dirty is not None:
            current_dirty = set(self.get_changed_files())
            unexpected = current_dirty - allowed_dirty
            if unexpected:
                logger.warning(
                    "Rollback: leaving %d unexpected uncommitted files untouched: %s",
                    len(unexpected), sorted(unexpected),
                )
                raise RuntimeError(
                    f"Rollback aborted: {len(unexpected)} unexpected uncommitted files: "
                    f"{sorted(unexpected)}"
                )

            # Only revert files that are both dirty and in the allowed set
            files_to_revert = current_dirty & allowed_dirty
            if snapshot:
                current = self._run("rev-parse", "HEAD").stdout.strip()
                if current != snapshot.commit_hash:
                    self._run("reset", "--hard", snapshot.commit_hash)
                    logger.info("Reset HEAD to snapshot %s", snapshot.commit_hash[:8])

            if files_to_revert:
                # Checkout tracked files
                self._run("checkout", "--", *sorted(files_to_revert), check=False)
                # Clean untracked files in the allowed set
                for f in sorted(files_to_revert):
                    fpath = Path(self.repo_dir) / f
                    if fpath.exists():
                        # Check if it's untracked
                        status = self._run("ls-files", "--error-unmatch", f, check=False)
                        if status.returncode != 0:
                            # Untracked — remove it
                            try:
                                if fpath.is_dir():
                                    shutil.rmtree(fpath, ignore_errors=True)
                                else:
                                    fpath.unlink()
                            except OSError:
                                pass
            logger.info("Targeted rollback: reverted %d files", len(files_to_revert))
            return

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
        if files is not None and len(files) == 0:
            logger.warning("commit() called with empty file list, nothing to commit")
            return ""

        # Git has a hard limit of 65536 bytes for commit messages.
        # Truncate to prevent a hard git failure.
        GIT_MAX_COMMIT_MSG_BYTES = 65536
        msg_bytes = len(message.encode("utf-8"))
        if msg_bytes > GIT_MAX_COMMIT_MSG_BYTES:
            logger.warning(
                "Commit message too long (%d bytes, limit %d), truncating",
                msg_bytes, GIT_MAX_COMMIT_MSG_BYTES,
            )
            suffix = "\n\n[message truncated]"
            # Truncate at a safe byte boundary
            truncated = message.encode("utf-8")[:GIT_MAX_COMMIT_MSG_BYTES - len(suffix.encode("utf-8"))]
            message = truncated.decode("utf-8", errors="ignore") + suffix

        if files:
            self._run("add", "--", *files)
        else:
            self._run("add", "-A")
        # Verify something is staged
        staged = self._run("diff", "--cached", "--name-only", check=False)
        if not staged.stdout.strip():
            logger.warning("No staged changes after git add, skipping commit")
            return ""
        result = self._run("commit", "-m", message, check=False, timeout=GIT_COMMIT_TIMEOUT)
        if result.returncode != 0:
            logger.warning(
                "git commit failed (exit code %d): %s",
                result.returncode, result.stderr.strip(),
            )
            return ""
        head = self._run("rev-parse", "HEAD")
        commit_hash = head.stdout.strip()
        logger.info("Committed: %s — %s", commit_hash[:8], message.split("\n", 1)[0])
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Full commit message:\n%s", message)
        return commit_hash

    def push(self) -> bool:
        """Push current branch to origin. Returns True on success."""
        result = self._run("push", check=False, timeout=GIT_PUSH_TIMEOUT)
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

    # ------------------------------------------------------------------
    # Worktree and branch management (for parallel workers)
    # ------------------------------------------------------------------

    def create_worktree(self, path: str, branch: str) -> None:
        """Create a new worktree with a new branch at the given path."""
        self._run("worktree", "add", "-b", branch, path)

    def remove_worktree(self, path: str, force: bool = False) -> None:
        """Remove a worktree directory."""
        args = ["worktree", "remove", path]
        if force:
            args.insert(2, "--force")
        self._run(*args, check=False)

    def delete_branch(self, branch: str, force: bool = False) -> None:
        """Delete a local branch."""
        flag = "-D" if force else "-d"
        self._run("branch", flag, branch, check=False)

    def merge_branch(self, branch: str) -> bool:
        """Merge a branch into the current branch. Returns True on success."""
        result = self._run("merge", branch, "--no-edit", check=False)
        return result.returncode == 0

    def merge_ff_only(self, branch: str) -> bool:
        """Try a fast-forward-only merge. Returns True on success."""
        result = self._run("merge", "--ff-only", branch, check=False)
        return result.returncode == 0

    def abort_merge(self) -> None:
        """Abort an in-progress merge."""
        self._run("merge", "--abort", check=False)

    def rebase_onto(self, target: str, branch: str) -> bool:
        """Rebase branch onto target. Returns True on success."""
        result = self._run("rebase", target, branch, check=False)
        if result.returncode != 0:
            self._run("rebase", "--abort", check=False)
            return False
        return True

    def get_current_branch(self) -> str:
        """Get the current branch name."""
        result = self._run("rev-parse", "--abbrev-ref", "HEAD")
        return result.stdout.strip()

    def checkout(self, branch: str) -> None:
        """Checkout a branch."""
        self._run("checkout", branch)

    def prune_worktrees(self) -> None:
        """Clean up stale worktree references."""
        self._run("worktree", "prune", check=False)
