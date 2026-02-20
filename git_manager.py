"""Git operations: snapshot, rollback, commit, changed files."""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
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
# Overall timeout for rollback operations (many per-file operations)
GIT_ROLLBACK_TIMEOUT = 300

# Retry settings for network-sensitive git operations (push, merge, rebase)
GIT_RETRY_MAX_ATTEMPTS = 3
GIT_RETRY_BASE_DELAY = 2  # seconds
GIT_RETRY_BACKOFF_FACTOR = 2  # exponential multiplier

# Stderr patterns that indicate a transient/network error worth retrying
_TRANSIENT_ERROR_PATTERNS = (
    "could not read from remote",
    "connection reset",
    "connection refused",
    "connection timed out",
    "timed out",
    "network is unreachable",
    "temporary failure",
    "name or service not known",
    "ssl",
    "unable to access",
    "the remote end hung up",
    "early eof",
    "unexpected disconnect",
    "fatal: unable to connect",
    "gnutls",
    "couldn't resolve host",
)


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

    @staticmethod
    def _is_transient_error(result: subprocess.CompletedProcess) -> bool:
        """Check if a git command failure looks like a transient network error."""
        stderr_lower = (result.stderr or "").lower()
        return any(pat in stderr_lower for pat in _TRANSIENT_ERROR_PATTERNS)

    def _run_with_retry(
        self,
        *args: str,
        check: bool = False,
        timeout: int = GIT_DEFAULT_TIMEOUT,
        max_attempts: int = GIT_RETRY_MAX_ATTEMPTS,
        base_delay: float = GIT_RETRY_BASE_DELAY,
        backoff_factor: float = GIT_RETRY_BACKOFF_FACTOR,
    ) -> subprocess.CompletedProcess:
        """Run a git command with retry and exponential backoff on transient failures.

        Only retries when the error appears to be a transient network issue.
        Non-transient failures (e.g. merge conflicts) are returned immediately.
        """
        last_result = None
        for attempt in range(max_attempts):
            result = self._run(*args, check=False, timeout=timeout)
            if result.returncode == 0:
                return result
            last_result = result
            if attempt < max_attempts - 1 and self._is_transient_error(result):
                delay = base_delay * (backoff_factor ** attempt)
                logger.warning(
                    "git %s failed with transient error (attempt %d/%d), "
                    "retrying in %.1fs: %s",
                    args[0] if args else "?",
                    attempt + 1, max_attempts, delay,
                    result.stderr.strip()[:200],
                )
                time.sleep(delay)
            else:
                break
        if check and last_result is not None and last_result.returncode != 0:
            raise subprocess.CalledProcessError(
                last_result.returncode, ["git"] + list(args),
                output=last_result.stdout, stderr=last_result.stderr,
            )
        return last_result

    def capture_worktree_state(self) -> set:
        """Return the set of currently modified/untracked files (before Claude runs)."""
        return set(self.get_changed_files())

    def create_snapshot(self) -> Snapshot:
        """Record current HEAD hash as a snapshot for potential rollback."""
        result = self._run("rev-parse", "HEAD")
        commit_hash = result.stdout.strip()
        logger.info("Snapshot created: %s", commit_hash[:8])
        return Snapshot(commit_hash=commit_hash)

    def rollback(self, snapshot: Optional[Snapshot] = None, allowed_dirty: Optional[Set[str]] = None, timeout: int = GIT_ROLLBACK_TIMEOUT) -> None:
        """Discard working tree changes, optionally targeting only specific files.

        If a snapshot is provided and HEAD has moved, reset to that commit.

        If allowed_dirty is provided:
          - Only revert files that are in the allowed set (Claude's changes).
          - If unexpected dirty files exist beyond what Claude changed,
            log a warning and refuse to clean them, preventing data loss.
        If allowed_dirty is None: blanket clean (legacy behavior).

        Args:
            timeout: Overall deadline (seconds) for the rollback operation.
                     Raises TimeoutError if exceeded during per-file operations.
        """
        deadline = time.monotonic() + timeout

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
                reverted_count = 0
                total_count = len(files_to_revert)
                for f in sorted(files_to_revert):
                    if time.monotonic() > deadline:
                        logger.warning(
                            "Rollback timeout: reverted %d/%d files before "
                            "exceeding %ds deadline",
                            reverted_count, total_count, timeout,
                        )
                        raise TimeoutError(
                            f"Rollback exceeded {timeout}s deadline: "
                            f"reverted {reverted_count}/{total_count} files"
                        )
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
                    reverted_count += 1
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
        """Push current branch to origin. Returns True on success.

        Retries with exponential backoff on transient network errors.
        """
        result = self._run_with_retry("push", timeout=GIT_PUSH_TIMEOUT)
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
        """Merge a branch into the current branch. Returns True on success.

        Retries with exponential backoff on transient network errors.
        """
        result = self._run_with_retry("merge", branch, "--no-edit")
        return result.returncode == 0

    def merge_ff_only(self, branch: str) -> bool:
        """Try a fast-forward-only merge. Returns True on success.

        Retries with exponential backoff on transient network errors.
        """
        result = self._run_with_retry("merge", "--ff-only", branch)
        return result.returncode == 0

    def abort_merge(self) -> None:
        """Abort an in-progress merge."""
        self._run("merge", "--abort", check=False)

    def rebase_onto(self, target: str, branch: str) -> bool:
        """Rebase branch onto target. Returns True on success.

        Retries with exponential backoff on transient network errors.
        """
        result = self._run_with_retry("rebase", target, branch)
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
        self._run("worktree", "prune", check=False, timeout=30)
