"""Tests for git_manager module."""

import logging
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from git_manager import GitManager, Snapshot


class TestGitManager:
    def test_create_snapshot(self, tmp_git_repo):
        gm = GitManager(tmp_git_repo)
        snap = gm.create_snapshot()
        assert len(snap.commit_hash) == 40
        assert snap.commit_hash.isalnum()

    def test_is_clean_on_fresh_repo(self, tmp_git_repo):
        gm = GitManager(tmp_git_repo)
        assert gm.is_clean() is True

    def test_is_clean_after_modification(self, tmp_git_repo):
        gm = GitManager(tmp_git_repo)
        Path(tmp_git_repo, "new_file.txt").write_text("hello")
        assert gm.is_clean() is False

    def test_get_changed_files(self, tmp_git_repo):
        gm = GitManager(tmp_git_repo)
        Path(tmp_git_repo, "a.txt").write_text("a")
        Path(tmp_git_repo, "b.txt").write_text("b")
        changed = gm.get_changed_files()
        assert "a.txt" in changed
        assert "b.txt" in changed

    def test_get_changed_files_empty_when_clean(self, tmp_git_repo):
        gm = GitManager(tmp_git_repo)
        changed = gm.get_changed_files()
        assert changed == []

    def test_commit(self, tmp_git_repo):
        gm = GitManager(tmp_git_repo)
        Path(tmp_git_repo, "feature.py").write_text("# feature\n")
        commit_hash = gm.commit("[auto] feat: add feature")
        assert len(commit_hash) == 40
        assert gm.is_clean() is True

    def test_rollback_discards_changes(self, tmp_git_repo):
        gm = GitManager(tmp_git_repo)
        snap = gm.create_snapshot()
        Path(tmp_git_repo, "bad_file.txt").write_text("oops")
        Path(tmp_git_repo, "README.md").write_text("modified")
        gm.rollback(snap)
        assert gm.is_clean() is True
        assert not Path(tmp_git_repo, "bad_file.txt").exists()
        assert Path(tmp_git_repo, "README.md").read_text() == "# Test\n"

    def test_rollback_without_snapshot(self, tmp_git_repo):
        gm = GitManager(tmp_git_repo)
        Path(tmp_git_repo, "junk.txt").write_text("junk")
        # Modify tracked file
        Path(tmp_git_repo, "README.md").write_text("changed")
        gm.rollback()
        assert gm.is_clean() is True

    def test_commit_only_specified_files(self, tmp_git_repo):
        gm = GitManager(tmp_git_repo)
        Path(tmp_git_repo, "wanted.txt").write_text("keep me")
        Path(tmp_git_repo, "unwanted.txt").write_text("leave me alone")
        commit_hash = gm.commit("commit only wanted", files=["wanted.txt"])
        assert len(commit_hash) == 40
        # unwanted.txt should still be untracked
        changed = gm.get_changed_files()
        assert "unwanted.txt" in changed
        assert "wanted.txt" not in changed

    def test_get_new_changed_files(self, tmp_git_repo):
        gm = GitManager(tmp_git_repo)
        # Create a pre-existing untracked file
        Path(tmp_git_repo, "pre_existing.txt").write_text("old stuff")
        pre_existing = gm.capture_worktree_state()
        assert "pre_existing.txt" in pre_existing
        # Simulate Claude creating a new file
        Path(tmp_git_repo, "new_file.txt").write_text("claude wrote this")
        new_files = gm.get_new_changed_files(pre_existing)
        assert "new_file.txt" in new_files
        assert "pre_existing.txt" not in new_files

    def test_commit_all_when_no_files_specified(self, tmp_git_repo):
        gm = GitManager(tmp_git_repo)
        Path(tmp_git_repo, "file_a.txt").write_text("a")
        Path(tmp_git_repo, "file_b.txt").write_text("b")
        commit_hash = gm.commit("commit everything")
        assert len(commit_hash) == 40
        assert gm.is_clean() is True


class TestGetChangedFilesErrorHandling:
    def test_get_changed_files_raises_on_git_failure(self, tmp_git_repo):
        """When all git commands fail, get_changed_files should raise RuntimeError."""
        from unittest.mock import patch, MagicMock
        import subprocess

        gm = GitManager(tmp_git_repo)
        failed = subprocess.CompletedProcess(
            args=["git"], returncode=128, stdout="", stderr="fatal: not a git repository"
        )
        with patch.object(gm, "_run", return_value=failed):
            with pytest.raises(RuntimeError, match="All git commands failed"):
                gm.get_changed_files()

    def test_get_changed_files_warns_on_partial_failure(self, tmp_git_repo, caplog):
        """When one git command fails but others succeed, result is still returned with a warning."""
        import logging
        from unittest.mock import patch, MagicMock
        import subprocess

        gm = GitManager(tmp_git_repo)
        # Create a file so there's something to detect
        Path(tmp_git_repo, "test_file.txt").write_text("content")

        call_count = 0
        original_run = gm._run

        def partial_fail(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First command (diff --cached) fails
                return subprocess.CompletedProcess(
                    args=["git"], returncode=128, stdout="", stderr="simulated failure"
                )
            return original_run(*args, **kwargs)

        with patch.object(gm, "_run", side_effect=partial_fail):
            with caplog.at_level(logging.WARNING):
                result = gm.get_changed_files()

        # Should still return files from the successful commands
        assert "test_file.txt" in result
        # Should have logged a warning about the failed command
        assert any("failed" in r.message for r in caplog.records)


class TestCommitEmptyChanges:
    def test_commit_empty_file_list(self, tmp_git_repo, caplog):
        """commit() with files=[] should return empty string without error."""
        import logging
        gm = GitManager(tmp_git_repo)
        with caplog.at_level(logging.WARNING):
            result = gm.commit("empty commit", files=[])
        assert result == ""
        assert any("empty file list" in r.message for r in caplog.records)

    def test_commit_no_staged_changes(self, tmp_git_repo, caplog):
        """commit() with no actual staged changes should return empty string."""
        import logging
        gm = GitManager(tmp_git_repo)
        # README.md is already committed, staging it again won't create a diff
        with caplog.at_level(logging.WARNING):
            result = gm.commit("nothing to commit", files=["README.md"])
        assert result == ""
        assert any("No staged changes" in r.message for r in caplog.records)


class TestCommitTimeout:
    def test_commit_timeout_returns_empty_string(self, tmp_git_repo, caplog):
        """When git commit times out (returncode -1), commit() returns empty string."""
        import logging
        from unittest.mock import patch, call
        gm = GitManager(tmp_git_repo)
        Path(tmp_git_repo, "timeout_test.txt").write_text("test content")

        original_run = gm._run

        def mock_run(*args, **kwargs):
            # Intercept the "commit" call and simulate a timeout
            if args and args[0] == "commit":
                return subprocess.CompletedProcess(
                    args=["git", "commit"], returncode=-1,
                    stdout="", stderr="timed out",
                )
            return original_run(*args, **kwargs)

        with patch.object(gm, "_run", side_effect=mock_run):
            with caplog.at_level(logging.WARNING):
                result = gm.commit("this should timeout", files=["timeout_test.txt"])

        assert result == ""
        assert any("git commit failed" in r.message for r in caplog.records)

    def test_commit_nonzero_exit_returns_empty_string(self, tmp_git_repo, caplog):
        """When git commit fails with nonzero exit, commit() returns empty string."""
        import logging
        from unittest.mock import patch
        gm = GitManager(tmp_git_repo)
        Path(tmp_git_repo, "fail_test.txt").write_text("test content")

        original_run = gm._run

        def mock_run(*args, **kwargs):
            if args and args[0] == "commit":
                return subprocess.CompletedProcess(
                    args=["git", "commit"], returncode=1,
                    stdout="", stderr="pre-commit hook failed",
                )
            return original_run(*args, **kwargs)

        with patch.object(gm, "_run", side_effect=mock_run):
            with caplog.at_level(logging.WARNING):
                result = gm.commit("hook should fail", files=["fail_test.txt"])

        assert result == ""
        assert any("git commit failed" in r.message for r in caplog.records)


class TestRollbackSafety:
    def test_rollback_raises_on_unexpected_dirty_files(self, tmp_git_repo):
        """Rollback should raise RuntimeError when unexpected dirty files exist."""
        gm = GitManager(tmp_git_repo)
        snap = gm.create_snapshot()
        # Create unexpected uncommitted files
        Path(tmp_git_repo, "unexpected.txt").write_text("surprise")
        Path(tmp_git_repo, "also_unexpected.txt").write_text("another surprise")
        with pytest.raises(RuntimeError, match="unexpected uncommitted files"):
            gm.rollback(snap, allowed_dirty=set())

    def test_rollback_with_allowed_dirty_succeeds(self, tmp_git_repo):
        """Rollback should succeed when all dirty files are in the allowed set."""
        gm = GitManager(tmp_git_repo)
        snap = gm.create_snapshot()
        Path(tmp_git_repo, "expected.txt").write_text("expected change")
        gm.rollback(snap, allowed_dirty={"expected.txt"})
        assert gm.is_clean() is True

    def test_rollback_without_allowed_dirty_backward_compat(self, tmp_git_repo):
        """Rollback without allowed_dirty param should behave as before."""
        gm = GitManager(tmp_git_repo)
        snap = gm.create_snapshot()
        Path(tmp_git_repo, "anything.txt").write_text("any content")
        gm.rollback(snap)
        assert gm.is_clean() is True
        assert not Path(tmp_git_repo, "anything.txt").exists()


class TestCommitMessageLengthValidation:
    def test_commit_truncates_oversized_message(self, tmp_git_repo, caplog):
        """Commit messages exceeding 65536 bytes are truncated."""
        gm = GitManager(tmp_git_repo)
        Path(tmp_git_repo, "file.txt").write_text("content")
        # Create a message well over 65536 bytes
        oversized_message = "A" * 70000
        with caplog.at_level(logging.WARNING):
            commit_hash = gm.commit(oversized_message, files=["file.txt"])
        assert len(commit_hash) == 40
        assert any("too long" in r.message for r in caplog.records)
        # Verify the commit actually succeeded
        assert gm.is_clean() is True

    def test_commit_accepts_normal_message(self, tmp_git_repo):
        """Messages under 65536 bytes work normally."""
        gm = GitManager(tmp_git_repo)
        Path(tmp_git_repo, "file.txt").write_text("content")
        commit_hash = gm.commit("Normal length message", files=["file.txt"])
        assert len(commit_hash) == 40


class TestWorktreeManagement:
    def test_create_and_remove_worktree(self, tmp_git_repo):
        """Can create and remove a worktree."""
        gm = GitManager(tmp_git_repo)
        wt_path = str(Path(tmp_git_repo) / ".worktrees" / "test-wt")
        Path(wt_path).parent.mkdir(parents=True, exist_ok=True)

        gm.create_worktree(wt_path, "test-branch")
        assert Path(wt_path).exists()
        assert Path(wt_path, "README.md").exists()

        gm.remove_worktree(wt_path, force=True)
        assert not Path(wt_path).exists()

        gm.delete_branch("test-branch", force=True)

    def test_get_current_branch(self, tmp_git_repo):
        """Can get the current branch name."""
        gm = GitManager(tmp_git_repo)
        # In a new git repo the branch is typically "main" or "master"
        branch = gm.get_current_branch()
        assert branch in ("main", "master")

    def test_checkout_branch(self, tmp_git_repo):
        """Can checkout between branches."""
        gm = GitManager(tmp_git_repo)
        original = gm.get_current_branch()

        # Create a worktree with a new branch, then remove it
        wt_path = str(Path(tmp_git_repo) / ".worktrees" / "checkout-test")
        Path(wt_path).parent.mkdir(parents=True, exist_ok=True)
        gm.create_worktree(wt_path, "checkout-branch")
        gm.remove_worktree(wt_path, force=True)

        # Checkout the new branch
        gm.checkout("checkout-branch")
        assert gm.get_current_branch() == "checkout-branch"

        # Return to original
        gm.checkout(original)
        assert gm.get_current_branch() == original

        gm.delete_branch("checkout-branch", force=True)

    def test_merge_ff_only(self, tmp_git_repo):
        """Fast-forward merge works when possible."""
        gm = GitManager(tmp_git_repo)
        original = gm.get_current_branch()

        # Create worktree, add commit
        wt_path = str(Path(tmp_git_repo) / ".worktrees" / "ff-test")
        Path(wt_path).parent.mkdir(parents=True, exist_ok=True)
        gm.create_worktree(wt_path, "ff-branch")

        wt_gm = GitManager(wt_path)
        Path(wt_path, "ff_file.txt").write_text("fast forward")
        wt_gm.commit("Add ff file", files=["ff_file.txt"])

        gm.remove_worktree(wt_path, force=True)

        # FF merge should succeed (main hasn't moved)
        assert gm.merge_ff_only("ff-branch") is True
        assert Path(tmp_git_repo, "ff_file.txt").exists()

        gm.delete_branch("ff-branch", force=True)

    def test_merge_ff_only_fails_on_diverged(self, tmp_git_repo):
        """Fast-forward merge fails when branches have diverged."""
        gm = GitManager(tmp_git_repo)

        # Create worktree with a commit
        wt_path = str(Path(tmp_git_repo) / ".worktrees" / "div-test")
        Path(wt_path).parent.mkdir(parents=True, exist_ok=True)
        gm.create_worktree(wt_path, "div-branch")

        wt_gm = GitManager(wt_path)
        Path(wt_path, "branch_file.txt").write_text("branch")
        wt_gm.commit("Branch commit", files=["branch_file.txt"])
        gm.remove_worktree(wt_path, force=True)

        # Add a commit to main too (diverge)
        Path(tmp_git_repo, "main_file.txt").write_text("main")
        gm.commit("Main commit", files=["main_file.txt"])

        # FF merge should fail
        assert gm.merge_ff_only("div-branch") is False

        gm.delete_branch("div-branch", force=True)

    def test_merge_branch_with_no_conflicts(self, tmp_git_repo):
        """Auto-merge succeeds when there are no conflicts."""
        gm = GitManager(tmp_git_repo)

        # Create branch with commit on a different file
        wt_path = str(Path(tmp_git_repo) / ".worktrees" / "merge-test")
        Path(wt_path).parent.mkdir(parents=True, exist_ok=True)
        gm.create_worktree(wt_path, "merge-branch")

        wt_gm = GitManager(wt_path)
        Path(wt_path, "branch_only.txt").write_text("from branch")
        wt_gm.commit("Branch file", files=["branch_only.txt"])
        gm.remove_worktree(wt_path, force=True)

        # Commit different file on main
        Path(tmp_git_repo, "main_only.txt").write_text("from main")
        gm.commit("Main file", files=["main_only.txt"])

        # Merge should succeed
        assert gm.merge_branch("merge-branch") is True
        assert Path(tmp_git_repo, "branch_only.txt").exists()
        assert Path(tmp_git_repo, "main_only.txt").exists()

        gm.delete_branch("merge-branch", force=True)

    def test_prune_worktrees(self, tmp_git_repo):
        """prune_worktrees doesn't error on a clean repo."""
        gm = GitManager(tmp_git_repo)
        gm.prune_worktrees()  # Should not raise

    def test_delete_branch_nonexistent(self, tmp_git_repo):
        """Deleting a nonexistent branch doesn't raise (check=False)."""
        gm = GitManager(tmp_git_repo)
        gm.delete_branch("nonexistent-branch")  # Should not raise


class TestRollbackTimeout:
    def test_rollback_raises_timeout_when_deadline_exceeded(self, tmp_git_repo):
        """Rollback with many files should raise TimeoutError when deadline is exceeded."""
        import time as time_mod
        gm = GitManager(tmp_git_repo)
        snap = gm.create_snapshot()

        # Create several files to revert
        for i in range(5):
            Path(tmp_git_repo, f"file_{i}.txt").write_text(f"content {i}")

        allowed = {f"file_{i}.txt" for i in range(5)}

        # Mock time.monotonic to simulate deadline expiration after first file
        call_count = {"n": 0}
        original_monotonic = time_mod.monotonic

        def advancing_monotonic():
            nonlocal call_count
            call_count["n"] += 1
            # First call (deadline calculation) returns 0
            # Second call (first check in loop) returns well past deadline
            if call_count["n"] <= 1:
                return 0.0
            return 1000.0  # Way past any deadline

        with patch("git_manager.time.monotonic", side_effect=advancing_monotonic):
            with pytest.raises(TimeoutError, match="exceeded.*deadline"):
                gm.rollback(snap, allowed_dirty=allowed, timeout=10)

    def test_rollback_completes_within_timeout(self, tmp_git_repo):
        """Normal rollback with default timeout succeeds without error."""
        gm = GitManager(tmp_git_repo)
        snap = gm.create_snapshot()
        Path(tmp_git_repo, "file_a.txt").write_text("a")
        Path(tmp_git_repo, "file_b.txt").write_text("b")
        allowed = {"file_a.txt", "file_b.txt"}

        gm.rollback(snap, allowed_dirty=allowed)
        assert gm.is_clean() is True
