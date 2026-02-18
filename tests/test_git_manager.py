"""Tests for git_manager module."""

import subprocess
from pathlib import Path

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
