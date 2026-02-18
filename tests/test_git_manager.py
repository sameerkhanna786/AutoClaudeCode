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
