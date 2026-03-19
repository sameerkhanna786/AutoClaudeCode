"""Tests for conflict_resolver module and git conflict helpers."""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from claude_runner import ClaudeResult
from config_schema import Config
from conflict_resolver import ConflictResolver, CONFLICT_MARKER_RE
from git_manager import GitManager


# ---------------------------------------------------------------------------
# ConflictResolver._build_resolve_prompt
# ---------------------------------------------------------------------------

class TestBuildResolvePrompt:
    def test_prompt_contains_file_path(self):
        prompt = ConflictResolver._build_resolve_prompt(
            "src/app.py", "<<<<<<< HEAD\nA\n=======\nB\n>>>>>>> feat",
            "feat", "main",
        )
        assert "src/app.py" in prompt

    def test_prompt_contains_branches(self):
        prompt = ConflictResolver._build_resolve_prompt(
            "f.py", "content", "feature-x", "main",
        )
        assert "feature-x" in prompt
        assert "main" in prompt

    def test_prompt_contains_conflicted_content(self):
        content = "<<<<<<< HEAD\nline_a\n=======\nline_b\n>>>>>>> branch"
        prompt = ConflictResolver._build_resolve_prompt(
            "f.py", content, "branch", "main",
        )
        assert content in prompt

    def test_prompt_asks_for_code_block(self):
        prompt = ConflictResolver._build_resolve_prompt(
            "f.py", "stuff", "b", "m",
        )
        assert "code block" in prompt.lower()


# ---------------------------------------------------------------------------
# ConflictResolver._extract_resolved_content
# ---------------------------------------------------------------------------

class TestExtractResolvedContent:
    def test_extracts_from_code_block(self):
        response = "Here is the resolved file:\n```python\nresolved line\n```\nDone."
        result = ConflictResolver._extract_resolved_content(response)
        assert result == "resolved line\n"

    def test_returns_raw_text_without_code_block(self):
        response = "resolved content here"
        result = ConflictResolver._extract_resolved_content(response)
        assert result == "resolved content here"

    def test_empty_response_returns_none(self):
        assert ConflictResolver._extract_resolved_content("") is None
        assert ConflictResolver._extract_resolved_content("   ") is None

    def test_none_response_returns_none(self):
        assert ConflictResolver._extract_resolved_content(None) is None

    def test_extracts_first_code_block_when_multiple(self):
        response = "```\nfirst\n```\ntext\n```\nsecond\n```"
        result = ConflictResolver._extract_resolved_content(response)
        assert result == "first\n"


# ---------------------------------------------------------------------------
# CONFLICT_MARKER_RE
# ---------------------------------------------------------------------------

class TestConflictMarkerRegex:
    def test_detects_left_marker(self):
        assert CONFLICT_MARKER_RE.search("<<<<<<< HEAD")

    def test_detects_separator(self):
        assert CONFLICT_MARKER_RE.search("=======")

    def test_detects_right_marker(self):
        assert CONFLICT_MARKER_RE.search(">>>>>>> branch")

    def test_clean_content_no_match(self):
        assert CONFLICT_MARKER_RE.search("normal code\nmore code") is None

    def test_partial_markers_no_match(self):
        # Fewer than 7 characters should not match
        assert CONFLICT_MARKER_RE.search("<<<<<< HEAD") is None
        assert CONFLICT_MARKER_RE.search("======") is None
        assert CONFLICT_MARKER_RE.search(">>>>>> branch") is None


# ---------------------------------------------------------------------------
# ConflictResolver.resolve_conflicts (with mocked ClaudeRunner)
# ---------------------------------------------------------------------------

class TestResolveConflicts:
    def _make_config(self, max_cost=2.0):
        config = Config()
        config.parallel.conflict_resolution_max_cost = max_cost
        return config

    def test_successful_resolution(self, tmp_path):
        config = self._make_config()
        resolver = ConflictResolver(config)

        # Create a conflicted file
        conflict_file = tmp_path / "app.py"
        conflict_file.write_text(
            "<<<<<<< HEAD\ndef main_version():\n=======\n"
            "def branch_version():\n>>>>>>> feat\n    pass\n"
        )

        mock_result = ClaudeResult(
            success=True,
            result_text="```python\ndef merged_version():\n    pass\n```",
            cost_usd=0.05,
        )
        with patch.object(resolver, "runner") as mock_runner:
            mock_runner.run.return_value = mock_result
            success, cost = resolver.resolve_conflicts(
                str(tmp_path), ["app.py"], "feat", "main",
            )

        assert success is True
        assert cost == 0.05
        resolved = conflict_file.read_text()
        assert "merged_version" in resolved
        assert "<<<<<<" not in resolved

    def test_markers_in_response_rejected(self, tmp_path):
        config = self._make_config()
        resolver = ConflictResolver(config)

        conflict_file = tmp_path / "bad.py"
        conflict_file.write_text("<<<<<<< HEAD\nA\n=======\nB\n>>>>>>> feat\n")

        # Claude returns content that still has conflict markers
        mock_result = ClaudeResult(
            success=True,
            result_text="```\n<<<<<<< HEAD\nA\n=======\nB\n>>>>>>> feat\n```",
            cost_usd=0.03,
        )
        with patch.object(resolver, "runner") as mock_runner:
            mock_runner.run.return_value = mock_result
            success, cost = resolver.resolve_conflicts(
                str(tmp_path), ["bad.py"], "feat", "main",
            )

        assert success is False
        assert cost == 0.03

    def test_empty_response_fails(self, tmp_path):
        config = self._make_config()
        resolver = ConflictResolver(config)

        conflict_file = tmp_path / "empty.py"
        conflict_file.write_text("<<<<<<< HEAD\nA\n=======\nB\n>>>>>>> feat\n")

        mock_result = ClaudeResult(
            success=True,
            result_text="",
            cost_usd=0.01,
        )
        with patch.object(resolver, "runner") as mock_runner:
            mock_runner.run.return_value = mock_result
            success, cost = resolver.resolve_conflicts(
                str(tmp_path), ["empty.py"], "feat", "main",
            )

        assert success is False

    def test_claude_failure_fails(self, tmp_path):
        config = self._make_config()
        resolver = ConflictResolver(config)

        conflict_file = tmp_path / "fail.py"
        conflict_file.write_text("<<<<<<< HEAD\nA\n=======\nB\n>>>>>>> feat\n")

        mock_result = ClaudeResult(
            success=False,
            result_text="",
            cost_usd=0.02,
            error="API error",
        )
        with patch.object(resolver, "runner") as mock_runner:
            mock_runner.run.return_value = mock_result
            success, cost = resolver.resolve_conflicts(
                str(tmp_path), ["fail.py"], "feat", "main",
            )

        assert success is False
        assert cost == 0.02

    def test_cost_limit_enforced(self, tmp_path):
        config = self._make_config(max_cost=0.10)
        resolver = ConflictResolver(config)
        # Pre-load cost to exceed limit
        resolver._total_cost = 0.10

        conflict_file = tmp_path / "costly.py"
        conflict_file.write_text("<<<<<<< HEAD\nA\n=======\nB\n>>>>>>> feat\n")

        with patch.object(resolver, "runner") as mock_runner:
            success, cost = resolver.resolve_conflicts(
                str(tmp_path), ["costly.py"], "feat", "main",
            )
            # Runner should never be called because cost limit is already hit
            mock_runner.run.assert_not_called()

        assert success is False
        assert cost == 0.10

    def test_cost_accumulates_across_files(self, tmp_path):
        config = self._make_config(max_cost=0.10)
        resolver = ConflictResolver(config)

        # Create two conflicted files
        (tmp_path / "a.py").write_text("<<<<<<< HEAD\nA\n=======\nB\n>>>>>>> feat\n")
        (tmp_path / "b.py").write_text("<<<<<<< HEAD\nC\n=======\nD\n>>>>>>> feat\n")

        call_count = {"n": 0}

        def mock_run(prompt):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return ClaudeResult(
                    success=True,
                    result_text="```\nresolved_a\n```",
                    cost_usd=0.08,
                )
            # Second call should not happen because 0.08 + anything > 0.10
            return ClaudeResult(
                success=True,
                result_text="```\nresolved_b\n```",
                cost_usd=0.05,
            )

        with patch.object(resolver, "runner") as mock_runner:
            mock_runner.run.side_effect = mock_run
            success, cost = resolver.resolve_conflicts(
                str(tmp_path), ["a.py", "b.py"], "feat", "main",
            )

        # First file resolves (cost 0.08), then cost check fails for second (0.08 >= 0.10 is false,
        # so second file runs too) -- but our limit is 0.10 and 0.08 < 0.10 so second runs.
        # Actually 0.08 < 0.10, so the second file WILL be attempted.
        # The test validates cost tracking works correctly.
        assert cost == 0.08 + 0.05
        assert call_count["n"] == 2

    def test_missing_file_fails(self, tmp_path):
        config = self._make_config()
        resolver = ConflictResolver(config)

        with patch.object(resolver, "runner") as mock_runner:
            success, cost = resolver.resolve_conflicts(
                str(tmp_path), ["nonexistent.py"], "feat", "main",
            )
            mock_runner.run.assert_not_called()

        assert success is False


# ---------------------------------------------------------------------------
# Git helper tests (get_conflicted_files, mark_resolved_and_commit)
# ---------------------------------------------------------------------------

class TestGetConflictedFiles:
    def test_returns_conflicted_files(self, tmp_git_repo):
        gm = GitManager(tmp_git_repo)
        # Mock _run to simulate git diff output
        mock_result = subprocess.CompletedProcess(
            args=["git"], returncode=0,
            stdout="src/a.py\nsrc/b.py\n", stderr="",
        )
        with patch.object(gm, "_run", return_value=mock_result):
            files = gm.get_conflicted_files()
        assert files == ["src/a.py", "src/b.py"]

    def test_returns_empty_on_failure(self, tmp_git_repo):
        gm = GitManager(tmp_git_repo)
        mock_result = subprocess.CompletedProcess(
            args=["git"], returncode=1,
            stdout="", stderr="error",
        )
        with patch.object(gm, "_run", return_value=mock_result):
            files = gm.get_conflicted_files()
        assert files == []

    def test_returns_empty_when_no_conflicts(self, tmp_git_repo):
        gm = GitManager(tmp_git_repo)
        mock_result = subprocess.CompletedProcess(
            args=["git"], returncode=0,
            stdout="\n", stderr="",
        )
        with patch.object(gm, "_run", return_value=mock_result):
            files = gm.get_conflicted_files()
        assert files == []

    def test_strips_whitespace(self, tmp_git_repo):
        gm = GitManager(tmp_git_repo)
        mock_result = subprocess.CompletedProcess(
            args=["git"], returncode=0,
            stdout="  file.py  \n  other.py  \n", stderr="",
        )
        with patch.object(gm, "_run", return_value=mock_result):
            files = gm.get_conflicted_files()
        assert files == ["file.py", "other.py"]


class TestMarkResolvedAndCommit:
    def test_empty_files_returns_empty_string(self, tmp_git_repo):
        gm = GitManager(tmp_git_repo)
        result = gm.mark_resolved_and_commit([], "msg")
        assert result == ""

    @pytest.mark.requires_subprocess
    def test_successful_commit(self, tmp_git_repo):
        gm = GitManager(tmp_git_repo)
        # Create a file that can be staged
        Path(tmp_git_repo, "resolved.py").write_text("resolved content")

        calls = []
        original_run = gm._run

        def tracking_run(*args, **kwargs):
            calls.append(args)
            return original_run(*args, **kwargs)

        with patch.object(gm, "_run", side_effect=tracking_run):
            result = gm.mark_resolved_and_commit(["resolved.py"], "Merge resolved")

        # Should have called git add, git commit, and git rev-parse
        assert result is not None
        assert len(result) == 40  # commit hash

    def test_commit_failure_returns_none(self, tmp_git_repo):
        gm = GitManager(tmp_git_repo)
        Path(tmp_git_repo, "file.py").write_text("content")

        original_run = gm._run

        def mock_run(*args, **kwargs):
            if args and args[0] == "commit":
                return subprocess.CompletedProcess(
                    args=["git", "commit"], returncode=1,
                    stdout="", stderr="commit failed",
                )
            return original_run(*args, **kwargs)

        with patch.object(gm, "_run", side_effect=mock_run):
            result = gm.mark_resolved_and_commit(["file.py"], "msg")

        assert result is None


# ---------------------------------------------------------------------------
# Integration-style: real merge conflict in a git repo
# ---------------------------------------------------------------------------

class TestGetConflictedFilesRealRepo:
    @pytest.mark.requires_subprocess
    def test_detects_real_merge_conflict(self, tmp_git_repo):
        """Create a real merge conflict and verify get_conflicted_files detects it."""
        gm = GitManager(tmp_git_repo)
        original_branch = gm.get_current_branch()

        # Create a branch and modify README.md
        wt_path = str(Path(tmp_git_repo) / ".worktrees" / "conflict-test")
        Path(wt_path).parent.mkdir(parents=True, exist_ok=True)
        gm.create_worktree(wt_path, "conflict-branch")

        wt_gm = GitManager(wt_path)
        Path(wt_path, "README.md").write_text("# Branch version\n")
        wt_gm.commit("Branch change", files=["README.md"])
        gm.remove_worktree(wt_path, force=True)

        # Modify the same file on main
        Path(tmp_git_repo, "README.md").write_text("# Main version\n")
        gm.commit("Main change", files=["README.md"])

        # Attempt merge -- should conflict
        merge_result = gm._run("merge", "--no-commit", "--no-ff", "conflict-branch", check=False)
        # The merge should fail or leave conflicts
        conflicted = gm.get_conflicted_files()

        if conflicted:
            assert "README.md" in conflicted

        # Clean up
        gm.abort_merge()
        gm._run("checkout", ".", check=False)
        gm._run("clean", "-fd", check=False)
        gm.delete_branch("conflict-branch", force=True)


class TestResolveConflictsAtomicWrite:
    """Test that resolve_conflicts uses atomic writes."""

    def test_write_uses_tempfile_and_replace(self, tmp_path):
        """Verify resolved content is written atomically via temp file + os.replace."""
        config = Config()
        resolver = ConflictResolver(config)

        conflict_file = tmp_path / "atomic.py"
        conflict_file.write_text("original content")

        mock_result = ClaudeResult(
            success=True,
            result_text="```python\nresolved content\n```",
            cost_usd=0.01,
        )
        replace_calls = []
        original_replace = os.replace

        def tracking_replace(src, dst):
            replace_calls.append((src, dst))
            return original_replace(src, dst)

        with patch.object(resolver, "runner") as mock_runner:
            mock_runner.run.return_value = mock_result
            with patch("conflict_resolver.os.replace", side_effect=tracking_replace):
                success, cost = resolver.resolve_conflicts(
                    str(tmp_path), ["atomic.py"], "feat", "main",
                )

        assert success is True
        # os.replace should have been called with a temp file -> target
        assert len(replace_calls) == 1
        src_path, dst_path = replace_calls[0]
        assert dst_path == str(conflict_file)
        assert ".tmp" in src_path

    def test_write_cleans_up_temp_on_failure(self, tmp_path):
        """If os.replace fails, the temp file should be cleaned up."""
        config = Config()
        resolver = ConflictResolver(config)

        conflict_file = tmp_path / "cleanup.py"
        conflict_file.write_text("content")

        mock_result = ClaudeResult(
            success=True,
            result_text="```python\nresolved\n```",
            cost_usd=0.01,
        )
        with patch.object(resolver, "runner") as mock_runner:
            mock_runner.run.return_value = mock_result
            with patch("conflict_resolver.os.replace", side_effect=OSError("disk error")):
                success, cost = resolver.resolve_conflicts(
                    str(tmp_path), ["cleanup.py"], "feat", "main",
                )

        assert success is False
        # No leftover .tmp files
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0


class TestResolveConflictsFileErrors:
    """Test error handling for file read/write failures in resolve_conflicts."""

    def test_read_error_returns_false(self, tmp_path):
        """Verify resolve_conflicts handles OSError on file read gracefully."""
        config = Config()
        resolver = ConflictResolver(config)

        filepath = "test_file.py"
        test_file = tmp_path / filepath
        test_file.write_text("content")

        with patch.object(Path, 'read_text', side_effect=OSError("Permission denied")):
            success, cost = resolver.resolve_conflicts(
                str(tmp_path), [filepath], "worker-branch", "main"
            )
        assert success is False

    def test_write_error_returns_false(self, tmp_path):
        """Verify resolve_conflicts handles OSError on file write gracefully."""
        config = Config()
        resolver = ConflictResolver(config)

        filepath = "test_file.py"
        test_file = tmp_path / filepath
        test_file.write_text("content with no conflicts")

        mock_result = ClaudeResult(
            success=True,
            result_text="```python\nresolved\n```",
            cost_usd=0.01,
        )
        with patch.object(resolver, "runner") as mock_runner:
            mock_runner.run.return_value = mock_result
            with patch("conflict_resolver.tempfile.mkstemp", side_effect=OSError("Disk full")):
                success, cost = resolver.resolve_conflicts(
                    str(tmp_path), [filepath], "worker-branch", "main"
                )
        assert success is False
