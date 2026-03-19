"""Tests for feedback module."""

import logging
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from config_schema import Config
from feedback import FeedbackManager
from task_discovery import Task


@pytest.fixture
def fb_mgr(tmp_path, default_config):
    default_config.paths.feedback_dir = str(tmp_path / "feedback")
    default_config.paths.feedback_done_dir = str(tmp_path / "feedback" / "done")
    default_config.paths.feedback_failed_dir = str(tmp_path / "feedback" / "failed")
    return FeedbackManager(default_config)


class TestFeedbackManager:
    def test_no_feedback_files(self, fb_mgr):
        tasks = fb_mgr.get_pending_feedback()
        assert tasks == []

    def test_read_single_feedback(self, fb_mgr):
        fb_dir = Path(fb_mgr.feedback_dir)
        (fb_dir / "fix-bug.md").write_text("Fix the login bug in auth.py")
        tasks = fb_mgr.get_pending_feedback()
        assert len(tasks) == 1
        assert tasks[0].description == "Fix the login bug in auth.py"
        assert tasks[0].priority == 1
        assert tasks[0].source == "feedback"

    def test_priority_from_filename(self, fb_mgr):
        fb_dir = Path(fb_mgr.feedback_dir)
        (fb_dir / "03-low-priority.md").write_text("Low priority task")
        (fb_dir / "01-high-priority.md").write_text("High priority task")
        tasks = fb_mgr.get_pending_feedback()
        assert len(tasks) == 2
        # Should be sorted by filename (01 before 03)
        assert tasks[0].priority == 1
        assert tasks[1].priority == 3

    def test_skip_empty_files(self, fb_mgr):
        fb_dir = Path(fb_mgr.feedback_dir)
        (fb_dir / "empty.md").write_text("")
        (fb_dir / "real.md").write_text("Do something")
        tasks = fb_mgr.get_pending_feedback()
        assert len(tasks) == 1

    def test_skip_non_md_txt_files(self, fb_mgr):
        fb_dir = Path(fb_mgr.feedback_dir)
        (fb_dir / "notes.py").write_text("not a task")
        (fb_dir / "task.txt").write_text("A real task")
        tasks = fb_mgr.get_pending_feedback()
        assert len(tasks) == 1
        assert tasks[0].description == "A real task"

    def test_skip_gitkeep(self, fb_mgr):
        fb_dir = Path(fb_mgr.feedback_dir)
        (fb_dir / ".gitkeep").write_text("")
        tasks = fb_mgr.get_pending_feedback()
        assert tasks == []

    def test_mark_done(self, fb_mgr):
        fb_dir = Path(fb_mgr.feedback_dir)
        done_dir = Path(fb_mgr.done_dir)
        task_file = fb_dir / "fix.md"
        task_file.write_text("Fix it")

        fb_mgr.mark_done(str(task_file))
        assert not task_file.exists()
        assert (done_dir / "fix.md").exists()

    def test_mark_done_avoids_overwrite(self, fb_mgr):
        fb_dir = Path(fb_mgr.feedback_dir)
        done_dir = Path(fb_mgr.done_dir)

        # Put a file in done already
        (done_dir / "fix.md").write_text("old")

        task_file = fb_dir / "fix.md"
        task_file.write_text("new")

        fb_mgr.mark_done(str(task_file))
        assert not task_file.exists()
        # Should have created fix_1.md
        assert (done_dir / "fix_1.md").exists()

    def test_mark_done_nonexistent(self, fb_mgr):
        # Should not raise
        fb_mgr.mark_done("/nonexistent/file.md")

    def test_extract_priority_default(self, fb_mgr):
        assert fb_mgr._extract_priority("task.md") == 1

    def test_extract_priority_with_number(self, fb_mgr):
        assert fb_mgr._extract_priority("05-task.md") == 5
        assert fb_mgr._extract_priority("1-urgent.txt") == 1

    def test_mark_failed(self, fb_mgr):
        fb_dir = Path(fb_mgr.feedback_dir)
        failed_dir = Path(fb_mgr.failed_dir)
        task_file = fb_dir / "broken.md"
        task_file.write_text("Broken task")

        fb_mgr.mark_failed(str(task_file))
        assert not task_file.exists()
        assert (failed_dir / "broken.md").exists()

    def test_mark_failed_avoids_overwrite(self, fb_mgr):
        fb_dir = Path(fb_mgr.feedback_dir)
        failed_dir = Path(fb_mgr.failed_dir)

        # Put a file in failed already
        (failed_dir / "broken.md").write_text("old")

        task_file = fb_dir / "broken.md"
        task_file.write_text("new")

        fb_mgr.mark_failed(str(task_file))
        assert not task_file.exists()
        assert (failed_dir / "broken_1.md").exists()

    def test_mark_failed_nonexistent(self, fb_mgr):
        # Should not raise
        fb_mgr.mark_failed("/nonexistent/file.md")

    def test_failed_dir_created(self, fb_mgr):
        assert Path(fb_mgr.failed_dir).exists()


class TestAtomicMoveRetry:
    def test_atomic_move_read_failure_propagates_immediately(self, fb_mgr, tmp_path):
        """Read failures propagate immediately since content is read once before retries."""
        fb_dir = Path(fb_mgr.feedback_dir)
        done_dir = Path(fb_mgr.done_dir)
        src = fb_dir / "task.md"
        src.write_text("task content")
        dst = done_dir / "task.md"

        read_count = 0
        original_read_text = Path.read_text

        def failing_read_text(self_path, *args, **kwargs):
            nonlocal read_count
            if self_path == src:
                read_count += 1
                raise OSError("read failure")
            return original_read_text(self_path, *args, **kwargs)

        with patch.object(Path, "read_text", failing_read_text):
            with pytest.raises(OSError, match="read failure"):
                fb_mgr._atomic_move(src, dst)

        # Content is read exactly once (no retries on read failures)
        assert read_count == 1

    def test_atomic_move_reads_content_only_once(self, fb_mgr, tmp_path):
        """Content is read once before the retry loop, not on each attempt."""
        fb_dir = Path(fb_mgr.feedback_dir)
        done_dir = Path(fb_mgr.done_dir)
        src = fb_dir / "task.md"
        src.write_text("task content")
        dst = done_dir / "task.md"

        read_count = 0
        original_read_text = Path.read_text

        def counting_read_text(self_path, *args, **kwargs):
            nonlocal read_count
            if self_path == src:
                read_count += 1
            return original_read_text(self_path, *args, **kwargs)

        replace_count = 0
        original_replace = os.replace

        def failing_replace(src_path, dst_path):
            nonlocal replace_count
            replace_count += 1
            if replace_count <= 2:
                raise OSError("transient replace failure")
            return original_replace(src_path, dst_path)

        with patch.object(Path, "read_text", counting_read_text):
            with patch("os.replace", failing_replace):
                with patch("feedback.time.sleep"):
                    fb_mgr._atomic_move(src, dst)

        # Content was read exactly once, even though replace retried
        assert read_count == 1
        assert replace_count == 3  # failed twice, succeeded on third

    def test_atomic_move_source_already_moved(self, fb_mgr, tmp_path):
        """When src disappears on retry (another process moved it), treat as success."""
        fb_dir = Path(fb_mgr.feedback_dir)
        done_dir = Path(fb_mgr.done_dir)
        src = fb_dir / "task.md"
        src.write_text("task content")
        dst = done_dir / "task.md"

        original_replace = os.replace

        def fail_and_remove_src(src_path, dst_path):
            # Fail the first attempt and also remove the source file
            # to simulate another process moving it
            if src.exists():
                src.unlink()
            raise OSError("file contention")

        with patch("os.replace", fail_and_remove_src):
            with patch("feedback.time.sleep"):
                # Should not raise — source disappearing is treated as success
                fb_mgr._atomic_move(src, dst)

    def test_atomic_move_all_retries_exhausted(self, fb_mgr, tmp_path):
        """When every replace attempt fails, the last exception is raised."""
        fb_dir = Path(fb_mgr.feedback_dir)
        done_dir = Path(fb_mgr.done_dir)
        src = fb_dir / "task.md"
        src.write_text("task content")
        dst = done_dir / "task.md"

        def always_fail_replace(src_path, dst_path):
            raise OSError("persistent failure")

        with patch("os.replace", always_fail_replace):
            with patch("feedback.time.sleep"):
                with pytest.raises(OSError, match="persistent failure"):
                    fb_mgr._atomic_move(src, dst)

    def test_atomic_move_progressive_backoff_uses_sleep(self, fb_mgr, tmp_path):
        """Verify progressive backoff calls sleep with increasing delays."""
        fb_dir = Path(fb_mgr.feedback_dir)
        done_dir = Path(fb_mgr.done_dir)
        src = fb_dir / "task.md"
        src.write_text("task content")
        dst = done_dir / "task.md"

        replace_count = 0
        original_replace = os.replace

        def fail_three_times(src_path, dst_path):
            nonlocal replace_count
            replace_count += 1
            if replace_count <= 3:
                raise OSError("transient failure")
            return original_replace(src_path, dst_path)

        with patch("os.replace", fail_three_times):
            with patch("feedback.time.sleep") as mock_sleep:
                with patch("feedback.random.random", return_value=0.5):
                    fb_mgr._atomic_move(src, dst)

        # Should have slept 3 times (before attempts 2, 3, 4)
        assert mock_sleep.call_count == 3
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        # Delays should be increasing (exponential backoff)
        assert delays[0] < delays[1] < delays[2]


class TestFeedbackCleanup:
    def test_old_done_files_cleaned(self, fb_mgr):
        """Files older than 7 days in done/ should be removed."""
        done_dir = Path(fb_mgr.done_dir)
        old_file = done_dir / "old-task.md"
        old_file.write_text("old completed task")
        # Set mtime to 10 days ago
        old_mtime = time.time() - (10 * 86400)
        os.utime(old_file, (old_mtime, old_mtime))

        fb_mgr.get_pending_feedback()

        assert not old_file.exists()

    def test_recent_done_files_preserved(self, fb_mgr):
        """Files newer than 7 days in done/ should be preserved."""
        done_dir = Path(fb_mgr.done_dir)
        recent_file = done_dir / "recent-task.md"
        recent_file.write_text("recent completed task")

        fb_mgr.get_pending_feedback()

        assert recent_file.exists()

    def test_old_failed_files_cleaned(self, fb_mgr):
        """Files older than 7 days in failed/ should be removed."""
        failed_dir = Path(fb_mgr.failed_dir)
        old_file = failed_dir / "old-failed.md"
        old_file.write_text("old failed task")
        old_mtime = time.time() - (10 * 86400)
        os.utime(old_file, (old_mtime, old_mtime))

        fb_mgr.get_pending_feedback()

        assert not old_file.exists()

    def test_gitkeep_preserved(self, fb_mgr):
        """The .gitkeep file should never be removed."""
        done_dir = Path(fb_mgr.done_dir)
        gitkeep = done_dir / ".gitkeep"
        gitkeep.write_text("")
        old_mtime = time.time() - (30 * 86400)
        os.utime(gitkeep, (old_mtime, old_mtime))

        fb_mgr.get_pending_feedback()

        assert gitkeep.exists()

    def test_cleanup_handles_iterdir_oserror(self, fb_mgr):
        """_cleanup_old_files should not crash if iterdir() raises OSError."""
        done_dir = Path(fb_mgr.done_dir)
        # Put a file so the directory is non-empty
        (done_dir / "task.md").write_text("content")
        old_mtime = time.time() - (10 * 86400)
        os.utime(done_dir / "task.md", (old_mtime, old_mtime))

        with patch.object(Path, "iterdir", side_effect=OSError("Permission denied")):
            # Should not raise
            fb_mgr._cleanup_old_files(done_dir)

    def test_single_pass_iterdir_finds_both_md_and_prd(self, fb_mgr):
        """get_pending_feedback reads both .md and .prd.yaml in a single directory pass."""
        feedback_dir = Path(fb_mgr.feedback_dir)
        (feedback_dir / "task1.md").write_text("Fix a bug")
        (feedback_dir / "task2.txt").write_text("Add a feature")
        tasks = fb_mgr.get_pending_feedback()
        descs = [t.description for t in tasks]
        assert "Fix a bug" in descs
        assert "Add a feature" in descs

    def test_iterdir_oserror_returns_empty(self, fb_mgr):
        """get_pending_feedback returns empty list if iterdir raises OSError."""
        with patch.object(Path, "iterdir", side_effect=OSError("Permission denied")):
            tasks = fb_mgr.get_pending_feedback()
        assert tasks == []


class TestFeedbackPathValidation:
    """Tests that mark_done/mark_failed reject path traversal attempts."""

    def test_mark_done_rejects_path_outside_feedback_dir(self, fb_mgr, tmp_path):
        """mark_done should not process files outside the feedback directory."""
        outside_file = tmp_path / "outside" / "secret.md"
        outside_file.parent.mkdir(parents=True, exist_ok=True)
        outside_file.write_text("sensitive data")

        fb_mgr.mark_done(str(outside_file))

        # File should NOT have been moved — it's outside the feedback dir
        assert outside_file.exists()
        done_dir = Path(fb_mgr.done_dir)
        assert not (done_dir / "secret.md").exists()

    def test_mark_failed_rejects_path_outside_feedback_dir(self, fb_mgr, tmp_path):
        """mark_failed should not process files outside the feedback directory."""
        outside_file = tmp_path / "outside" / "secret.md"
        outside_file.parent.mkdir(parents=True, exist_ok=True)
        outside_file.write_text("sensitive data")

        fb_mgr.mark_failed(str(outside_file))

        # File should NOT have been moved — it's outside the feedback dir
        assert outside_file.exists()
        failed_dir = Path(fb_mgr.failed_dir)
        assert not (failed_dir / "secret.md").exists()

    def test_mark_done_allows_files_in_feedback_dir(self, fb_mgr):
        """mark_done should work normally for files inside the feedback directory."""
        fb_dir = Path(fb_mgr.feedback_dir)
        task_file = fb_dir / "legit-task.md"
        task_file.write_text("Fix the bug")

        fb_mgr.mark_done(str(task_file))
        assert not task_file.exists()
        assert (Path(fb_mgr.done_dir) / "legit-task.md").exists()

    def test_mark_done_rejects_symlink_in_feedback_dir(self, fb_mgr, tmp_path):
        """mark_done should reject symlinks within the feedback directory."""
        fb_dir = Path(fb_mgr.feedback_dir)
        # Create a file outside feedback dir
        outside_file = tmp_path / "outside" / "secret.md"
        outside_file.parent.mkdir(parents=True, exist_ok=True)
        outside_file.write_text("sensitive data")

        # Create a symlink inside feedback dir pointing outside
        symlink = fb_dir / "sneaky-link.md"
        symlink.symlink_to(outside_file)

        fb_mgr.mark_done(str(symlink))

        # Symlink should NOT have been processed
        assert outside_file.exists()  # Original file untouched
        done_dir = Path(fb_mgr.done_dir)
        assert not (done_dir / "sneaky-link.md").exists()

    def test_mark_failed_rejects_symlink_in_feedback_dir(self, fb_mgr, tmp_path):
        """mark_failed should reject symlinks within the feedback directory."""
        fb_dir = Path(fb_mgr.feedback_dir)
        outside_file = tmp_path / "outside" / "secret.md"
        outside_file.parent.mkdir(parents=True, exist_ok=True)
        outside_file.write_text("sensitive data")

        symlink = fb_dir / "sneaky-link.md"
        symlink.symlink_to(outside_file)

        fb_mgr.mark_failed(str(symlink))

        assert outside_file.exists()
        failed_dir = Path(fb_mgr.failed_dir)
        assert not (failed_dir / "sneaky-link.md").exists()


class TestPathTraversalPrefixBypass:
    """Test that _is_within_feedback_dir rejects sibling dirs with shared prefix."""

    def test_sibling_dir_with_shared_prefix_rejected(self, tmp_path, default_config):
        """A path like /feedback_evil/file should NOT pass the feedback dir check."""
        # Create two sibling directories with shared prefix
        feedback_dir = tmp_path / "feedback"
        feedback_evil = tmp_path / "feedback_evil"
        feedback_dir.mkdir()
        feedback_evil.mkdir()

        default_config.paths.feedback_dir = str(feedback_dir)
        default_config.paths.feedback_done_dir = str(feedback_dir / "done")
        default_config.paths.feedback_failed_dir = str(feedback_dir / "failed")
        mgr = FeedbackManager(default_config)

        evil_file = feedback_evil / "evil.md"
        evil_file.write_text("malicious content")

        # This should be rejected — it's outside the feedback dir
        assert mgr._is_within_feedback_dir(evil_file) is False

    def test_valid_file_inside_feedback_accepted(self, fb_mgr):
        fb_dir = Path(fb_mgr.feedback_dir)
        fb_dir.mkdir(parents=True, exist_ok=True)
        valid_file = fb_dir / "valid.md"
        valid_file.write_text("legit content")
        assert fb_mgr._is_within_feedback_dir(valid_file) is True


class TestMarkClaimedSecurityCheck:
    """Test that mark_done_claimed and mark_failed_claimed check _is_within_feedback_dir."""

    def test_mark_done_claimed_rejects_outside_path(self, fb_mgr, tmp_path):
        """mark_done_claimed should reject paths outside the feedback dir."""
        outside = tmp_path / "outside" / "evil.md"
        outside.parent.mkdir(parents=True, exist_ok=True)
        # Create the .claimed file
        claimed = outside.with_suffix(".md.claimed")
        claimed.write_text("malicious")

        fb_mgr.mark_done_claimed(str(outside))

        # File should NOT have been moved to done/
        done_dir = Path(fb_mgr.done_dir)
        assert not (done_dir / "evil.md").exists()
        # Claimed file should still be where it was
        assert claimed.exists()

    def test_mark_failed_claimed_rejects_outside_path(self, fb_mgr, tmp_path):
        """mark_failed_claimed should reject paths outside the feedback dir."""
        outside = tmp_path / "outside" / "evil.md"
        outside.parent.mkdir(parents=True, exist_ok=True)
        claimed = outside.with_suffix(".md.claimed")
        claimed.write_text("malicious")

        fb_mgr.mark_failed_claimed(str(outside))

        failed_dir = Path(fb_mgr.failed_dir)
        assert not (failed_dir / "evil.md").exists()
        assert claimed.exists()


class TestGetPendingFeedbackSymlinkSecurity:
    def test_symlink_in_feedback_dir_is_skipped(self, fb_mgr, tmp_path):
        """Symlinks in feedback dir should be skipped during get_pending_feedback."""
        fb_dir = Path(fb_mgr.feedback_dir)
        fb_dir.mkdir(parents=True, exist_ok=True)

        # Create a file outside feedback dir
        secret = tmp_path / "secret.txt"
        secret.write_text("sensitive data")

        # Create symlink inside feedback dir pointing to it
        link = fb_dir / "symlink.txt"
        link.symlink_to(secret)

        tasks = fb_mgr.get_pending_feedback()
        # The symlink should be skipped - no task with "sensitive data"
        for t in tasks:
            assert "sensitive data" not in t.description


class TestUniqueDst:
    """Test the _unique_dst helper for collision avoidance."""

    def test_no_collision(self, fb_mgr):
        """When no file exists, returns the original name."""
        dst = fb_mgr._unique_dst(fb_mgr.done_dir, "task.txt")
        assert dst == fb_mgr.done_dir / "task.txt"

    def test_collision_increments(self, fb_mgr):
        """When file exists, appends _1, _2, etc."""
        (fb_mgr.done_dir / "task.txt").write_text("existing")
        dst = fb_mgr._unique_dst(fb_mgr.done_dir, "task.txt")
        assert dst == fb_mgr.done_dir / "task_1.txt"

    def test_multiple_collisions(self, fb_mgr):
        """Skips over multiple existing files."""
        (fb_mgr.done_dir / "task.txt").write_text("existing")
        (fb_mgr.done_dir / "task_1.txt").write_text("existing")
        (fb_mgr.done_dir / "task_2.txt").write_text("existing")
        dst = fb_mgr._unique_dst(fb_mgr.done_dir, "task.txt")
        assert dst == fb_mgr.done_dir / "task_3.txt"

    def test_all_slots_exhausted_uses_timestamp(self, fb_mgr):
        """When all 1000 numbered slots are exhausted, uses timestamp suffix."""
        done_dir = fb_mgr.done_dir
        # Create the base file and all 999 numbered variants
        (done_dir / "task.txt").write_text("existing")
        for i in range(1, 1000):
            (done_dir / f"task_{i}.txt").write_text("existing")
        dst = fb_mgr._unique_dst(done_dir, "task.txt")
        # Should NOT be an existing file (no silent overwrite)
        assert not dst.exists()
        # Should contain a timestamp-like number (>= 1000)
        assert "task_" in dst.name
        assert dst.suffix == ".txt"


class TestClaimFeedback:
    def test_claim_success(self, fb_mgr):
        fb_dir = Path(fb_mgr.feedback_dir)
        src = fb_dir / "task.md"
        src.write_text("fix bug")
        assert fb_mgr.claim_feedback(str(src)) is True
        assert not src.exists()
        assert src.with_suffix(".md.claimed").exists()

    def test_claim_missing_file_returns_false(self, fb_mgr):
        fb_dir = Path(fb_mgr.feedback_dir)
        src = fb_dir / "nonexistent.md"
        assert fb_mgr.claim_feedback(str(src)) is False

    def test_claim_permission_error_returns_false(self, fb_mgr):
        """OSError (e.g. permission denied) should return False, not crash."""
        fb_dir = Path(fb_mgr.feedback_dir)
        src = fb_dir / "task.md"
        src.write_text("fix bug")
        with patch("feedback.os.rename", side_effect=PermissionError("denied")):
            assert fb_mgr.claim_feedback(str(src)) is False


class TestAtomicMoveFileNotFound:
    def test_atomic_move_missing_source_returns_gracefully(self, fb_mgr):
        """If source file vanishes before read, _atomic_move should not crash."""
        fb_dir = Path(fb_mgr.feedback_dir)
        done_dir = Path(fb_mgr.done_dir)
        src = fb_dir / "gone.md"
        dst = done_dir / "gone.md"
        # Source doesn't exist — should return gracefully (no exception)
        fb_mgr._atomic_move(src, dst)
        # Destination should not have been created
        assert not dst.exists()

    def test_atomic_move_file_deleted_after_existence_check(self, fb_mgr):
        """Simulate TOCTOU: file exists at check time but gone at read time."""
        fb_dir = Path(fb_mgr.feedback_dir)
        done_dir = Path(fb_mgr.done_dir)
        src = fb_dir / "vanishing.md"
        dst = done_dir / "vanishing.md"
        # Create file, then delete it before _atomic_move reads it
        src.write_text("content")
        src.unlink()
        # Should not raise — handles FileNotFoundError gracefully
        fb_mgr._atomic_move(src, dst)
