"""Tests for feedback module."""

import logging
import os
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
    def test_atomic_move_retries_on_read_failure(self, fb_mgr, tmp_path):
        """_atomic_move retries when src.read_text() fails on first attempt."""
        fb_dir = Path(fb_mgr.feedback_dir)
        done_dir = Path(fb_mgr.done_dir)
        src = fb_dir / "task.md"
        src.write_text("task content")
        dst = done_dir / "task.md"

        call_count = 0
        original_read_text = Path.read_text

        def failing_read_text(self_path, *args, **kwargs):
            nonlocal call_count
            if self_path == src:
                call_count += 1
                if call_count == 1:
                    raise OSError("temporary read failure")
            return original_read_text(self_path, *args, **kwargs)

        with patch.object(Path, "read_text", failing_read_text):
            with patch("feedback.time.sleep"):
                fb_mgr._atomic_move(src, dst)

        assert dst.exists()
        assert dst.read_text() == "task content"
        assert call_count >= 1

    def test_atomic_move_source_already_moved(self, fb_mgr, tmp_path):
        """When src disappears on retry (another process moved it), treat as success."""
        fb_dir = Path(fb_mgr.feedback_dir)
        done_dir = Path(fb_mgr.done_dir)
        src = fb_dir / "task.md"
        src.write_text("task content")
        dst = done_dir / "task.md"

        def always_fail_read(self_path, *args, **kwargs):
            # Fail on first attempt, then src won't exist for retry
            if self_path == src:
                # Remove src to simulate another process moving it
                if src.exists():
                    src.unlink()
                raise OSError("file gone")
            return Path.read_text(self_path, *args, **kwargs)

        with patch.object(Path, "read_text", always_fail_read):
            with patch("feedback.time.sleep"):
                # Should not raise — source disappearing is treated as success
                fb_mgr._atomic_move(src, dst)

    def test_atomic_move_all_retries_exhausted(self, fb_mgr, tmp_path):
        """When every attempt fails, the last exception is raised."""
        fb_dir = Path(fb_mgr.feedback_dir)
        done_dir = Path(fb_mgr.done_dir)
        src = fb_dir / "task.md"
        src.write_text("task content")
        dst = done_dir / "task.md"

        def always_fail_read(self_path, *args, **kwargs):
            if self_path == src:
                raise OSError("persistent failure")
            return Path.read_text(self_path, *args, **kwargs)

        with patch.object(Path, "read_text", always_fail_read):
            with patch("feedback.time.sleep"):
                with pytest.raises(OSError, match="persistent failure"):
                    fb_mgr._atomic_move(src, dst)
