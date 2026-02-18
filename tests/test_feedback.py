"""Tests for feedback module."""

import logging
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
