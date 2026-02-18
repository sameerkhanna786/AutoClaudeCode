"""Tests for state module."""

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from config_schema import Config
from state import CycleRecord, StateManager


@pytest.fixture
def state_mgr(tmp_path, default_config):
    default_config.paths.history_file = str(tmp_path / "history.json")
    return StateManager(default_config)


class TestStateManager:
    def test_record_and_load(self, state_mgr):
        record = CycleRecord(
            timestamp=time.time(),
            task_description="Fix bug",
            success=True,
            cost_usd=0.05,
        )
        state_mgr.record_cycle(record)
        # Verify file was created
        assert Path(state_mgr.history_file).exists()
        data = json.loads(Path(state_mgr.history_file).read_text())
        assert len(data) == 1
        assert data[0]["task_description"] == "Fix bug"
        assert data[0]["success"] is True

    def test_multiple_records(self, state_mgr):
        for i in range(3):
            state_mgr.record_cycle(CycleRecord(
                timestamp=time.time(),
                task_description=f"Task {i}",
                success=i % 2 == 0,
            ))
        data = json.loads(Path(state_mgr.history_file).read_text())
        assert len(data) == 3

    def test_was_recently_attempted(self, state_mgr):
        state_mgr.record_cycle(CycleRecord(
            timestamp=time.time(),
            task_description="Fix bug",
        ))
        assert state_mgr.was_recently_attempted("Fix bug") is True
        assert state_mgr.was_recently_attempted("Other task") is False

    def test_was_recently_attempted_respects_lookback(self, state_mgr):
        state_mgr.record_cycle(CycleRecord(
            timestamp=time.time() - 7200,  # 2 hours ago
            task_description="Old task",
        ))
        assert state_mgr.was_recently_attempted("Old task", lookback_seconds=3600) is False

    def test_get_cycle_count_last_hour(self, state_mgr):
        now = time.time()
        for i in range(5):
            state_mgr.record_cycle(CycleRecord(
                timestamp=now - i * 60,
                task_description=f"Task {i}",
            ))
        # Add one old record
        state_mgr.record_cycle(CycleRecord(
            timestamp=now - 7200,
            task_description="Old",
        ))
        assert state_mgr.get_cycle_count_last_hour() == 5

    def test_get_total_cost(self, state_mgr):
        now = time.time()
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="A", cost_usd=0.10,
        ))
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="B", cost_usd=0.20,
        ))
        assert abs(state_mgr.get_total_cost() - 0.30) < 0.001

    def test_get_consecutive_failures(self, state_mgr):
        now = time.time()
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="A", success=True,
        ))
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="B", success=False,
        ))
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="C", success=False,
        ))
        assert state_mgr.get_consecutive_failures() == 2

    def test_consecutive_failures_reset_on_success(self, state_mgr):
        now = time.time()
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="A", success=False,
        ))
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="B", success=True,
        ))
        assert state_mgr.get_consecutive_failures() == 0

    def test_empty_history(self, state_mgr):
        assert state_mgr.get_cycle_count_last_hour() == 0
        assert state_mgr.get_total_cost() == 0.0
        assert state_mgr.get_consecutive_failures() == 0
        assert state_mgr.was_recently_attempted("anything") is False

    def test_history_pruning(self, tmp_path, default_config):
        """Verify history is pruned to max_history_records."""
        default_config.paths.history_file = str(tmp_path / "history.json")
        default_config.safety.max_history_records = 10
        mgr = StateManager(default_config)

        now = time.time()
        for i in range(25):
            mgr.record_cycle(CycleRecord(
                timestamp=now + i,
                task_description=f"Task {i}",
            ))

        # On-disk file should have at most 10 records
        data = json.loads(Path(mgr.history_file).read_text())
        assert len(data) == 10
        # The most recent records should be preserved (Task 15..24)
        assert data[0]["task_description"] == "Task 15"
        assert data[-1]["task_description"] == "Task 24"

    def test_cache_invalidation(self, state_mgr):
        """Verify cache is invalidated when file is externally modified."""
        state_mgr.record_cycle(CycleRecord(
            timestamp=time.time(),
            task_description="Original",
        ))
        assert state_mgr.was_recently_attempted("Original") is True

        # Externally overwrite the history file with different content
        new_records = [{"timestamp": time.time(), "task_description": "External", "success": True}]
        Path(state_mgr.history_file).write_text(json.dumps(new_records))

        # The cache should detect the mtime change and reload
        assert state_mgr.was_recently_attempted("External") is True
        assert state_mgr.was_recently_attempted("Original") is False

    def test_cache_avoids_reread(self, state_mgr):
        """Verify that repeated reads use the cache instead of re-reading from disk."""
        state_mgr.record_cycle(CycleRecord(
            timestamp=time.time(),
            task_description="Cached task",
            success=True,
            cost_usd=0.05,
        ))

        # After record_cycle, the cache is populated via _save_history.
        # Subsequent calls should not re-read the file.
        with patch.object(Path, 'read_text', wraps=state_mgr.history_file.read_text) as mock_read:
            state_mgr.was_recently_attempted("Cached task")
            state_mgr.get_cycle_count_last_hour()
            state_mgr.get_total_cost()
            state_mgr.get_consecutive_failures()
            # None of these should have triggered a file read
            mock_read.assert_not_called()

    def test_get_task_failure_count_empty(self, state_mgr):
        assert state_mgr.get_task_failure_count("anything") == 0

    def test_get_task_failure_count_counts_failures(self, state_mgr):
        now = time.time()
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="Fix bug", task_type="feedback", success=False,
        ))
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="Fix bug", task_type="feedback", success=False,
        ))
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="Fix bug", task_type="feedback", success=True,
        ))
        assert state_mgr.get_task_failure_count("Fix bug") == 2

    def test_get_task_failure_count_filters_by_type(self, state_mgr):
        now = time.time()
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="Fix bug", task_type="feedback", success=False,
        ))
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="Fix bug", task_type="test_failure", success=False,
        ))
        assert state_mgr.get_task_failure_count("Fix bug", "feedback") == 1
        assert state_mgr.get_task_failure_count("Fix bug", "test_failure") == 1
        assert state_mgr.get_task_failure_count("Fix bug") == 2

    def test_get_task_failure_count_ignores_other_tasks(self, state_mgr):
        now = time.time()
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="Task A", task_type="feedback", success=False,
        ))
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="Task B", task_type="feedback", success=False,
        ))
        assert state_mgr.get_task_failure_count("Task A", "feedback") == 1


class TestBatchCycleRecord:
    def test_batch_record_stores_descriptions(self, state_mgr):
        record = CycleRecord(
            timestamp=time.time(),
            task_description="Fix bug in foo.py",
            task_type="test_failure",
            success=True,
            task_descriptions=["Fix bug in foo.py", "Address TODO in bar.py"],
            task_types=["test_failure", "todo"],
        )
        state_mgr.record_cycle(record)
        data = json.loads(Path(state_mgr.history_file).read_text())
        assert len(data) == 1
        assert data[0]["task_descriptions"] == ["Fix bug in foo.py", "Address TODO in bar.py"]
        assert data[0]["task_types"] == ["test_failure", "todo"]

    def test_was_recently_attempted_checks_batch_descriptions(self, state_mgr):
        record = CycleRecord(
            timestamp=time.time(),
            task_description="Fix bug in foo.py",
            task_type="test_failure",
            task_descriptions=["Fix bug in foo.py", "Address TODO in bar.py"],
            task_types=["test_failure", "todo"],
        )
        state_mgr.record_cycle(record)
        assert state_mgr.was_recently_attempted("Fix bug in foo.py") is True
        assert state_mgr.was_recently_attempted("Address TODO in bar.py") is True
        assert state_mgr.was_recently_attempted("Unrelated task") is False

    def test_backward_compat_old_records(self, state_mgr):
        """Old records without list fields still work correctly."""
        old_record = {
            "timestamp": time.time(),
            "task_description": "Legacy task",
            "task_type": "test_failure",
            "success": False,
        }
        Path(state_mgr.history_file).write_text(json.dumps([old_record]))
        assert state_mgr.was_recently_attempted("Legacy task") is True
        assert state_mgr.get_task_failure_count("Legacy task") == 1

    def test_get_task_failure_count_checks_batch(self, state_mgr):
        record = CycleRecord(
            timestamp=time.time(),
            task_description="Fix bug in foo.py",
            task_type="test_failure",
            success=False,
            task_descriptions=["Fix bug in foo.py", "Address TODO in bar.py"],
            task_types=["test_failure", "todo"],
        )
        state_mgr.record_cycle(record)
        assert state_mgr.get_task_failure_count("Address TODO in bar.py") == 1
        assert state_mgr.get_task_failure_count("Address TODO in bar.py", "todo") == 1
        assert state_mgr.get_task_failure_count("Address TODO in bar.py", "lint") == 0

    def test_save_history_retries_on_replace_failure(self, state_mgr):
        """_save_history should retry os.replace and succeed on a subsequent attempt."""
        call_count = 0
        original_replace = os.replace

        def failing_replace(src, dst):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OSError("file is locked")
            return original_replace(src, dst)

        with patch("state.os.replace", side_effect=failing_replace):
            with patch("state.time.sleep") as mock_sleep:
                state_mgr.record_cycle(CycleRecord(
                    timestamp=time.time(),
                    task_description="Retry test",
                    success=True,
                ))

        # Should have retried and eventually succeeded
        assert call_count == 3
        # Should have slept between retries
        assert mock_sleep.call_count == 2
        # Data should be persisted correctly
        data = json.loads(Path(state_mgr.history_file).read_text())
        assert len(data) == 1
        assert data[0]["task_description"] == "Retry test"
