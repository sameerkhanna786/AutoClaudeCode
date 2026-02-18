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
