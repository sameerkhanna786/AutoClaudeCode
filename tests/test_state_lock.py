"""Tests for state_lock module."""

import json
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from config_schema import Config
from state import CycleRecord
from state_lock import LockedStateManager


@pytest.fixture
def locked_state(tmp_path):
    """Create a LockedStateManager with a temp state directory."""
    config = Config()
    config.paths.state_dir = str(tmp_path / "state")
    config.paths.history_file = str(tmp_path / "state" / "history.json")
    return LockedStateManager(config)


class TestLockedStateManager:
    def test_record_cycle_basic(self, locked_state):
        """Basic record_cycle works through the lock."""
        record = CycleRecord(
            timestamp=time.time(),
            task_description="Test task",
            task_type="lint",
            success=True,
        )
        locked_state.record_cycle(record)

        # Verify it was written
        history = locked_state._load_history()
        assert len(history) == 1
        assert history[0]["task_description"] == "Test task"

    def test_was_recently_attempted(self, locked_state):
        """was_recently_attempted works through the lock."""
        record = CycleRecord(
            timestamp=time.time(),
            task_description="Test task",
            task_type="lint",
            success=False,
        )
        locked_state.record_cycle(record)
        assert locked_state.was_recently_attempted("Test task") is True
        assert locked_state.was_recently_attempted("Other task") is False

    def test_concurrent_record_cycles(self, locked_state):
        """Multiple threads can record cycles without data loss."""
        errors = []
        num_threads = 5

        def record_cycle(thread_id):
            try:
                record = CycleRecord(
                    timestamp=time.time(),
                    task_description=f"Task from thread {thread_id}",
                    task_type="lint",
                    success=True,
                )
                locked_state.record_cycle(record)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=record_cycle, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent writes: {errors}"

        history = locked_state._load_history()
        assert len(history) == num_threads

    def test_get_cycle_count_last_hour(self, locked_state):
        """get_cycle_count_last_hour works through the lock."""
        record = CycleRecord(
            timestamp=time.time(),
            task_description="Recent task",
            task_type="lint",
            success=True,
        )
        locked_state.record_cycle(record)
        assert locked_state.get_cycle_count_last_hour() == 1

    def test_get_total_cost(self, locked_state):
        """get_total_cost works through the lock."""
        record = CycleRecord(
            timestamp=time.time(),
            task_description="Costly task",
            task_type="lint",
            success=True,
            cost_usd=1.50,
        )
        locked_state.record_cycle(record)
        assert locked_state.get_total_cost() == 1.50

    def test_get_consecutive_failures(self, locked_state):
        """get_consecutive_failures works through the lock."""
        for i in range(3):
            record = CycleRecord(
                timestamp=time.time(),
                task_description=f"Failing task {i}",
                task_type="lint",
                success=False,
            )
            locked_state.record_cycle(record)
        assert locked_state.get_consecutive_failures() == 3

    def test_lock_file_created(self, locked_state):
        """The lock file is created during operations."""
        record = CycleRecord(
            timestamp=time.time(),
            task_description="Test",
            task_type="lint",
            success=True,
        )
        locked_state.record_cycle(record)
        assert locked_state._lock_path.exists()
