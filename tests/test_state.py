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
        """_save_history should retry os.replace with increasing delays."""
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
        # Should have slept between retries with exponential backoff
        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list[0][0][0] == 0.1
        assert mock_sleep.call_args_list[1][0][0] == 0.3
        # Data should be persisted correctly
        data = json.loads(Path(state_mgr.history_file).read_text())
        assert len(data) == 1
        assert data[0]["task_description"] == "Retry test"

    def test_save_history_all_retries_fail(self, state_mgr):
        """_save_history should raise OSError when all 5 retries fail."""
        call_count = 0

        def always_fail(src, dst):
            nonlocal call_count
            call_count += 1
            raise OSError("permanently locked")

        with patch("state.os.replace", side_effect=always_fail):
            with patch("state.time.sleep"):
                with pytest.raises(OSError, match="permanently locked"):
                    state_mgr.record_cycle(CycleRecord(
                        timestamp=time.time(),
                        task_description="Doomed task",
                        success=True,
                    ))
        # Should have attempted all 5 retries
        assert call_count == 5


class TestAdaptiveBatchSize:
    def test_empty_history_returns_initial(self, state_mgr):
        assert state_mgr.compute_adaptive_batch_size() == 3

    def test_all_successes_grows_to_max(self, state_mgr):
        now = time.time()
        for i in range(20):
            state_mgr.record_cycle(CycleRecord(
                timestamp=now + i,
                task_description=f"Task {i}",
                success=True,
            ))
        assert state_mgr.compute_adaptive_batch_size() == 10

    def test_all_failures_shrinks_to_min(self, state_mgr):
        now = time.time()
        for i in range(20):
            state_mgr.record_cycle(CycleRecord(
                timestamp=now + i,
                task_description=f"Task {i}",
                success=False,
            ))
        assert state_mgr.compute_adaptive_batch_size() == 1

    def test_mixed_results(self, state_mgr):
        """Starting from initial=3: +1(S), +1(S), -2(F), +1(S) -> 3+1+1-2+1=4"""
        now = time.time()
        results = [True, True, False, True]
        for i, success in enumerate(results):
            state_mgr.record_cycle(CycleRecord(
                timestamp=now + i,
                task_description=f"Task {i}",
                success=success,
            ))
        assert state_mgr.compute_adaptive_batch_size() == 4

    def test_window_limits_history(self, tmp_path, default_config):
        """Only the last adaptive_batch_window records should matter."""
        default_config.paths.history_file = str(tmp_path / "history.json")
        default_config.orchestrator.adaptive_batch_window = 3
        mgr = StateManager(default_config)

        now = time.time()
        # Write 5 failures (old), then 3 successes (recent)
        for i in range(5):
            mgr.record_cycle(CycleRecord(
                timestamp=now + i,
                task_description=f"Fail {i}",
                success=False,
            ))
        for i in range(3):
            mgr.record_cycle(CycleRecord(
                timestamp=now + 5 + i,
                task_description=f"Success {i}",
                success=True,
            ))
        # Window=3 means only the last 3 (all successes) are considered
        # initial=3 + 1 + 1 + 1 = 6
        assert mgr.compute_adaptive_batch_size() == 6


class TestKeyBasedDedup:
    def test_was_recently_attempted_by_key(self, state_mgr):
        record = CycleRecord(
            timestamp=time.time(),
            task_description="Fix error handling in safety.py",
            task_type="claude_idea",
            task_keys=["claude_idea:safety.py"],
        )
        state_mgr.record_cycle(record)
        # Different description but same key should match
        assert state_mgr.was_recently_attempted(
            "Improve error handling in safety.py",
            task_key="claude_idea:safety.py",
        ) is True
        # Same description, no key match
        assert state_mgr.was_recently_attempted("Unrelated task") is False

    def test_backward_compat_old_records_no_task_keys(self, state_mgr):
        """Old records without task_keys field still work correctly."""
        old_record = {
            "timestamp": time.time(),
            "task_description": "Legacy task",
            "task_type": "test_failure",
            "success": False,
        }
        Path(state_mgr.history_file).write_text(json.dumps([old_record]))
        assert state_mgr.was_recently_attempted("Legacy task") is True
        assert state_mgr.was_recently_attempted(
            "Different desc", task_key="test_failure:foo.py"
        ) is False

    def test_get_task_failure_count_by_key(self, state_mgr):
        now = time.time()
        state_mgr.record_cycle(CycleRecord(
            timestamp=now,
            task_description="Fix bug version 1",
            task_type="claude_idea",
            success=False,
            task_keys=["claude_idea:safety.py"],
        ))
        state_mgr.record_cycle(CycleRecord(
            timestamp=now + 1,
            task_description="Fix bug version 2",
            task_type="claude_idea",
            success=False,
            task_keys=["claude_idea:safety.py"],
        ))
        # Neither description matches, but key does
        assert state_mgr.get_task_failure_count(
            "Fix bug version 3", task_key="claude_idea:safety.py"
        ) == 2

    def test_get_task_failure_count_by_key_with_type_filter(self, state_mgr):
        now = time.time()
        state_mgr.record_cycle(CycleRecord(
            timestamp=now,
            task_description="Some task",
            task_type="claude_idea",
            success=False,
            task_keys=["claude_idea:safety.py"],
            task_types=["claude_idea"],
        ))
        assert state_mgr.get_task_failure_count(
            "Different desc", "claude_idea", task_key="claude_idea:safety.py"
        ) == 1
        assert state_mgr.get_task_failure_count(
            "Different desc", "feedback", task_key="claude_idea:safety.py"
        ) == 0


class TestFailureRecovery:
    def test_reset_consecutive_failures(self, state_mgr):
        """Injecting a synthetic success resets the consecutive failure counter."""
        now = time.time()
        for i in range(5):
            state_mgr.record_cycle(CycleRecord(
                timestamp=now + i,
                task_description=f"Fail {i}",
                success=False,
            ))
        assert state_mgr.get_consecutive_failures() == 5

        state_mgr.reset_consecutive_failures("test reset")
        assert state_mgr.get_consecutive_failures() == 0

        # Verify the synthetic record exists
        data = json.loads(Path(state_mgr.history_file).read_text())
        last = data[-1]
        assert last["task_type"] == "system_reset"
        assert last["success"] is True
        assert "test reset" in last["task_description"]

    def test_auto_reset_after_idle(self, state_mgr):
        """Auto-reset triggers when system has been idle for over min_idle_seconds."""
        old_time = time.time() - 7200  # 2 hours ago
        for i in range(5):
            state_mgr.record_cycle(CycleRecord(
                timestamp=old_time + i,
                task_description=f"Fail {i}",
                success=False,
            ))
        assert state_mgr.get_consecutive_failures() == 5
        assert state_mgr.should_auto_reset_failures(min_idle_seconds=3600) is True

    def test_no_reset_when_recently_active(self, state_mgr):
        """Auto-reset does NOT trigger when last cycle was recent."""
        now = time.time()
        for i in range(5):
            state_mgr.record_cycle(CycleRecord(
                timestamp=now - 60 + i,  # Very recent
                task_description=f"Fail {i}",
                success=False,
            ))
        assert state_mgr.get_consecutive_failures() == 5
        assert state_mgr.should_auto_reset_failures(min_idle_seconds=3600) is False

    def test_no_auto_reset_below_limit(self, state_mgr):
        """Auto-reset does NOT trigger when failures are below the limit."""
        old_time = time.time() - 7200
        for i in range(2):
            state_mgr.record_cycle(CycleRecord(
                timestamp=old_time + i,
                task_description=f"Fail {i}",
                success=False,
            ))
        assert state_mgr.get_consecutive_failures() == 2
        assert state_mgr.should_auto_reset_failures(min_idle_seconds=3600) is False


class TestCorruptHistoryBackup:
    def test_corrupt_history_backed_up(self, state_mgr):
        """Corrupted JSON history file is backed up before returning empty."""
        corrupted_content = "{this is not valid json!!"
        Path(state_mgr.history_file).write_text(corrupted_content)

        result = state_mgr._load_history()

        assert result == []
        assert state_mgr.get_consecutive_failures() == 0
        backup_path = Path(str(state_mgr.history_file) + ".corrupt")
        assert backup_path.exists()
        assert backup_path.read_text() == corrupted_content

    def test_corrupt_history_not_destroyed_by_record_cycle(self, state_mgr):
        """record_cycle after corruption preserves the backup and writes new data."""
        corrupted_content = "{this is not valid json!!"
        Path(state_mgr.history_file).write_text(corrupted_content)

        state_mgr.record_cycle(CycleRecord(
            timestamp=time.time(),
            task_description="New record after corruption",
            success=True,
        ))

        backup_path = Path(str(state_mgr.history_file) + ".corrupt")
        assert backup_path.exists()
        assert backup_path.read_text() == corrupted_content

        data = json.loads(Path(state_mgr.history_file).read_text())
        assert len(data) == 1
        assert data[0]["task_description"] == "New record after corruption"

    def test_corrupt_history_cache_prevents_repeated_backup(self, state_mgr):
        """After first load of corrupted file, cache prevents re-reading on second call."""
        corrupted_content = "{this is not valid json!!"
        Path(state_mgr.history_file).write_text(corrupted_content)

        # First call: triggers backup
        state_mgr._load_history()
        backup_path = Path(str(state_mgr.history_file) + ".corrupt")
        assert backup_path.exists()

        # Remove backup to verify second call doesn't recreate it
        backup_path.unlink()

        # Second call: should use cache, not re-read file
        result = state_mgr._load_history()
        assert result == []
        assert not backup_path.exists()  # backup was NOT recreated
