"""Tests for state module."""

import errno
import json
import os
import tempfile
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

    def test_record_cycle_does_not_mutate_cache(self, state_mgr):
        """record_cycle must not mutate the in-memory cache before save succeeds."""
        state_mgr.record_cycle(CycleRecord(
            timestamp=time.time(),
            task_description="First",
            success=True,
        ))
        # Load cache
        cached = state_mgr._load_history()
        cached_len_before = len(cached)

        # Record another cycle
        state_mgr.record_cycle(CycleRecord(
            timestamp=time.time(),
            task_description="Second",
            success=True,
        ))
        # The previously returned cache reference should not have been mutated
        assert len(cached) == cached_len_before, (
            "record_cycle must not append to the cached list directly"
        )

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
        """_save_history should retry os.replace with exponential backoff."""
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
                with patch("state.random.random", return_value=0.5):
                    state_mgr.record_cycle(CycleRecord(
                        timestamp=time.time(),
                        task_description="Retry test",
                        success=True,
                    ))

        # Should have retried and eventually succeeded
        assert call_count == 3
        # Should have slept between retries with exponential backoff + jitter
        assert mock_sleep.call_count == 2
        # With random()=0.5, jitter factor is 0.5 + 0.5*0.5 = 0.75
        # delay0 = 0.1 * 3^0 * 0.75 = 0.075
        # delay1 = 0.1 * 3^1 * 0.75 = 0.225
        assert abs(mock_sleep.call_args_list[0][0][0] - 0.075) < 0.01
        assert abs(mock_sleep.call_args_list[1][0][0] - 0.225) < 0.01
        # Data should be persisted correctly
        data = json.loads(Path(state_mgr.history_file).read_text())
        assert len(data) == 1
        assert data[0]["task_description"] == "Retry test"

    def test_save_history_all_retries_fail(self, state_mgr):
        """_save_history should raise OSError when all retries fail."""
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
        # Should have attempted all 7 retries (exponential backoff)
        assert call_count == 7


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

    def test_cost_aware_no_grow_above_ceiling(self, tmp_path, default_config):
        """Batch size should not grow when cost exceeds batch_cost_ceiling."""
        default_config.paths.history_file = str(tmp_path / "history.json")
        default_config.orchestrator.batch_cost_ceiling = 5.0
        mgr = StateManager(default_config)

        now = time.time()
        # Record successes with high cost
        for i in range(5):
            mgr.record_cycle(CycleRecord(
                timestamp=now + i,
                task_description=f"Expensive task {i}",
                success=True,
                cost_usd=6.0,  # Above ceiling
            ))
        # initial=3, no growth for any of these (all above ceiling)
        assert mgr.compute_adaptive_batch_size() == 3

    def test_cost_aware_grows_below_ceiling(self, tmp_path, default_config):
        """Batch size should grow when cost is below batch_cost_ceiling."""
        default_config.paths.history_file = str(tmp_path / "history.json")
        default_config.orchestrator.batch_cost_ceiling = 10.0
        mgr = StateManager(default_config)

        now = time.time()
        for i in range(5):
            mgr.record_cycle(CycleRecord(
                timestamp=now + i,
                task_description=f"Cheap task {i}",
                success=True,
                cost_usd=2.0,  # Below ceiling
            ))
        # initial=3 + 5*1 = 8
        assert mgr.compute_adaptive_batch_size() == 8

    def test_cost_aware_mixed_costs(self, tmp_path, default_config):
        """Mixed cost cycles: cheap ones grow, expensive ones hold steady."""
        default_config.paths.history_file = str(tmp_path / "history.json")
        default_config.orchestrator.batch_cost_ceiling = 5.0
        mgr = StateManager(default_config)

        now = time.time()
        # Cheap success (+1), expensive success (hold), cheap success (+1)
        mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="Cheap", success=True, cost_usd=2.0,
        ))
        mgr.record_cycle(CycleRecord(
            timestamp=now + 1, task_description="Expensive", success=True, cost_usd=7.0,
        ))
        mgr.record_cycle(CycleRecord(
            timestamp=now + 2, task_description="Cheap again", success=True, cost_usd=1.0,
        ))
        # initial=3 + 1 (cheap) + 0 (expensive, held) + 1 (cheap) = 5
        assert mgr.compute_adaptive_batch_size() == 5


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
    @staticmethod
    def _find_corrupt_backup(history_file):
        """Find the timestamped .corrupt.* backup file next to history_file."""
        parent = Path(history_file).parent
        base_name = Path(history_file).name
        backups = list(parent.glob(f"{base_name}.corrupt.*"))
        return backups[0] if backups else None

    def test_corrupt_history_backed_up(self, state_mgr):
        """Corrupted JSON history file is backed up before returning empty."""
        corrupted_content = "{this is not valid json!!"
        Path(state_mgr.history_file).write_text(corrupted_content)

        result = state_mgr._load_history()

        assert result == []
        assert state_mgr.get_consecutive_failures() == 0
        backup_path = self._find_corrupt_backup(state_mgr.history_file)
        assert backup_path is not None
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

        backup_path = self._find_corrupt_backup(state_mgr.history_file)
        assert backup_path is not None
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
        backup_path = self._find_corrupt_backup(state_mgr.history_file)
        assert backup_path is not None

        # Remove backup to verify second call doesn't recreate it
        backup_path.unlink()

        # Second call: should use cache, not re-read file
        result = state_mgr._load_history()
        assert result == []
        # No new backup should be created
        assert self._find_corrupt_backup(state_mgr.history_file) is None

    def test_corrupt_backups_have_unique_timestamps(self, state_mgr):
        """Multiple corruptions create separate timestamped backup files."""
        corrupted1 = "{corrupt1!!"
        Path(state_mgr.history_file).write_text(corrupted1)
        state_mgr._cache = None  # force re-read
        state_mgr._cache_mtime = 0.0
        state_mgr._load_history()

        # Simulate a second corruption with a different timestamp
        corrupted2 = "{corrupt2!!"
        Path(state_mgr.history_file).write_text(corrupted2)
        state_mgr._cache = None
        state_mgr._cache_mtime = 0.0
        # Patch time.time to get a different timestamp
        with patch("state.time.time", return_value=time.time() + 1):
            state_mgr._load_history()

        parent = Path(state_mgr.history_file).parent
        base_name = Path(state_mgr.history_file).name
        backups = list(parent.glob(f"{base_name}.corrupt.*"))
        assert len(backups) >= 2


class TestSaveHistoryFdLeak:
    def test_save_history_fd_leak_on_fdopen_failure(self, state_mgr):
        """If os.fdopen raises, the raw fd should be explicitly closed."""
        original_mkstemp = tempfile.mkstemp

        # Track the fd allocated by mkstemp
        allocated_fd = None

        def tracking_mkstemp(**kwargs):
            nonlocal allocated_fd
            fd, path = original_mkstemp(**kwargs)
            allocated_fd = fd
            return fd, path

        with patch("state.tempfile.mkstemp", side_effect=tracking_mkstemp):
            with patch("state.os.fdopen", side_effect=OSError("fdopen failed")):
                with patch("state.os.close") as mock_close:
                    with pytest.raises(OSError, match="fdopen failed"):
                        state_mgr._save_history([{"test": True}])
                    # os.close should have been called with the raw fd
                    mock_close.assert_called_once_with(allocated_fd)

    def test_save_history_json_dumps_failure_returns_early(self, state_mgr, caplog):
        """If json.dumps raises (non-serializable data), save returns early with a log."""
        import logging

        class NotSerializable:
            pass

        with caplog.at_level(logging.ERROR):
            state_mgr._save_history([{"obj": NotSerializable()}])

        assert any("not JSON-serializable" in r.message for r in caplog.records)
        # Verify no leftover .tmp files in the history directory
        tmp_files = list(Path(state_mgr.history_file).parent.glob("*.tmp"))
        assert tmp_files == []


class TestSaveHistoryENOSPC:
    def test_save_history_enospc_logs_warning(self, state_mgr, caplog):
        """ENOSPC during os.replace should log a warning and not raise."""
        import logging

        enospc_err = OSError(errno.ENOSPC, "No space left on device")

        with patch("state.os.replace", side_effect=enospc_err):
            with caplog.at_level(logging.WARNING):
                # Should NOT raise
                state_mgr._save_history([{"test": True}])

        assert any("Disk full" in r.message for r in caplog.records)

    def test_save_history_enospc_cleans_temp_file(self, state_mgr):
        """ENOSPC should clean up the temp file."""
        enospc_err = OSError(errno.ENOSPC, "No space left on device")

        with patch("state.os.replace", side_effect=enospc_err):
            state_mgr._save_history([{"test": True}])

        # No temp files should remain
        tmp_files = list(Path(state_mgr.history_file).parent.glob("*.tmp"))
        assert tmp_files == []

    def test_save_history_non_enospc_oserror_still_raises(self, state_mgr):
        """Non-ENOSPC OSError should still propagate."""
        perm_err = OSError(errno.EACCES, "Permission denied")

        with patch("state.os.replace", side_effect=perm_err):
            with pytest.raises(OSError, match="Permission denied"):
                state_mgr._save_history([{"test": True}])

    def test_save_history_enospc_invalidates_cache(self, state_mgr):
        """ENOSPC must invalidate the in-memory cache to prevent stale reads."""
        # Prime the cache with some data
        state_mgr._cache = [{"old": "data"}]
        state_mgr._cache_mtime = 12345.0

        enospc_err = OSError(errno.ENOSPC, "No space left on device")
        with patch("state.os.replace", side_effect=enospc_err):
            state_mgr._save_history([{"new": "data"}])

        # Cache must be invalidated so next _load_history re-reads from disk
        assert state_mgr._cache is None
        assert state_mgr._cache_mtime == 0.0


class TestDiskSpacePreCheck:
    def test_low_disk_space_skips_save(self, state_mgr, caplog):
        """When disk space is below 10 MB, save should be skipped with a warning."""
        import logging
        from collections import namedtuple

        DiskUsage = namedtuple("DiskUsage", ["total", "used", "free"])
        low_space = DiskUsage(total=100 * 1024 * 1024, used=95 * 1024 * 1024, free=5 * 1024 * 1024)

        with patch("state.shutil.disk_usage", return_value=low_space):
            with caplog.at_level(logging.WARNING):
                state_mgr._save_history([{"test": True}])

        assert any("Low disk space" in r.message for r in caplog.records)
        # File should NOT have been written
        assert not Path(state_mgr.history_file).exists()

    def test_sufficient_disk_space_proceeds(self, state_mgr):
        """When disk space is above 10 MB, save should proceed normally."""
        from collections import namedtuple

        DiskUsage = namedtuple("DiskUsage", ["total", "used", "free"])
        plenty = DiskUsage(total=100 * 1024 * 1024, used=50 * 1024 * 1024, free=50 * 1024 * 1024)

        with patch("state.shutil.disk_usage", return_value=plenty):
            state_mgr._save_history([{"task": "test", "success": True}])

        assert Path(state_mgr.history_file).exists()
        data = json.loads(Path(state_mgr.history_file).read_text())
        assert len(data) == 1

    def test_disk_check_failure_continues(self, state_mgr, caplog):
        """If shutil.disk_usage itself raises, save should still proceed."""
        import logging

        with patch("state.shutil.disk_usage", side_effect=OSError("no mount")):
            with caplog.at_level(logging.DEBUG):
                state_mgr._save_history([{"task": "test", "success": True}])

        # Save should have completed despite the disk check failure
        assert Path(state_mgr.history_file).exists()
        data = json.loads(Path(state_mgr.history_file).read_text())
        assert len(data) == 1


class TestSaveHistoryRaceDetection:
    def test_external_modification_logs_warning(self, state_mgr, caplog):
        """Writing after external file modification should log a warning."""
        import logging

        # Write initial record to populate cache and mtime
        state_mgr.record_cycle(CycleRecord(
            timestamp=time.time(),
            task_description="First record",
            success=True,
        ))

        # Externally modify the file to change its mtime without going
        # through StateManager (simulates another process writing)
        time.sleep(0.05)  # Ensure mtime differs
        Path(state_mgr.history_file).write_text(
            json.dumps([{"timestamp": time.time(), "task_description": "External", "success": True}])
        )

        # Directly call _save_history to bypass _load_history (which would
        # update _cache_mtime). This simulates the race: read happened
        # before the external modification, write happens after.
        with caplog.at_level(logging.WARNING):
            state_mgr._save_history([{"timestamp": time.time(), "task_description": "Second", "success": True}])

        assert any(
            "modified externally" in r.message for r in caplog.records
        )

    def test_no_warning_on_normal_write(self, state_mgr, caplog):
        """Sequential writes without external modification should not warn."""
        import logging

        with caplog.at_level(logging.WARNING):
            state_mgr.record_cycle(CycleRecord(
                timestamp=time.time(),
                task_description="Record A",
                success=True,
            ))
            state_mgr.record_cycle(CycleRecord(
                timestamp=time.time(),
                task_description="Record B",
                success=True,
            ))

        assert not any(
            "modified externally" in r.message for r in caplog.records
        )


class TestRecentTaskSummaries:
    def test_empty_history(self, state_mgr):
        assert state_mgr.get_recent_task_summaries() == []

    def test_returns_recent_tasks(self, state_mgr):
        now = time.time()
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="Fix bug A", success=True,
        ))
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="Fix bug B", success=False,
        ))
        summaries = state_mgr.get_recent_task_summaries()
        assert len(summaries) == 2
        assert "- Fix bug A (succeeded)" in summaries[0]
        assert "- Fix bug B (failed)" in summaries[1]

    def test_respects_lookback(self, state_mgr):
        now = time.time()
        state_mgr.record_cycle(CycleRecord(
            timestamp=now - 200000, task_description="Old task", success=True,
        ))
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="Recent task", success=True,
        ))
        summaries = state_mgr.get_recent_task_summaries(lookback_seconds=3600)
        assert len(summaries) == 1
        assert "Recent task" in summaries[0]

    def test_respects_max_items(self, state_mgr):
        now = time.time()
        for i in range(10):
            state_mgr.record_cycle(CycleRecord(
                timestamp=now + i, task_description=f"Task {i}", success=True,
            ))
        summaries = state_mgr.get_recent_task_summaries(max_items=3)
        assert len(summaries) == 3
        # Should be the most recent 3
        assert "Task 7" in summaries[0]
        assert "Task 8" in summaries[1]
        assert "Task 9" in summaries[2]

    def test_truncates_long_descriptions(self, state_mgr):
        now = time.time()
        long_desc = "A" * 150
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description=long_desc, success=True,
        ))
        summaries = state_mgr.get_recent_task_summaries()
        assert len(summaries) == 1
        assert summaries[0].endswith("... (succeeded)")
        # 97 chars of description + "..."
        assert "A" * 97 + "..." in summaries[0]

    def test_early_termination_with_old_records(self, state_mgr):
        """Should stop early when encountering many consecutive old records."""
        now = time.time()
        # Add 10 old records
        for i in range(10):
            state_mgr.record_cycle(CycleRecord(
                timestamp=now - 200000 + i,
                task_description=f"Old task {i}",
                success=True,
            ))
        # Add 2 recent records
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="Recent A", success=True,
        ))
        state_mgr.record_cycle(CycleRecord(
            timestamp=now + 1, task_description="Recent B", success=False,
        ))
        summaries = state_mgr.get_recent_task_summaries(lookback_seconds=3600)
        assert len(summaries) == 2
        assert "Recent A" in summaries[0]
        assert "Recent B" in summaries[1]


class TestSuccessRateByType:
    def test_empty_history(self, state_mgr):
        assert state_mgr.get_success_rate_by_type() == {}

    def test_mixed_success_fail(self, state_mgr):
        now = time.time()
        for i in range(3):
            state_mgr.record_cycle(CycleRecord(
                timestamp=now + i, task_description=f"Test {i}",
                task_type="test_failure", success=(i < 2),
            ))
        for i in range(4):
            state_mgr.record_cycle(CycleRecord(
                timestamp=now + 10 + i, task_description=f"Lint {i}",
                task_type="lint", success=(i == 0),
            ))
        rates = state_mgr.get_success_rate_by_type()
        assert abs(rates["test_failure"] - 2 / 3) < 0.01
        assert abs(rates["lint"] - 1 / 4) < 0.01

    def test_respects_lookback(self, state_mgr):
        now = time.time()
        # Old record outside lookback window
        state_mgr.record_cycle(CycleRecord(
            timestamp=now - 200000, task_description="Old",
            task_type="todo", success=True,
        ))
        state_mgr.record_cycle(CycleRecord(
            timestamp=now - 200001, task_description="Old2",
            task_type="todo", success=False,
        ))
        # Recent records
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="Recent A",
            task_type="feedback", success=True,
        ))
        state_mgr.record_cycle(CycleRecord(
            timestamp=now + 1, task_description="Recent B",
            task_type="feedback", success=True,
        ))
        rates = state_mgr.get_success_rate_by_type(lookback_seconds=3600)
        # Old "todo" records should be excluded (outside lookback)
        assert "todo" not in rates
        assert abs(rates["feedback"] - 1.0) < 0.01

    def test_skips_types_with_fewer_than_two_attempts(self, state_mgr):
        now = time.time()
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="Solo",
            task_type="rare_type", success=True,
        ))
        state_mgr.record_cycle(CycleRecord(
            timestamp=now + 1, task_description="A",
            task_type="common_type", success=True,
        ))
        state_mgr.record_cycle(CycleRecord(
            timestamp=now + 2, task_description="B",
            task_type="common_type", success=False,
        ))
        rates = state_mgr.get_success_rate_by_type()
        assert "rare_type" not in rates
        assert "common_type" in rates


class TestStrategyPerformance:
    def test_empty_history(self, state_mgr):
        assert state_mgr.get_strategy_performance() == {}

    def test_computes_avg_cost_duration_success_rate(self, state_mgr):
        now = time.time()
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="A", task_type="feedback",
            success=True, cost_usd=1.0, duration_seconds=100.0,
        ))
        state_mgr.record_cycle(CycleRecord(
            timestamp=now + 1, task_description="B", task_type="feedback",
            success=False, cost_usd=3.0, duration_seconds=200.0,
        ))
        perf = state_mgr.get_strategy_performance()
        fb = perf["feedback"]
        assert fb["total"] == 2
        assert fb["successes"] == 1
        assert abs(fb["success_rate"] - 0.5) < 0.01
        assert abs(fb["avg_cost"] - 2.0) < 0.01
        assert abs(fb["avg_duration"] - 150.0) < 0.01

    def test_respects_lookback(self, state_mgr):
        now = time.time()
        state_mgr.record_cycle(CycleRecord(
            timestamp=now - 200000, task_description="Old",
            task_type="test_failure", success=True, cost_usd=5.0,
        ))
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="Recent",
            task_type="feedback", success=True, cost_usd=1.0,
        ))
        perf = state_mgr.get_strategy_performance(lookback_seconds=3600)
        assert "test_failure" not in perf
        assert "feedback" in perf

    def test_multiple_sources(self, state_mgr):
        now = time.time()
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="A", task_type="feedback",
            success=True, cost_usd=1.0, duration_seconds=60.0,
        ))
        state_mgr.record_cycle(CycleRecord(
            timestamp=now + 1, task_description="B", task_type="lint",
            success=False, cost_usd=0.5, duration_seconds=30.0,
        ))
        perf = state_mgr.get_strategy_performance()
        assert "feedback" in perf
        assert "lint" in perf
        assert perf["feedback"]["success_rate"] == 1.0
        assert perf["lint"]["success_rate"] == 0.0


class TestTaskSuccessHistory:
    def test_empty_history(self, state_mgr):
        result = state_mgr.get_task_success_history("Fix bug")
        assert result == []

    def test_returns_matching_attempts(self, state_mgr):
        now = time.time()
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="Fix bug",
            success=False, error="SyntaxError in foo.py",
            validation_summary="tests: FAIL",
        ))
        state_mgr.record_cycle(CycleRecord(
            timestamp=now + 1, task_description="Fix bug",
            success=True, validation_summary="tests: PASS",
        ))
        result = state_mgr.get_task_success_history("Fix bug")
        assert len(result) == 2
        assert result[0]["success"] is False
        assert result[0]["error"] == "SyntaxError in foo.py"
        assert result[1]["success"] is True

    def test_matches_by_task_key(self, state_mgr):
        now = time.time()
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="Fix bug v1",
            success=False, error="assertion failed",
            task_keys=["test:foo.py"],
        ))
        result = state_mgr.get_task_success_history(
            "Fix bug v2", task_key="test:foo.py",
        )
        assert len(result) == 1
        assert result[0]["error"] == "assertion failed"

    def test_matches_batch_descriptions(self, state_mgr):
        now = time.time()
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="Batch task",
            success=False, error="lint failed",
            task_descriptions=["Fix A", "Fix B"],
        ))
        result = state_mgr.get_task_success_history("Fix A")
        assert len(result) == 1

    def test_respects_max_attempts(self, state_mgr):
        now = time.time()
        for i in range(10):
            state_mgr.record_cycle(CycleRecord(
                timestamp=now + i, task_description="Fix bug",
                success=False, error=f"error {i}",
            ))
        result = state_mgr.get_task_success_history("Fix bug", max_attempts=3)
        assert len(result) == 3
        # Should return the most recent 3
        assert result[0]["error"] == "error 7"
        assert result[2]["error"] == "error 9"

    def test_ignores_unrelated_tasks(self, state_mgr):
        now = time.time()
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="Other task",
            success=False, error="other error",
        ))
        state_mgr.record_cycle(CycleRecord(
            timestamp=now + 1, task_description="Fix bug",
            success=False, error="relevant error",
        ))
        result = state_mgr.get_task_success_history("Fix bug")
        assert len(result) == 1
        assert result[0]["error"] == "relevant error"


class TestProductiveFiles:
    def test_empty_history(self, state_mgr):
        assert state_mgr.get_productive_files() == []

    def test_extracts_files_from_descriptions(self, state_mgr):
        now = time.time()
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="Fix bug in safety.py and state.py",
            success=True,
            task_descriptions=["Fix bug in safety.py and state.py"],
        ))
        state_mgr.record_cycle(CycleRecord(
            timestamp=now + 1, task_description="Update safety.py error handling",
            success=True,
            task_descriptions=["Update safety.py error handling"],
        ))
        files = state_mgr.get_productive_files()
        assert files[0] == "safety.py"  # Most frequent
        assert "state.py" in files

    def test_ignores_failed_cycles(self, state_mgr):
        now = time.time()
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="Fix bug in never_seen.py",
            success=False,
            task_descriptions=["Fix bug in never_seen.py"],
        ))
        state_mgr.record_cycle(CycleRecord(
            timestamp=now + 1, task_description="Fix bug in seen.py",
            success=True,
            task_descriptions=["Fix bug in seen.py"],
        ))
        files = state_mgr.get_productive_files()
        assert "never_seen.py" not in files
        assert "seen.py" in files

    def test_sorted_by_frequency(self, state_mgr):
        now = time.time()
        # a.py mentioned 3 times, b.py mentioned 1 time
        for i in range(3):
            state_mgr.record_cycle(CycleRecord(
                timestamp=now + i, task_description="Work on a.py",
                success=True,
                task_descriptions=["Work on a.py"],
            ))
        state_mgr.record_cycle(CycleRecord(
            timestamp=now + 10, task_description="Work on b.py",
            success=True,
            task_descriptions=["Work on b.py"],
        ))
        files = state_mgr.get_productive_files()
        assert files.index("a.py") < files.index("b.py")

    def test_respects_lookback(self, state_mgr):
        now = time.time()
        state_mgr.record_cycle(CycleRecord(
            timestamp=now - 200000, task_description="Fix old.py",
            success=True,
            task_descriptions=["Fix old.py"],
        ))
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="Fix recent.py",
            success=True,
            task_descriptions=["Fix recent.py"],
        ))
        files = state_mgr.get_productive_files(lookback_seconds=3600)
        assert "old.py" not in files
        assert "recent.py" in files


class TestStrategyPerformanceReport:
    def test_empty_history(self, state_mgr):
        report = state_mgr.get_strategy_performance_report()
        assert report == "No task history in the last 24 hours."

    def test_report_format(self, state_mgr):
        now = time.time()
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="Fix test",
            task_type="test_failure", success=True,
            cost_usd=0.10, duration_seconds=30.0,
        ))
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="Fix test 2",
            task_type="test_failure", success=False,
            cost_usd=0.20, duration_seconds=60.0,
        ))
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="Fix lint",
            task_type="lint", success=True,
            cost_usd=0.05, duration_seconds=10.0,
        ))
        report = state_mgr.get_strategy_performance_report()
        assert "Strategy Performance (last 24h):" in report
        # lint has 100% success rate, should appear first
        assert report.index("lint") < report.index("test_failure")
        assert "1/1 succeeded" in report  # lint
        assert "1/2 succeeded" in report  # test_failure
        assert "avg cost" in report
        assert "avg duration" in report

    def test_old_records_excluded(self, state_mgr):
        old = time.time() - 200000
        state_mgr.record_cycle(CycleRecord(
            timestamp=old, task_description="Old task",
            task_type="lint", success=True,
            cost_usd=0.10, duration_seconds=20.0,
        ))
        report = state_mgr.get_strategy_performance_report()
        assert report == "No task history in the last 24 hours."


class TestTryRestoreFromBackupsTOCTOU:
    """Tests for TOCTOU race condition fix in _try_restore_from_backups."""

    def test_backup_deleted_during_sort(self, state_mgr):
        """If a .corrupt file is deleted between glob() and stat(), sorting should not crash."""
        parent = Path(state_mgr.history_file).parent
        parent.mkdir(parents=True, exist_ok=True)

        # Create two backup files
        backup1 = parent / "history.json.corrupt"
        backup2 = parent / "history.json.corrupt.1"
        backup1.write_text(json.dumps([{"task_description": "backup1", "timestamp": 1.0}]))
        backup2.write_text(json.dumps([{"task_description": "backup2", "timestamp": 2.0}]))

        # Patch stat to simulate deletion of backup2 during sorting
        original_stat = Path.stat

        call_count = 0

        def flaky_stat(self_path):
            nonlocal call_count
            if "corrupt.1" in str(self_path):
                call_count += 1
                if call_count == 1:
                    raise OSError("No such file or directory")
            return original_stat(self_path)

        with patch.object(Path, 'stat', flaky_stat):
            result = state_mgr._try_restore_from_backups()

        # Should succeed with backup1 (the one that didn't fail stat)
        assert result is not None
        assert len(result) == 1
        assert result[0]["task_description"] == "backup1"

    def test_all_backups_deleted_during_sort(self, state_mgr):
        """If all .corrupt files are deleted during sort, should return None gracefully."""
        parent = Path(state_mgr.history_file).parent
        parent.mkdir(parents=True, exist_ok=True)

        backup1 = parent / "history.json.corrupt"
        backup1.write_text(json.dumps([{"task_description": "backup1"}]))

        original_stat = Path.stat

        def always_fail_stat(self_path):
            if "corrupt" in str(self_path):
                raise OSError("No such file or directory")
            return original_stat(self_path)

        with patch.object(Path, 'stat', always_fail_stat):
            result = state_mgr._try_restore_from_backups()

        # No backups survived stat, so should return None
        assert result is None


class TestFilePatternRegex:
    def test_file_pattern_regex_is_precompiled(self):
        """The file pattern regex should be pre-compiled at module level for performance."""
        import state
        assert hasattr(state, '_FILE_PATTERN_RE'), (
            "state module should have a _FILE_PATTERN_RE pre-compiled regex"
        )
        import re
        assert isinstance(state._FILE_PATTERN_RE, re.Pattern)

    def test_productive_files_uses_precompiled_regex(self, state_mgr):
        """get_productive_files should use the pre-compiled regex."""
        import time
        from state import CycleRecord
        record = CycleRecord(
            timestamp=time.time(),
            task_description="Fix bug in main.py and config.yaml",
            success=True,
            cost_usd=0.01,
        )
        state_mgr.record_cycle(record)
        files = state_mgr.get_productive_files()
        assert "main.py" in files
        assert "config.yaml" in files

    def test_transient_io_error_preserves_cached_history(self, state_mgr):
        """A transient OSError in _load_history should return cached data, not empty list."""
        import time
        from state import CycleRecord
        # Write some history
        for i in range(3):
            state_mgr.record_cycle(CycleRecord(
                timestamp=time.time(),
                task_description=f"Task {i}",
                success=True,
            ))
        # Verify we have 3 records cached
        assert len(state_mgr._load_history()) == 3

        # Now simulate a transient IO error during _load_history
        original_stat = Path.stat
        call_count = 0
        def failing_stat(self_path, *a, **kw):
            nonlocal call_count
            call_count += 1
            if str(self_path).endswith("history.json") and call_count > 6:
                raise OSError("Transient NFS error")
            return original_stat(self_path, *a, **kw)

        with patch.object(Path, "stat", failing_stat):
            result = state_mgr._load_history()
        # Should return the cached 3 records, not an empty list
        assert len(result) == 3


class TestLoadHistoryTypeValidation:
    """Tests that _load_history validates the deserialized JSON is a list."""

    def test_dict_history_treated_as_corrupt(self, state_mgr):
        """If history.json contains a JSON object instead of array, treat as corrupt."""
        state_mgr.history_file.write_text('{"not": "a list"}')
        state_mgr._cache = None
        result = state_mgr._load_history()
        # Should return empty list (corrupt data), not a dict
        assert isinstance(result, list)

    def test_string_history_treated_as_corrupt(self, state_mgr):
        """If history.json contains a JSON string, treat as corrupt."""
        state_mgr.history_file.write_text('"just a string"')
        state_mgr._cache = None
        result = state_mgr._load_history()
        assert isinstance(result, list)

    def test_valid_list_history_loads_normally(self, state_mgr):
        """A valid JSON array loads normally."""
        import json, time
        records = [{"timestamp": time.time(), "task_description": "test", "success": True}]
        state_mgr.history_file.write_text(json.dumps(records))
        state_mgr._cache = None
        result = state_mgr._load_history()
        assert isinstance(result, list)
        assert len(result) == 1

    def test_history_with_unicode_content(self, state_mgr):
        """History with non-ASCII content should round-trip correctly."""
        record = CycleRecord(
            timestamp=time.time(),
            task_description="Fix résumé parsing in café.py",
            success=True,
        )
        state_mgr.record_cycle(record)
        state_mgr._cache = None  # force re-read from disk
        records = state_mgr._load_history()
        assert len(records) == 1
        assert "résumé" in records[0]["task_description"]
        assert "café" in records[0]["task_description"]


class TestRecursionErrorHandling:
    """Verify RecursionError during JSON serialization is caught gracefully."""

    def test_recursive_structure_does_not_crash(self, state_mgr):
        """A self-referencing data structure should be caught by the save guard."""
        record = CycleRecord(
            timestamp=time.time(),
            task_description="Recursive task",
            success=True,
        )
        state_mgr.record_cycle(record)
        # Now inject a self-referencing structure into the cache
        evil = []
        evil.append(evil)
        state_mgr._cache = [{"task_description": "ok", "extra": evil}]
        # _save_history should catch RecursionError, not crash
        state_mgr._save_history(state_mgr._cache)
        # The file should still contain the original valid record
        state_mgr._cache = None
        records = state_mgr._load_history()
        assert len(records) == 1
        assert records[0]["task_description"] == "Recursive task"


class TestReverseIterationOptimization:
    """Test that get_cycle_count_last_hour and get_total_cost use reverse iteration.

    Records are chronologically ordered, so scanning in reverse and breaking
    early when we pass the cutoff avoids scanning the entire history.
    """

    def test_cycle_count_last_hour_skips_old_records(self, tmp_path, default_config):
        """get_cycle_count_last_hour should still return correct count."""
        default_config.paths.history_file = str(tmp_path / "history.json")
        sm = StateManager(default_config)
        now = time.time()
        records = [
            {"timestamp": now - 7200, "task_description": "old1", "success": True},
            {"timestamp": now - 3601, "task_description": "old2", "success": True},
            {"timestamp": now - 1800, "task_description": "recent1", "success": True},
            {"timestamp": now - 60, "task_description": "recent2", "success": True},
        ]
        sm._save_history(records)
        assert sm.get_cycle_count_last_hour() == 2

    def test_total_cost_skips_old_records(self, tmp_path, default_config):
        """get_total_cost should sum only recent records."""
        default_config.paths.history_file = str(tmp_path / "history.json")
        sm = StateManager(default_config)
        now = time.time()
        records = [
            {"timestamp": now - 7200, "task_description": "old", "cost_usd": 100.0},
            {"timestamp": now - 1800, "task_description": "recent1", "cost_usd": 0.5},
            {"timestamp": now - 60, "task_description": "recent2", "cost_usd": 0.3},
        ]
        sm._save_history(records)
        total = sm.get_total_cost(lookback_seconds=3600)
        assert abs(total - 0.8) < 0.001

    def test_uses_reversed_iteration(self):
        """Implementation should use reversed() for early exit on old records."""
        import inspect
        source = inspect.getsource(StateManager.get_cycle_count_last_hour)
        assert "reversed(" in source, (
            "get_cycle_count_last_hour should iterate in reverse for early exit"
        )


class TestTaskFailureCountLookback:
    """Test that get_task_failure_count respects its lookback window."""

    def test_old_failures_excluded_by_lookback(self, tmp_path, default_config):
        """Failures older than lookback_seconds should not be counted."""
        default_config.paths.history_file = str(tmp_path / "history.json")
        sm = StateManager(default_config)
        now = time.time()
        records = [
            {"timestamp": now - 100000, "task_description": "Fix bug",
             "task_type": "feedback", "success": False},
            {"timestamp": now - 60, "task_description": "Fix bug",
             "task_type": "feedback", "success": False},
        ]
        sm._save_history(records)
        # With 24h lookback (default), only the recent failure counts
        assert sm.get_task_failure_count("Fix bug") == 1
        # With a very large lookback, both count
        assert sm.get_task_failure_count("Fix bug", lookback_seconds=200000) == 2

    def test_default_lookback_is_24h(self, tmp_path, default_config):
        """Default lookback should be 86400 seconds (24 hours)."""
        import inspect
        sig = inspect.signature(StateManager.get_task_failure_count)
        assert sig.parameters["lookback_seconds"].default == 86400


class TestDashboardSinglePassCompute:
    """Test that compute_status computes all metrics in a single pass."""

    def test_single_pass_implementation(self):
        """compute_status should iterate records only once, not in 4 passes."""
        import inspect
        from dashboard import compute_status
        source = inspect.getsource(compute_status)
        # Should NOT have separate sum() comprehensions for each metric
        assert source.count("sum(1 for r in records") == 0, (
            "compute_status should use a single pass, not separate sum() calls"
        )
        assert source.count("sum(r.get") == 0, (
            "compute_status should use a single pass, not separate sum() calls"
        )


class TestHistoryFilePermissions:
    """Test that _save_history restricts temp file permissions to 0o600."""

    def test_save_history_calls_fchmod(self):
        """_save_history should call os.fchmod to restrict temp file permissions."""
        import inspect
        from state import StateManager
        source = inspect.getsource(StateManager._save_history)
        assert "fchmod" in source, (
            "_save_history should call os.fchmod to restrict temp file permissions"
        )

    def test_history_file_not_world_readable(self, tmp_path, default_config):
        """After saving, the history file should not be world-readable."""
        from state import StateManager
        history_file = tmp_path / "state" / "history.json"
        default_config.paths.history_file = str(history_file)
        sm = StateManager(default_config)
        sm._save_history([{"task_description": "test", "timestamp": "now"}])
        assert history_file.exists()
        mode = history_file.stat().st_mode & 0o777
        # Should not have group or other read/write bits
        assert mode & 0o077 == 0, (
            f"History file should not be readable by group/others, got {oct(mode)}"
        )


class TestOutOfOrderTimestamps:
    """Verify methods handle out-of-order timestamps from parallel workers."""

    def test_was_recently_attempted_finds_record_after_older_timestamp(self, state_mgr):
        """was_recently_attempted should not miss records when timestamps are unordered."""
        now = time.time()
        # Record A: recent
        state_mgr.record_cycle(CycleRecord(
            timestamp=now - 100, task_description="task-A", success=True, cost_usd=0.01,
        ))
        # Record B: old timestamp (written by a lagging parallel worker)
        state_mgr.record_cycle(CycleRecord(
            timestamp=now - 7200, task_description="task-old", success=True, cost_usd=0.01,
        ))
        # Record C: recent, appears after the old record
        state_mgr.record_cycle(CycleRecord(
            timestamp=now - 50, task_description="task-C", success=True, cost_usd=0.01,
        ))

        # task-C should be found even though an old record sits between A and C
        assert state_mgr.was_recently_attempted("task-C", lookback_seconds=3600)

    def test_get_task_failure_count_with_unordered_timestamps(self, state_mgr):
        """get_task_failure_count should count all failures within the window."""
        now = time.time()
        # Failure 1: recent
        state_mgr.record_cycle(CycleRecord(
            timestamp=now - 100, task_description="flaky", success=False, cost_usd=0.01,
        ))
        # Old record from a lagging worker
        state_mgr.record_cycle(CycleRecord(
            timestamp=now - 90000, task_description="other", success=True, cost_usd=0.01,
        ))
        # Failure 2: recent, appears after the old record
        state_mgr.record_cycle(CycleRecord(
            timestamp=now - 50, task_description="flaky", success=False, cost_usd=0.01,
        ))

        # Both failures should be counted
        assert state_mgr.get_task_failure_count("flaky", lookback_seconds=86400) == 2


class TestLoadHistoryReturnsCopy:
    """load_history() must return a copy so callers cannot corrupt the cache."""

    def test_mutating_returned_list_does_not_affect_cache(self, state_mgr):
        state_mgr.record_cycle(CycleRecord(
            timestamp=time.time(),
            task_description="Original",
            success=True,
        ))
        first = state_mgr.load_history()
        first.append({"fake": "record"})

        second = state_mgr.load_history()
        assert len(second) == 1, (
            "Appending to load_history() result must not mutate the cache"
        )

    def test_clearing_returned_list_does_not_affect_cache(self, state_mgr):
        state_mgr.record_cycle(CycleRecord(
            timestamp=time.time(),
            task_description="Keep me",
            success=True,
        ))
        result = state_mgr.load_history()
        result.clear()

        reloaded = state_mgr.load_history()
        assert len(reloaded) == 1, (
            "Clearing load_history() result must not wipe the cache"
        )


class TestEarlyTerminationInScans:
    """get_cycle_count_last_hour and get_total_cost should break early on old records."""

    def test_cycle_count_terminates_early(self, state_mgr):
        """Records older than cutoff + grace should not be scanned."""
        now = time.time()
        # Add old records (2 hours ago) and recent records
        for i in range(100):
            state_mgr.record_cycle(CycleRecord(
                timestamp=now - 7200 + i,  # 2 hours ago
                task_description=f"Old task {i}",
                success=True,
            ))
        for i in range(5):
            state_mgr.record_cycle(CycleRecord(
                timestamp=now - 60 + i,  # recent
                task_description=f"Recent task {i}",
                success=True,
            ))
        count = state_mgr.get_cycle_count_last_hour()
        assert count == 5

    def test_total_cost_terminates_early(self, state_mgr):
        """Total cost should only sum recent records."""
        now = time.time()
        for i in range(50):
            state_mgr.record_cycle(CycleRecord(
                timestamp=now - 7200 + i,
                task_description=f"Old task {i}",
                success=True,
                cost_usd=1.0,
            ))
        for i in range(3):
            state_mgr.record_cycle(CycleRecord(
                timestamp=now - 30 + i,
                task_description=f"Recent task {i}",
                success=True,
                cost_usd=0.5,
            ))
        total = state_mgr.get_total_cost(lookback_seconds=3600)
        assert total == pytest.approx(1.5)

    def test_out_of_order_records_tolerated(self, state_mgr):
        """A few out-of-order old records don't cause premature termination."""
        now = time.time()
        # Insert: 3 old, 1 recent, 2 old, 3 recent (in insertion order)
        for i in range(3):
            state_mgr.record_cycle(CycleRecord(
                timestamp=now - 7200 + i,
                task_description=f"Old batch1 {i}",
                success=True, cost_usd=1.0,
            ))
        state_mgr.record_cycle(CycleRecord(
            timestamp=now - 30,
            task_description="Recent 1",
            success=True, cost_usd=0.5,
        ))
        for i in range(2):
            state_mgr.record_cycle(CycleRecord(
                timestamp=now - 7200 + i,
                task_description=f"Old batch2 {i}",
                success=True, cost_usd=1.0,
            ))
        for i in range(3):
            state_mgr.record_cycle(CycleRecord(
                timestamp=now - 10 + i,
                task_description=f"Recent {i+2}",
                success=True, cost_usd=0.5,
            ))
        # 4 recent records total; early termination should not skip them
        count = state_mgr.get_cycle_count_last_hour()
        assert count == 4
        total = state_mgr.get_total_cost()
        assert total == pytest.approx(2.0)

    def test_was_recently_attempted_terminates_early(self, state_mgr):
        """was_recently_attempted should break early after consecutive old records."""
        now = time.time()
        # Insert many old records followed by recent records
        for i in range(100):
            state_mgr.record_cycle(CycleRecord(
                timestamp=now - 7200 + i,  # 2 hours ago
                task_description=f"Old task {i}",
                success=True,
            ))
        for i in range(5):
            state_mgr.record_cycle(CycleRecord(
                timestamp=now - 60 + i,  # recent
                task_description=f"Recent task {i}",
                success=True,
            ))
        # Recent task should be found
        assert state_mgr.was_recently_attempted("Recent task 3", lookback_seconds=3600)
        # Old task should not be found
        assert not state_mgr.was_recently_attempted("Old task 0", lookback_seconds=3600)

    def test_was_recently_attempted_tolerates_out_of_order(self, state_mgr):
        """A few out-of-order old records don't cause premature termination."""
        now = time.time()
        # 3 old, then 1 recent, then 2 old, then 3 recent
        for i in range(3):
            state_mgr.record_cycle(CycleRecord(
                timestamp=now - 7200 + i,
                task_description=f"Old batch1 {i}",
                success=True,
            ))
        state_mgr.record_cycle(CycleRecord(
            timestamp=now - 30,
            task_description="Target recent",
            success=True,
        ))
        for i in range(2):
            state_mgr.record_cycle(CycleRecord(
                timestamp=now - 7200 + i,
                task_description=f"Old batch2 {i}",
                success=True,
            ))
        for i in range(3):
            state_mgr.record_cycle(CycleRecord(
                timestamp=now - 10 + i,
                task_description=f"Recent {i}",
                success=True,
            ))
        # "Target recent" should still be found despite interleaved old records
        assert state_mgr.was_recently_attempted("Target recent", lookback_seconds=3600)


class TestHistoryFileSizeGuard:
    """_load_history must refuse to load oversized files to prevent OOM."""

    def test_oversized_history_returns_empty(self, state_mgr):
        """A history file exceeding the size limit should not be loaded."""
        # Write a file that's too large
        with patch.object(
            type(state_mgr), '_MAX_HISTORY_FILE_BYTES', 100
        ):
            # Write content larger than 100 bytes
            large_records = [{"timestamp": 1, "task_description": "x" * 200}]
            Path(state_mgr.history_file).write_text(
                json.dumps(large_records), encoding="utf-8"
            )
            # Invalidate cache
            state_mgr._cache = None
            state_mgr._cache_mtime = 0.0

            result = state_mgr._load_history()
            assert result == []

    def test_normal_sized_history_loads_fine(self, state_mgr):
        """Files under the size limit should load normally."""
        state_mgr.record_cycle(CycleRecord(
            timestamp=time.time(), task_description="Normal task", success=True,
        ))
        records = state_mgr._load_history()
        assert len(records) == 1
        assert records[0]["task_description"] == "Normal task"
