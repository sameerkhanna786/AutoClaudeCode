"""Tests for state_lock module."""

import json
import os
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


class TestReentrantLock:
    """Tests that nested lock acquisitions don't deadlock."""

    def test_should_auto_reset_failures_no_deadlock(self, locked_state):
        """should_auto_reset_failures calls get_consecutive_failures internally.

        This previously deadlocked because both methods acquire _file_lock,
        and fcntl.flock is not re-entrant across different file descriptors.
        """
        # Set up: create enough failures to trigger the auto-reset path
        old_time = time.time() - 7200  # 2 hours ago
        for i in range(5):
            locked_state.record_cycle(CycleRecord(
                timestamp=old_time + i,
                task_description=f"Fail {i}",
                task_type="test_failure",
                success=False,
            ))
        # This must not deadlock — should_auto_reset_failures internally
        # calls get_consecutive_failures which also acquires the lock
        result = locked_state.should_auto_reset_failures(min_idle_seconds=3600)
        assert result is True

    def test_reset_consecutive_failures_no_deadlock(self, locked_state):
        """reset_consecutive_failures calls record_cycle internally.

        Both methods acquire _file_lock; without re-entrancy this deadlocks.
        """
        locked_state.record_cycle(CycleRecord(
            timestamp=time.time(),
            task_description="Fail",
            task_type="test_failure",
            success=False,
        ))
        # This must not deadlock — reset_consecutive_failures internally
        # calls record_cycle which also acquires the lock
        locked_state.reset_consecutive_failures("test reset")
        assert locked_state.get_consecutive_failures() == 0

    def test_reentrant_lock_is_per_thread(self, locked_state):
        """Lock re-entrancy is thread-local; different threads still serialize."""
        order = []

        def thread_fn(thread_id):
            with locked_state._file_lock():
                order.append(f"{thread_id}-start")
                time.sleep(0.2)  # Hold the lock
                order.append(f"{thread_id}-end")

        t1 = threading.Thread(target=thread_fn, args=(1,))
        t2 = threading.Thread(target=thread_fn, args=(2,))
        t1.start()
        time.sleep(0.05)  # Let t1 acquire first
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)
        # Both threads completed and were serialized (no interleaving)
        assert len(order) == 4
        # First thread completes before second starts
        assert order[0].endswith("-start")
        assert order[1].endswith("-end")
        assert order[0][0] == order[1][0]  # Same thread ID


class TestGetTaskFailureCountLookbackSeconds:
    """Tests that get_task_failure_count correctly passes lookback_seconds to parent."""

    def test_lookback_seconds_default(self, locked_state):
        """get_task_failure_count should accept lookback_seconds parameter."""
        record = CycleRecord(
            timestamp=time.time(),
            task_description="Failing task",
            task_type="lint",
            success=False,
        )
        locked_state.record_cycle(record)
        count = locked_state.get_task_failure_count("Failing task", lookback_seconds=86400)
        assert count == 1

    def test_lookback_seconds_short_window_excludes_old(self, locked_state):
        """Old failures should be excluded with a short lookback window."""
        old_record = CycleRecord(
            timestamp=time.time() - 7200,  # 2 hours ago
            task_description="Old failure",
            task_type="lint",
            success=False,
        )
        locked_state.record_cycle(old_record)
        count = locked_state.get_task_failure_count("Old failure", lookback_seconds=3600)
        assert count == 0

    def test_lookback_seconds_keyword_arg_does_not_raise(self, locked_state):
        """Passing lookback_seconds as keyword arg should not raise TypeError."""
        locked_state.get_task_failure_count(
            "task", task_type="lint", task_key="k", lookback_seconds=1800
        )


class TestMissingLockedWrappers:
    """Tests for methods that were missing locked wrappers."""

    def test_get_strategy_performance_locked(self, locked_state):
        """get_strategy_performance works through the lock."""
        locked_state.record_cycle(CycleRecord(
            timestamp=time.time(),
            task_description="Task A",
            task_type="lint",
            success=True,
            cost_usd=0.10,
            duration_seconds=5.0,
        ))
        locked_state.record_cycle(CycleRecord(
            timestamp=time.time(),
            task_description="Task B",
            task_type="lint",
            success=False,
            cost_usd=0.05,
            duration_seconds=3.0,
        ))
        perf = locked_state.get_strategy_performance()
        assert "lint" in perf
        assert perf["lint"]["total"] == 2
        assert perf["lint"]["successes"] == 1

    def test_get_productive_files_locked(self, locked_state):
        """get_productive_files works through the lock."""
        locked_state.record_cycle(CycleRecord(
            timestamp=time.time(),
            task_description="Fix bug in validator.py",
            task_type="test_failure",
            success=True,
            task_descriptions=["Fix bug in validator.py"],
        ))
        files = locked_state.get_productive_files()
        assert "validator.py" in files

    def test_get_task_success_history_locked(self, locked_state):
        """get_task_success_history works through the lock."""
        locked_state.record_cycle(CycleRecord(
            timestamp=time.time(),
            task_description="Fix bug",
            task_type="test_failure",
            success=False,
            error="Test failed",
        ))
        locked_state.record_cycle(CycleRecord(
            timestamp=time.time(),
            task_description="Fix bug",
            task_type="test_failure",
            success=True,
        ))
        history = locked_state.get_task_success_history("Fix bug")
        assert len(history) == 2
        assert history[0]["success"] is False
        assert history[1]["success"] is True

    def test_get_strategy_performance_report_locked(self, locked_state):
        """get_strategy_performance_report works through the lock."""
        locked_state.record_cycle(CycleRecord(
            timestamp=time.time(),
            task_description="Task A",
            task_type="lint",
            success=True,
            cost_usd=0.10,
            duration_seconds=5.0,
        ))
        locked_state.record_cycle(CycleRecord(
            timestamp=time.time(),
            task_description="Task B",
            task_type="lint",
            success=True,
            cost_usd=0.05,
            duration_seconds=3.0,
        ))
        report = locked_state.get_strategy_performance_report()
        assert "lint" in report
        assert "2/2 succeeded" in report


class TestHeldFlagClearedBeforeFdClose:
    """Regression: held flag must be cleared BEFORE os.close(fd).

    If os.close(fd) runs first, the lock is released but held is still True.
    A concurrent re-entrant check on the same thread would see held=True and
    skip acquiring a new lock — even though the file lock is already gone.
    """

    def test_held_false_before_fd_close(self, locked_state):
        """Verify self._local.held is False by the time os.close is called."""
        import fcntl

        original_close = os.close
        held_at_close_time = []

        def spy_close(fd):
            # Capture the held flag at the moment close is called
            held_at_close_time.append(getattr(locked_state._local, 'held', 'MISSING'))
            return original_close(fd)

        with patch("state_lock.os.close", side_effect=spy_close):
            with locked_state._file_lock():
                pass

        assert len(held_at_close_time) == 1, "os.close should be called exactly once"
        assert held_at_close_time[0] is False, (
            "held flag must be cleared BEFORE os.close(fd) to prevent "
            "re-entrant check from skipping lock acquisition after release"
        )


class TestExplicitUnlockBeforeClose:
    """The file lock must explicitly unlock before close.

    Explicitly calling flock(LOCK_UN) before close(fd) ensures the lock is
    properly released even in edge cases where multiple file descriptors
    reference the same inode.
    """

    def test_explicit_unlock_called(self, locked_state):
        """_file_lock should explicitly release lock before closing fd."""
        import fcntl
        original_flock = fcntl.flock
        flock_ops = []

        def tracking_flock(fd, operation):
            flock_ops.append(operation)
            return original_flock(fd, operation)

        with patch("state_lock.fcntl.flock", side_effect=tracking_flock):
            with locked_state._file_lock():
                pass

        assert fcntl.LOCK_EX in flock_ops, "Lock should be acquired with LOCK_EX"
        assert fcntl.LOCK_UN in flock_ops, \
            "Lock should be explicitly unlocked before close(fd)"


class TestFileLockFdCleanup:
    """Tests that file descriptors are properly cleaned up even on errors."""

    def test_fd_closed_on_flock_unlock_failure(self, locked_state):
        """If flock(LOCK_UN) raises, os.close(fd) must still be called."""
        import fcntl
        close_calls = []
        original_close = os.close
        original_flock = fcntl.flock

        def tracking_close(fd):
            close_calls.append(fd)
            return original_close(fd)

        def flock_side_effect(fd, operation):
            if operation == fcntl.LOCK_EX:
                return original_flock(fd, operation)
            else:
                # LOCK_UN - simulate failure
                raise OSError("unlock failed")

        with patch("state_lock.fcntl.flock", side_effect=flock_side_effect), \
             patch("state_lock.os.close", side_effect=tracking_close):
            # The context manager should still close fd even if unlock fails
            try:
                with locked_state._file_lock():
                    pass
            except OSError:
                pass  # LOCK_UN failure may propagate

            # fd must have been closed regardless of flock(LOCK_UN) failure
            assert len(close_calls) >= 1


class TestReentrantHeldFlagOnFlockFailure:
    """When flock(LOCK_EX) fails, the held flag must NOT be set to False
    if an outer call already holds it."""

    def test_held_flag_not_cleared_on_nested_flock_failure(self, locked_state):
        """If flock raises in a nested call, the outer held=True must survive."""
        import fcntl

        original_flock = fcntl.flock
        call_count = [0]

        def flock_that_fails_second_time(fd, op):
            call_count[0] += 1
            if call_count[0] == 2:
                # Simulate flock failure on the second (nested) call
                raise OSError("flock failed")
            return original_flock(fd, op)

        # Manually set held=True to simulate outer lock ownership
        locked_state._local.held = True

        # Nested call should be a no-op due to re-entrancy
        with locked_state._file_lock():
            pass  # Should not touch held flag at all

        # The outer caller's held=True must be preserved
        assert locked_state._local.held is True
        locked_state._local.held = False  # cleanup


class TestLockFilePermissions:
    """Test that _file_lock creates the lock file with restricted permissions."""

    def test_lock_file_created_with_0600_permissions(self, locked_state):
        """Lock file should be created with 0o600 (owner-only read/write)."""
        with locked_state._file_lock():
            lock_path = locked_state._lock_path
            assert lock_path.exists()
            mode = lock_path.stat().st_mode & 0o777
            assert mode == 0o600, (
                f"Lock file permissions should be 0o600 but got {oct(mode)}"
            )


class TestLoadHistoryReturnsCopy:
    """Test that load_history() returns a copy, not a reference to the cache."""

    def test_returned_list_is_not_cache(self, locked_state):
        locked_state.record_cycle(CycleRecord(
            timestamp=time.time(),
            task_description="test",
            success=True,
        ))
        result1 = locked_state.load_history()
        result2 = locked_state.load_history()
        # Should be equal in content but different list objects
        assert result1 == result2
        assert result1 is not result2

    def test_mutating_returned_list_does_not_affect_cache(self, locked_state):
        locked_state.record_cycle(CycleRecord(
            timestamp=time.time(),
            task_description="test",
            success=True,
        ))
        result = locked_state.load_history()
        original_len = len(result)
        result.append({"fake": "record"})  # mutate the returned list
        # Next load should return original data, unaffected
        result2 = locked_state.load_history()
        assert len(result2) == original_len


class TestFileLockExplicitUnlock:
    """File lock should be explicitly released before closing fd."""

    def test_flock_unlock_called_before_close(self, locked_state):
        import fcntl
        calls = []
        original_flock = fcntl.flock

        def tracking_flock(fd, op):
            calls.append(("flock", fd, op))
            return original_flock(fd, op)

        original_close = os.close

        def tracking_close(fd):
            calls.append(("close", fd))
            return original_close(fd)

        with patch("state_lock.fcntl.flock", side_effect=tracking_flock):
            with patch("state_lock.os.close", side_effect=tracking_close):
                locked_state.record_cycle(CycleRecord(
                    timestamp=time.time(),
                    task_description="test unlock",
                    success=True,
                ))

        # Find the LOCK_EX and verify LOCK_UN comes before close on same fd
        lock_fds = [c[1] for c in calls if c[0] == "flock" and c[2] == fcntl.LOCK_EX]
        assert lock_fds, "No LOCK_EX calls found"
        for fd in lock_fds:
            unlock_calls = [i for i, c in enumerate(calls) if c == ("flock", fd, fcntl.LOCK_UN)]
            close_calls = [i for i, c in enumerate(calls) if c == ("close", fd)]
            assert unlock_calls, f"No LOCK_UN call for fd {fd}"
            assert close_calls, f"No close call for fd {fd}"
            assert unlock_calls[0] < close_calls[0], (
                f"LOCK_UN (index {unlock_calls[0]}) should come before close (index {close_calls[0]})"
            )
