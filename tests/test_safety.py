"""Tests for safety module."""

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config_schema import Config
from safety import GracefulDegradation, SafetyError, SafetyGuard
from state import CycleRecord, StateManager


@pytest.fixture
def state_mgr(tmp_path, default_config):
    default_config.paths.history_file = str(tmp_path / "history.json")
    return StateManager(default_config)


@pytest.fixture
def guard(tmp_path, default_config, state_mgr):
    default_config.paths.lock_file = str(tmp_path / "lock.pid")
    return SafetyGuard(default_config, state_mgr)


class TestSafetyGuard:
    def test_acquire_and_release_lock(self, guard):
        guard.acquire_lock()
        assert guard._lock_fd is not None
        assert guard.lock_path.exists()
        guard.release_lock()
        assert guard._lock_fd is None
        # Lock file should persist after release (close releases flock)
        assert guard.lock_path.exists()

    def test_double_lock_fails(self, guard, tmp_path, default_config, state_mgr):
        guard.acquire_lock()
        guard2 = SafetyGuard(default_config, state_mgr)
        with pytest.raises(SafetyError, match="already running"):
            guard2.acquire_lock()
        guard.release_lock()

    def test_check_disk_space_ok(self, guard):
        # Should not raise on a normal system
        guard.check_disk_space()

    def test_check_disk_space_low(self, guard):
        guard.config.safety.min_disk_space_mb = 999_999_999
        with pytest.raises(SafetyError, match="disk space"):
            guard.check_disk_space()

    def test_check_rate_limit_ok(self, guard):
        guard.check_rate_limit()

    def test_check_rate_limit_exceeded(self, guard, state_mgr):
        guard.config.safety.max_cycles_per_hour = 2
        now = time.time()
        state_mgr.record_cycle(CycleRecord(timestamp=now, task_description="A"))
        state_mgr.record_cycle(CycleRecord(timestamp=now, task_description="B"))
        with pytest.raises(SafetyError, match="Rate limit"):
            guard.check_rate_limit()

    def test_check_cost_limit_ok(self, guard):
        guard.check_cost_limit()

    def test_check_cost_limit_exceeded(self, guard, state_mgr):
        guard.config.safety.max_cost_usd_per_hour = 0.10
        now = time.time()
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="A", cost_usd=0.15,
        ))
        with pytest.raises(SafetyError, match="Cost limit"):
            guard.check_cost_limit()

    def test_check_consecutive_failures_ok(self, guard):
        guard.check_consecutive_failures()

    def test_check_consecutive_failures_exceeded(self, guard, state_mgr):
        guard.config.safety.max_consecutive_failures = 2
        now = time.time()
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="A", success=False,
        ))
        state_mgr.record_cycle(CycleRecord(
            timestamp=now, task_description="B", success=False,
        ))
        with pytest.raises(SafetyError, match="consecutive failures"):
            guard.check_consecutive_failures()

    def test_check_protected_files(self, guard):
        guard.check_protected_files(["foo.py", "bar.py"])  # OK

    def test_check_protected_files_violation(self, guard):
        with pytest.raises(SafetyError, match="Protected files"):
            guard.check_protected_files(["main.py", "other.py"])

    def test_check_protected_files_nested_path(self, guard):
        guard.check_protected_files(["src/main.py", "other.py"])  # OK, different file

    def test_check_protected_files_deep_nested_path(self, guard):
        guard.check_protected_files(["a/b/c/config.yaml"])  # OK, different file

    def test_check_protected_files_relative_path(self, guard):
        with pytest.raises(SafetyError, match="Protected files"):
            guard.check_protected_files(["./main.py"])

    def test_check_protected_files_safe_nested(self, guard):
        guard.check_protected_files(["src/foo.py"])  # OK, no false positive

    def test_check_file_count_ok(self, guard):
        guard.check_file_count(["a.py", "b.py"])

    def test_check_file_count_exceeded(self, guard):
        guard.config.orchestrator.max_changed_files = 2
        with pytest.raises(SafetyError, match="Too many files"):
            guard.check_file_count(["a.py", "b.py", "c.py"])

    def test_pre_flight_checks(self, guard):
        guard.pre_flight_checks()  # Should not raise with defaults

    def test_post_claude_checks(self, guard):
        guard.post_claude_checks(["foo.py"])  # Should not raise

    def test_lock_reacquire_after_release(self, guard):
        """Lock can be re-acquired after release even though file persists."""
        guard.acquire_lock()
        guard.release_lock()
        # File persists, but lock should be re-acquirable
        guard.acquire_lock()
        assert guard._lock_fd is not None
        guard.release_lock()

    def test_check_protected_files_symlink(self, guard, tmp_path):
        """Symlinks to protected files should be detected."""
        guard.config.target_dir = str(tmp_path)
        (tmp_path / "main.py").write_text("# protected\n")
        (tmp_path / "link_to_main.py").symlink_to(tmp_path / "main.py")
        with pytest.raises(SafetyError, match="Protected files"):
            guard.check_protected_files(["link_to_main.py"])

    def test_check_protected_files_nonexistent_file(self, guard, tmp_path):
        """When a changed file doesn't exist on disk, normpath comparison still catches it."""
        guard.config.target_dir = str(tmp_path)
        # Neither file exists on disk, but normpath match should still detect it
        with pytest.raises(SafetyError, match="Protected files"):
            guard.check_protected_files(["./main.py"])

    def test_check_protected_files_toctou_race(self, guard, tmp_path):
        """check_protected_files handles file disappearing between check and samefile."""
        guard.config.target_dir = str(tmp_path)
        (tmp_path / "main.py").write_text("# protected\n")

        # Simulate samefile raising OSError (file deleted between calls)
        original_samefile = os.path.samefile

        def flaky_samefile(a, b):
            if "vanish.py" in str(a):
                raise FileNotFoundError("file vanished")
            return original_samefile(a, b)

        with patch("safety.os.path.samefile", side_effect=flaky_samefile):
            # vanish.py doesn't match any protected file via normpath either
            guard.check_protected_files(["vanish.py"])

    def test_check_protected_files_no_existence_precheck(self, guard, tmp_path):
        """check_protected_files no longer pre-checks existence before samefile."""
        guard.config.target_dir = str(tmp_path)
        (tmp_path / "main.py").write_text("# protected\n")
        (tmp_path / "changed.py").write_text("# changed\n")

        # Verify os.path.exists is NOT called (no TOCTOU)
        with patch("safety.os.path.exists") as mock_exists:
            guard.check_protected_files(["changed.py"])
        mock_exists.assert_not_called()

    def test_check_protected_files_samefile_false_skips_realpath(self, guard, tmp_path):
        """When samefile returns False, realpath fallback should be skipped."""
        guard.config.target_dir = str(tmp_path)
        # Create the changed file and all protected files so samefile can be used
        (tmp_path / "changed.py").write_text("# changed\n")
        for p in guard.config.safety.protected_files:
            (tmp_path / p).write_text("# protected\n")
        # Track whether realpath is called
        original_realpath = os.path.realpath
        realpath_calls = []

        def tracking_realpath(p):
            realpath_calls.append(p)
            return original_realpath(p)

        with patch("safety.os.path.realpath", side_effect=tracking_realpath):
            guard.check_protected_files(["changed.py"])
        # samefile returned False for all protected files, so realpath should not be called
        assert len(realpath_calls) == 0


class TestFailureRecoveryGuard:
    def test_file_based_reset_trigger(self, guard, state_mgr, tmp_path):
        """The state/reset_failures file clears lockout and gets deleted."""
        guard.config.safety.max_consecutive_failures = 3
        guard.config.paths.state_dir = str(tmp_path)
        now = time.time()
        for i in range(3):
            state_mgr.record_cycle(CycleRecord(
                timestamp=now + i,
                task_description=f"Fail {i}",
                success=False,
            ))
        # Without reset file, should raise
        with pytest.raises(SafetyError, match="consecutive failures"):
            guard.check_consecutive_failures()

        # Create reset file
        reset_file = tmp_path / "reset_failures"
        reset_file.write_text("")

        # Now should NOT raise, and file should be deleted
        guard.check_consecutive_failures()
        assert not reset_file.exists()
        assert state_mgr.get_consecutive_failures() == 0

    def test_time_based_auto_reset(self, guard, state_mgr):
        """Auto-reset triggers when idle for 1+ hour."""
        guard.config.safety.max_consecutive_failures = 3
        old_time = time.time() - 7200  # 2 hours ago
        for i in range(3):
            state_mgr.record_cycle(CycleRecord(
                timestamp=old_time + i,
                task_description=f"Fail {i}",
                success=False,
            ))
        # Should NOT raise because idle for > 1 hour
        guard.check_consecutive_failures()
        assert state_mgr.get_consecutive_failures() == 0

    def test_no_auto_reset_when_recent(self, guard, state_mgr):
        """No auto-reset when failures are recent."""
        guard.config.safety.max_consecutive_failures = 3
        now = time.time()
        for i in range(3):
            state_mgr.record_cycle(CycleRecord(
                timestamp=now - 10 + i,  # Very recent
                task_description=f"Fail {i}",
                success=False,
            ))
        with pytest.raises(SafetyError, match="consecutive failures"):
            guard.check_consecutive_failures()

    def test_error_message_includes_reset_instructions(self, guard, state_mgr):
        """Error message tells user how to reset."""
        guard.config.safety.max_consecutive_failures = 2
        now = time.time()
        for i in range(2):
            state_mgr.record_cycle(CycleRecord(
                timestamp=now - 10 + i,
                task_description=f"Fail {i}",
                success=False,
            ))
        with pytest.raises(SafetyError, match="touch state/reset_failures"):
            guard.check_consecutive_failures()


class TestMemoryCheck:
    def test_check_memory_ok_on_current_system(self, guard):
        """Memory check should not raise on a system with sufficient RAM."""
        guard.config.safety.min_memory_mb = 1  # 1 MB — any system has this
        guard.check_memory()

    def test_check_memory_disabled_when_zero(self, guard):
        """Memory check should be skipped when min_memory_mb is 0."""
        guard.config.safety.min_memory_mb = 0
        guard.check_memory()  # Should not raise

    @patch("platform.system", return_value="Linux")
    def test_check_memory_low_on_linux(self, mock_system, guard):
        """Memory check raises SafetyError when available RAM is low (Linux)."""
        guard.config.safety.min_memory_mb = 999_999
        meminfo = "MemTotal:       16384000 kB\nMemAvailable:       1024 kB\n"
        with patch("builtins.open", create=True) as mock_open:
            from unittest.mock import mock_open as _mock_open
            mock_open.return_value = _mock_open(read_data=meminfo)()
            with pytest.raises(SafetyError, match="Low memory"):
                guard.check_memory()

    @patch("platform.system", return_value="Linux")
    def test_check_memory_sufficient_on_linux(self, mock_system, guard):
        """Memory check passes when available RAM exceeds minimum (Linux)."""
        guard.config.safety.min_memory_mb = 100
        meminfo = "MemTotal:       16384000 kB\nMemAvailable:       512000 kB\n"
        with patch("builtins.open", create=True) as mock_open:
            from unittest.mock import mock_open as _mock_open
            mock_open.return_value = _mock_open(read_data=meminfo)()
            guard.check_memory()

    @patch("platform.system", return_value="Linux")
    def test_check_memory_proc_unreadable(self, mock_system, guard):
        """Memory check is skipped gracefully when /proc/meminfo is unreadable."""
        guard.config.safety.min_memory_mb = 256
        with patch("builtins.open", side_effect=OSError("Permission denied")):
            guard.check_memory()  # Should not raise

    @patch("platform.system", return_value="Windows")
    def test_check_memory_unsupported_platform(self, mock_system, guard):
        """Memory check is skipped on unsupported platforms."""
        guard.config.safety.min_memory_mb = 256
        guard.check_memory()  # Should not raise

    @patch("platform.system", return_value="Linux")
    def test_check_memory_warning_threshold(self, mock_system, guard, caplog):
        """Memory check logs warning when approaching minimum."""
        import logging
        guard.config.safety.min_memory_mb = 300
        # 350 MB is between min (300) and 1.5x min (450), should warn
        meminfo = "MemAvailable:       358400 kB\n"  # ~350 MB
        with patch("builtins.open", create=True) as mock_open:
            from unittest.mock import mock_open as _mock_open
            mock_open.return_value = _mock_open(read_data=meminfo)()
            with caplog.at_level(logging.WARNING):
                guard.check_memory()
        assert any("approaching minimum" in r.message for r in caplog.records)

    def test_pre_flight_checks_includes_memory(self, guard):
        """pre_flight_checks should call check_memory."""
        guard.config.safety.min_memory_mb = 1
        with patch.object(guard, 'check_memory') as mock_mem:
            with patch.object(guard, 'check_disk_space'):
                with patch.object(guard, 'check_rate_limit'):
                    with patch.object(guard, 'check_cost_limit'):
                        with patch.object(guard, 'check_consecutive_failures'):
                            guard.pre_flight_checks()
            mock_mem.assert_called_once()

    def test_pre_flight_checks_collects_all_failures(self, guard):
        """pre_flight_checks should collect all failures into a single error."""
        with patch.object(guard, 'check_disk_space', side_effect=SafetyError("Low disk")):
            with patch.object(guard, 'check_memory', side_effect=SafetyError("Low memory")):
                with patch.object(guard, 'check_rate_limit'):
                    with patch.object(guard, 'check_cost_limit'):
                        with patch.object(guard, 'check_consecutive_failures'):
                            with pytest.raises(SafetyError, match="2 pre-flight check.*failed"):
                                guard.pre_flight_checks()

    def test_pre_flight_checks_combined_error_contains_all_messages(self, guard):
        """The combined error message should contain each individual failure."""
        with patch.object(guard, 'check_disk_space', side_effect=SafetyError("Low disk space")):
            with patch.object(guard, 'check_memory'):
                with patch.object(guard, 'check_rate_limit', side_effect=SafetyError("Rate limit reached")):
                    with patch.object(guard, 'check_cost_limit', side_effect=SafetyError("Cost limit reached")):
                        with patch.object(guard, 'check_consecutive_failures'):
                            with pytest.raises(SafetyError) as exc_info:
                                guard.pre_flight_checks()
                            msg = str(exc_info.value)
                            assert "Low disk space" in msg
                            assert "Rate limit reached" in msg
                            assert "Cost limit reached" in msg
                            assert "3 pre-flight check(s) failed" in msg

    def test_pre_flight_checks_single_failure_still_reports_count(self, guard):
        """Even a single failure should use the collected error format."""
        with patch.object(guard, 'check_disk_space'):
            with patch.object(guard, 'check_memory'):
                with patch.object(guard, 'check_rate_limit'):
                    with patch.object(guard, 'check_cost_limit', side_effect=SafetyError("Cost limit")):
                        with patch.object(guard, 'check_consecutive_failures'):
                            with pytest.raises(SafetyError, match="1 pre-flight check.*failed.*Cost limit"):
                                guard.pre_flight_checks()


class TestGracefulDegradation:
    @pytest.fixture
    def degradation(self, default_config):
        default_config.safety.max_cycles_per_hour = 100
        default_config.safety.max_cost_usd_per_hour = 100.0
        return GracefulDegradation(default_config)

    def test_normal_level_below_70(self, degradation):
        """At 69% usage, should return normal (level 0)."""
        result = degradation.check_and_adjust(69, 0.0)
        assert result["degraded"] is False
        assert result["level"] == 0
        assert result["batch_size_factor"] == 1.0
        assert result["sleep_multiplier"] == 1.0
        assert result["reason"] == ""
        assert degradation.is_degraded is False
        assert degradation.degradation_level == 0

    def test_mild_level_at_70(self, degradation):
        """At exactly 70% usage, should return mild (level 1)."""
        result = degradation.check_and_adjust(70, 0.0)
        assert result["degraded"] is True
        assert result["level"] == 1
        assert result["batch_size_factor"] == 0.75
        assert result["sleep_multiplier"] == 2.0
        assert "Mild" in result["reason"]
        assert degradation.is_degraded is True
        assert degradation.degradation_level == 1

    def test_moderate_level_at_85(self, degradation):
        """At exactly 85% usage, should return moderate (level 2)."""
        result = degradation.check_and_adjust(85, 0.0)
        assert result["degraded"] is True
        assert result["level"] == 2
        assert result["batch_size_factor"] == 0.5
        assert result["sleep_multiplier"] == 3.0
        assert "Moderate" in result["reason"]
        assert degradation.is_degraded is True
        assert degradation.degradation_level == 2

    def test_severe_level_at_95(self, degradation):
        """At exactly 95% usage, should return severe (level 3)."""
        result = degradation.check_and_adjust(95, 0.0)
        assert result["degraded"] is True
        assert result["level"] == 3
        assert result["batch_size_factor"] == 0.25
        assert result["sleep_multiplier"] == 4.0
        assert "Severe" in result["reason"]
        assert degradation.is_degraded is True
        assert degradation.degradation_level == 3

    def test_boundary_69_is_normal(self, degradation):
        """69% is just below the mild threshold."""
        result = degradation.check_and_adjust(69, 69.0)
        assert result["level"] == 0
        assert result["degraded"] is False

    def test_boundary_84_is_mild(self, degradation):
        """84% is between mild and moderate thresholds."""
        result = degradation.check_and_adjust(84, 0.0)
        assert result["level"] == 1

    def test_boundary_94_is_moderate(self, degradation):
        """94% is between moderate and severe thresholds."""
        result = degradation.check_and_adjust(94, 0.0)
        assert result["level"] == 2

    def test_cost_driven_degradation(self, degradation):
        """Cost percentage drives degradation when higher than rate."""
        result = degradation.check_and_adjust(0, 95.0)
        assert result["level"] == 3
        assert "cost" in result["reason"]

    def test_max_of_rate_and_cost(self, degradation):
        """The higher of rate_pct and cost_pct determines the level."""
        result = degradation.check_and_adjust(70, 95.0)
        assert result["level"] == 3  # cost at 95% dominates

    def test_reason_includes_both_when_both_above_70(self, degradation):
        """When both rate and cost are >= 70%, reason mentions both."""
        result = degradation.check_and_adjust(80, 90.0)
        assert "rate" in result["reason"]
        assert "cost" in result["reason"]

    def test_recovery_from_degraded_to_normal(self, degradation):
        """Degradation level returns to normal when usage drops."""
        degradation.check_and_adjust(95, 0.0)
        assert degradation.is_degraded is True
        result = degradation.check_and_adjust(50, 0.0)
        assert result["level"] == 0
        assert degradation.is_degraded is False


class TestActiveGuardsThreadSafety:
    """Tests for thread-safe _active_guards list management."""

    def test_active_guards_protected_by_lock(self):
        """Verify _active_guards_lock exists and is a threading.Lock."""
        from safety import _active_guards_lock
        import threading
        assert isinstance(_active_guards_lock, type(threading.Lock()))

    def test_acquire_release_updates_active_guards(self, guard):
        """Guard appears in _active_guards after acquire and is removed after release."""
        from safety import _active_guards
        guard.acquire_lock()
        assert guard in _active_guards
        guard.release_lock()
        assert guard not in _active_guards

    def test_concurrent_acquire_release(self, tmp_path, default_config, state_mgr):
        """Multiple guards from different threads don't corrupt _active_guards."""
        import threading
        from safety import _active_guards

        guards = []
        errors = []

        def worker(i):
            try:
                cfg = default_config
                cfg.paths.lock_file = str(tmp_path / f"lock_{i}.pid")
                g = SafetyGuard(cfg, state_mgr)
                g.acquire_lock()
                guards.append(g)
                import time
                time.sleep(0.01)
                g.release_lock()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors in threads: {errors}"
        # All guards should have been removed after release
        for g in guards:
            assert g not in _active_guards


class TestDiskSpaceOSError:
    """Tests for OSError handling in check_disk_space."""

    def test_oserror_raises_safety_error(self, guard):
        """OSError from shutil.disk_usage should become SafetyError."""
        with patch("safety.shutil.disk_usage", side_effect=OSError("mount not found")):
            with pytest.raises(SafetyError, match="Cannot check disk space"):
                guard.check_disk_space()

    def test_normal_disk_check_passes(self, guard):
        """Normal disk check with enough space should not raise."""
        guard.check_disk_space()  # should not raise on real filesystem


class TestMeasureDirSize:
    """Tests for SafetyGuard._measure_dir_size helper."""

    def test_measures_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.txt").write_text("world!")
        size = SafetyGuard._measure_dir_size(tmp_path)
        assert size == 5 + 6  # len("hello") + len("world!")

    def test_empty_dir_returns_zero(self, tmp_path):
        size = SafetyGuard._measure_dir_size(tmp_path)
        assert size == 0

    def test_nonexistent_dir_returns_zero(self, tmp_path):
        # os.walk silently yields nothing for nonexistent dirs
        size = SafetyGuard._measure_dir_size(tmp_path / "nonexistent")
        assert size == 0

    def test_nested_dirs(self, tmp_path):
        sub = tmp_path / "sub" / "deep"
        sub.mkdir(parents=True)
        (sub / "file.txt").write_text("abc")
        (tmp_path / "top.txt").write_text("de")
        size = SafetyGuard._measure_dir_size(tmp_path)
        assert size == 3 + 2
