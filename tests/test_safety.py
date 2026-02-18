"""Tests for safety module."""

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config_schema import Config
from safety import SafetyError, SafetyGuard
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
        guard.release_lock()
        assert guard._lock_fd is None

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
        with pytest.raises(SafetyError, match="Protected files"):
            guard.check_protected_files(["src/main.py", "other.py"])

    def test_check_protected_files_deep_nested_path(self, guard):
        with pytest.raises(SafetyError, match="Protected files"):
            guard.check_protected_files(["a/b/c/config.yaml"])

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
