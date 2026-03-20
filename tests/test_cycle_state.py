"""Tests for cycle_state module."""

import json
import os
from pathlib import Path

import pytest

from cycle_state import CycleState, CycleStateWriter, read_cycle_state


class TestCycleStateWriter:
    def test_write_and_read(self, tmp_path):
        writer = CycleStateWriter(str(tmp_path))
        state = CycleState(
            phase="executing",
            task_description="Fix bug",
            task_type="test_failure",
            started_at=1000.0,
            batch_size=2,
        )
        writer.write(state)

        result = read_cycle_state(str(tmp_path))
        assert result is not None
        assert result.phase == "executing"
        assert result.task_description == "Fix bug"
        assert result.task_type == "test_failure"
        assert result.started_at == 1000.0
        assert result.batch_size == 2

    def test_clear_removes_file(self, tmp_path):
        writer = CycleStateWriter(str(tmp_path))
        state = CycleState(phase="executing")
        writer.write(state)
        assert (tmp_path / "current_cycle.json").exists()

        writer.clear()
        assert not (tmp_path / "current_cycle.json").exists()

    def test_clear_no_file_is_noop(self, tmp_path):
        writer = CycleStateWriter(str(tmp_path))
        writer.clear()  # should not raise

    def test_update_merges_fields(self, tmp_path):
        writer = CycleStateWriter(str(tmp_path))
        state = CycleState(
            phase="planning",
            task_description="Fix bug",
            accumulated_cost=0.01,
        )
        writer.write(state)

        writer.update(phase="executing", accumulated_cost=0.05)

        result = read_cycle_state(str(tmp_path))
        assert result is not None
        assert result.phase == "executing"
        assert result.task_description == "Fix bug"  # unchanged
        assert result.accumulated_cost == 0.05

    def test_update_creates_state_if_none(self, tmp_path):
        writer = CycleStateWriter(str(tmp_path))
        writer.update(phase="validating", retry_count=2)

        result = read_cycle_state(str(tmp_path))
        assert result is not None
        assert result.phase == "validating"
        assert result.retry_count == 2

    def test_atomic_write(self, tmp_path):
        """Write should be atomic — no partial files."""
        writer = CycleStateWriter(str(tmp_path))
        state = CycleState(phase="executing", task_description="Test atomicity")
        writer.write(state)

        # Read the raw file and verify it's valid JSON
        content = (tmp_path / "current_cycle.json").read_text()
        data = json.loads(content)
        assert data["phase"] == "executing"

    def test_creates_state_dir(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "dir"
        writer = CycleStateWriter(str(nested))
        state = CycleState(phase="test")
        writer.write(state)
        assert (nested / "current_cycle.json").exists()

    def test_path_property(self, tmp_path):
        writer = CycleStateWriter(str(tmp_path))
        assert writer.path == str(tmp_path / "current_cycle.json")


class TestReadCycleState:
    def test_no_file_returns_none(self, tmp_path):
        result = read_cycle_state(str(tmp_path))
        assert result is None

    def test_empty_file_returns_none(self, tmp_path):
        (tmp_path / "current_cycle.json").write_text("")
        result = read_cycle_state(str(tmp_path))
        assert result is None

    def test_invalid_json_returns_none(self, tmp_path):
        (tmp_path / "current_cycle.json").write_text("{broken json")
        result = read_cycle_state(str(tmp_path))
        assert result is None

    def test_all_fields(self, tmp_path):
        data = {
            "phase": "retrying",
            "task_description": "Fix lint",
            "task_type": "lint",
            "task_descriptions": ["Fix lint", "Fix todo"],
            "started_at": 1234.5,
            "pipeline_agent": "coder",
            "pipeline_revision": 2,
            "accumulated_cost": 0.15,
            "batch_size": 3,
            "retry_count": 1,
        }
        (tmp_path / "current_cycle.json").write_text(json.dumps(data))
        result = read_cycle_state(str(tmp_path))
        assert result is not None
        assert result.phase == "retrying"
        assert result.pipeline_agent == "coder"
        assert result.pipeline_revision == 2
        assert result.accumulated_cost == 0.15
        assert result.batch_size == 3
        assert result.retry_count == 1
        assert result.task_descriptions == ["Fix lint", "Fix todo"]

    def test_missing_fields_get_defaults(self, tmp_path):
        (tmp_path / "current_cycle.json").write_text('{"phase": "test"}')
        result = read_cycle_state(str(tmp_path))
        assert result is not None
        assert result.phase == "test"
        assert result.task_description == ""
        assert result.accumulated_cost == 0.0
        assert result.batch_size == 1

    def test_invalid_json_logs_warning(self, tmp_path, caplog):
        """Invalid JSON should log a warning message."""
        import logging
        (tmp_path / "current_cycle.json").write_text("{broken json")
        with caplog.at_level(logging.WARNING):
            result = read_cycle_state(str(tmp_path))
        assert result is None
        # Verify a warning was logged
        assert len(caplog.records) > 0
        assert any("current_cycle.json" in r.message.lower() or "json" in r.message.lower()
                   for r in caplog.records)


class TestCycleStateEncoding:
    def test_unicode_round_trip(self, tmp_path):
        """Cycle state with non-ASCII chars should round-trip through write+read."""
        writer = CycleStateWriter(str(tmp_path))
        state = CycleState(
            phase="executing",
            task_description="Fix résumé handling in café.py",
        )
        writer.write(state)
        loaded = read_cycle_state(str(tmp_path))
        assert loaded is not None
        assert "résumé" in loaded.task_description
        assert "café" in loaded.task_description


class TestCycleStateWriterThreadSafety:
    """Tests that write() acquires the lock to prevent races with update()."""

    def test_write_acquires_lock(self, tmp_path):
        """write() should hold the lock during the write operation."""
        writer = CycleStateWriter(str(tmp_path))
        state = CycleState(phase="executing")

        # Manually acquire the lock; write() should block
        writer._lock.acquire()
        import threading
        result_holder = []

        def do_write():
            writer.write(state)
            result_holder.append("done")

        t = threading.Thread(target=do_write)
        t.start()
        # Give thread a moment to attempt the lock
        t.join(timeout=0.1)
        # Thread should still be blocked (no result yet)
        assert result_holder == []

        # Release lock; write should now complete
        writer._lock.release()
        t.join(timeout=2.0)
        assert result_holder == ["done"]

        # Verify the state was written correctly
        result = read_cycle_state(str(tmp_path))
        assert result is not None
        assert result.phase == "executing"

    def test_concurrent_write_and_update(self, tmp_path):
        """Concurrent write() and update() should not corrupt state."""
        import threading

        writer = CycleStateWriter(str(tmp_path))
        writer.write(CycleState(phase="initial", retry_count=0))
        errors = []

        def do_writes():
            try:
                for i in range(20):
                    writer.write(CycleState(phase=f"write_{i}", retry_count=i))
            except Exception as e:
                errors.append(e)

        def do_updates():
            try:
                for i in range(20):
                    writer.update(accumulated_cost=float(i))
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=do_writes)
        t2 = threading.Thread(target=do_updates)
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        assert errors == []
        # Final state should be valid JSON
        result = read_cycle_state(str(tmp_path))
        assert result is not None


class TestWriteUnlockedExceptionCleanup:
    """Temp files must be cleaned up even for non-OSError exceptions."""

    def test_type_error_cleans_up_temp_file(self, tmp_path):
        """A TypeError from json.dump should not leave orphaned temp files."""
        writer = CycleStateWriter(str(tmp_path))
        state = CycleState(phase="test")
        # Patch json.dump to raise TypeError (simulates non-serializable data)
        import json as json_mod
        original_dump = json_mod.dump

        def bad_dump(*args, **kwargs):
            raise TypeError("Object not serializable")

        import unittest.mock
        with unittest.mock.patch("cycle_state.json.dump", side_effect=bad_dump):
            writer.write(state)  # Should not raise, just log warning

        # No orphaned .tmp files should remain
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], f"Orphaned temp files found: {tmp_files}"

    def test_fdopen_uses_utf8_encoding(self, tmp_path):
        """os.fdopen calls should specify encoding='utf-8'."""
        import inspect
        from cycle_state import CycleStateWriter
        source = inspect.getsource(CycleStateWriter._write_unlocked)
        assert 'encoding="utf-8"' in source or "encoding='utf-8'" in source


class TestCycleStateClearThreadSafety:
    """clear() must hold the lock to prevent races with concurrent update() calls."""

    def test_clear_holds_lock(self, tmp_path):
        """clear() should acquire self._lock before setting self._current = None."""
        import threading
        writer = CycleStateWriter(str(tmp_path))
        writer.write(CycleState(phase="executing"))

        # Manually hold the lock; clear() should block
        writer._lock.acquire()
        result_holder = []

        def do_clear():
            writer.clear()
            result_holder.append("done")

        t = threading.Thread(target=do_clear)
        t.start()
        t.join(timeout=0.1)
        # Thread should be blocked (lock held by us)
        assert result_holder == [], "clear() should block when lock is held"

        writer._lock.release()
        t.join(timeout=2.0)
        assert result_holder == ["done"]

    def test_concurrent_clear_and_update(self, tmp_path):
        """Concurrent clear() and update() should not corrupt state."""
        import threading
        writer = CycleStateWriter(str(tmp_path))
        writer.write(CycleState(phase="initial"))
        errors = []

        def do_clears():
            try:
                for _ in range(20):
                    writer.clear()
            except Exception as e:
                errors.append(e)

        def do_updates():
            try:
                for i in range(20):
                    writer.update(phase=f"update_{i}")
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=do_clears)
        t2 = threading.Thread(target=do_updates)
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)
        assert errors == []


class TestCycleStateUpdateSkipsDiskRead:
    """update() should use in-memory state instead of reading from disk."""

    def test_update_uses_cached_state(self, tmp_path):
        """After write(), subsequent update() should not read from disk."""
        writer = CycleStateWriter(str(tmp_path))
        state = CycleState(phase="executing", task_description="Test")
        writer.write(state)

        # Delete the file on disk to prove update uses in-memory state
        cycle_path = tmp_path / "current_cycle.json"
        cycle_path.unlink()

        # update() should still work using cached state
        writer.update(phase="validating")
        # File should be re-created from in-memory state
        assert cycle_path.exists()
        result = read_cycle_state(str(tmp_path))
        assert result.phase == "validating"
        assert result.task_description == "Test"

    def test_update_without_prior_write_creates_default(self, tmp_path):
        """update() with no prior write should create a default state."""
        writer = CycleStateWriter(str(tmp_path))
        writer.update(phase="starting")
        result = read_cycle_state(str(tmp_path))
        assert result is not None
        assert result.phase == "starting"
