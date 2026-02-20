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
