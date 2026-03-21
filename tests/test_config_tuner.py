"""Tests for config_tuner.py — atomic write and recommendation analysis."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from config_tuner import ConfigTuner, TuningRecommendation


class TestSaveRecommendations:
    def test_saves_json_atomically(self, tmp_path):
        """save_recommendations should produce a valid JSON file."""
        tuner = ConfigTuner(str(tmp_path))
        recs = [
            TuningRecommendation(
                field="claude.max_turns",
                current_value=25,
                recommended_value=20,
                reason="test reason",
                confidence=0.5,
            )
        ]
        tuner.save_recommendations(recs)

        out_file = tmp_path / "tuning_recommendations.json"
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert "timestamp" in data
        assert len(data["recommendations"]) == 1
        assert data["recommendations"][0]["field"] == "claude.max_turns"

    def test_no_partial_file_on_crash(self, tmp_path):
        """If writing fails, the original file should remain intact."""
        tuner = ConfigTuner(str(tmp_path))
        recs = [
            TuningRecommendation(
                field="test", current_value=1,
                recommended_value=2, reason="r", confidence=0.5,
            )
        ]
        # Write initial valid file
        tuner.save_recommendations(recs)
        out_file = tmp_path / "tuning_recommendations.json"
        original_content = out_file.read_text()

        # Make the directory read-only to simulate a write failure
        os.chmod(str(tmp_path), 0o444)
        try:
            recs2 = [
                TuningRecommendation(
                    field="broken", current_value=99,
                    recommended_value=100, reason="x", confidence=1.0,
                )
            ]
            tuner.save_recommendations(recs2)
        finally:
            os.chmod(str(tmp_path), 0o755)

        # After restoring permissions, original file should still be intact
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert data["recommendations"][0]["field"] == "test"

    def test_empty_recs_skipped(self, tmp_path):
        """save_recommendations with empty list should be a no-op."""
        tuner = ConfigTuner(str(tmp_path))
        tuner.save_recommendations([])
        out_file = tmp_path / "tuning_recommendations.json"
        assert not out_file.exists()

    def test_no_temp_files_left_on_success(self, tmp_path):
        """No .tmp files should remain after a successful save."""
        tuner = ConfigTuner(str(tmp_path))
        recs = [
            TuningRecommendation(
                field="test", current_value=1,
                recommended_value=2, reason="r", confidence=0.5,
            )
        ]
        tuner.save_recommendations(recs)
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0


class TestAnalyzeLoopInterval:
    """Tests for _analyze_loop_interval no-task detection heuristic."""

    def _make_config(self):
        from unittest.mock import MagicMock
        config = MagicMock()
        config.orchestrator.loop_interval_seconds = 60
        config.orchestrator.max_batch_size = 5
        config.orchestrator.max_validation_retries = 3
        config.claude.max_turns = 25
        return config

    def _make_records(self, error_messages):
        """Create recent records with given error messages (all failures)."""
        import time
        return [
            {"timestamp": time.time(), "success": False, "error": msg}
            for msg in error_messages
        ]

    def test_counts_no_tasks_error(self):
        """'No tasks found' should count as a no-task cycle."""
        config = self._make_config()
        tuner = ConfigTuner("/tmp/test")
        records = self._make_records(["No tasks found"] * 12 + ["ok"] * 8)
        # Mark some as successes
        for r in records[12:]:
            r["success"] = True
            r["error"] = ""
        recs = tuner._analyze_loop_interval(records, config)
        assert len(recs) == 1
        assert "interval" in recs[0].field.lower() or "loop" in recs[0].field.lower()

    def test_counts_no_actionable_tasks_error(self):
        """'No actionable tasks' should count as a no-task cycle."""
        config = self._make_config()
        tuner = ConfigTuner("/tmp/test")
        records = self._make_records(["No actionable tasks"] * 15 + ["ok"] * 5)
        for r in records[15:]:
            r["success"] = True
            r["error"] = ""
        recs = tuner._analyze_loop_interval(records, config)
        assert len(recs) == 1

    def test_does_not_match_false_positive_errors(self):
        """Errors like 'Cannot find task node' should NOT count as no-task cycles."""
        config = self._make_config()
        tuner = ConfigTuner("/tmp/test")
        records = self._make_records([
            "Cannot find task node",
            "Token not valid for task",
            "Validation failed: task dependency missing",
            "No response from task runner",
        ] * 5)
        recs = tuner._analyze_loop_interval(records, config)
        # These errors should NOT trigger the "no tasks" recommendation
        assert all("interval" not in r.field.lower() for r in recs)


class TestSaveRecommendationsExceptionCleanup:
    """Temp files must be cleaned up for any exception, not just OSError."""

    def test_non_os_error_cleans_up_temp_file(self, tmp_path):
        """A TypeError during JSON serialization should not leave temp files."""
        tuner = ConfigTuner(str(tmp_path))
        recs = [
            TuningRecommendation(
                field="test.field",
                current_value=1,
                recommended_value=2,
                reason="test",
                confidence=0.9,
            ),
        ]
        import unittest.mock
        with unittest.mock.patch("config_tuner.json.dump", side_effect=TypeError("bad")):
            tuner.save_recommendations(recs)  # Should not raise

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], f"Orphaned temp files: {tmp_files}"

    def test_fdopen_uses_utf8_encoding(self, tmp_path):
        """os.fdopen calls should specify encoding='utf-8'."""
        import inspect
        source = inspect.getsource(ConfigTuner.save_recommendations)
        assert 'encoding="utf-8"' in source or "encoding='utf-8'" in source


class TestAnalyzeLoopIntervalNoneError:
    """Regression: _analyze_loop_interval crashes when error is None in JSON."""

    def _make_config(self):
        from unittest.mock import MagicMock
        config = MagicMock()
        config.orchestrator.loop_interval_seconds = 60
        return config

    def test_none_error_field_does_not_crash(self):
        """Records with error=None (from JSON null) must not cause TypeError.

        dict.get("error", "") returns None when the key exists with value None,
        and re.search(pattern, None) raises TypeError.
        """
        import time
        config = self._make_config()
        tuner = ConfigTuner("/tmp/test")
        records = [
            {"timestamp": time.time(), "success": False, "error": None}
            for _ in range(20)
        ]
        # Should not raise TypeError
        tuner._analyze_loop_interval(records, config)


class TestRetryAnalysisNoDeadCode:
    """Ensure _analyze_retries has no unused computed variables."""

    def test_no_unused_avg_retries_variable(self):
        """avg_retries was computed but never used — it should be removed."""
        import inspect
        source = inspect.getsource(ConfigTuner._analyze_retries)
        assert "avg_retries" not in source, (
            "avg_retries was dead code and should have been removed"
        )


class TestAnalyzeLoopIntervalNoTaskSource:
    """The orchestrator returns early (no CycleRecord) when no tasks are found.

    This means _analyze_loop_interval's no-task detection via error field
    matching can never fire from real orchestrator data — the detection
    should use a dedicated 'no_tasks' field instead of parsing error strings.
    """

    def _make_config(self):
        from unittest.mock import MagicMock
        config = MagicMock()
        config.orchestrator.loop_interval_seconds = 60
        return config

    def test_no_task_detection_uses_dedicated_field(self):
        """_analyze_loop_interval should detect no-task cycles via 'no_tasks'
        field, not by parsing the error string."""
        import time
        config = self._make_config()
        tuner = ConfigTuner("/tmp/test")

        # Simulate records with the dedicated no_tasks=True field
        records = [
            {"timestamp": time.time(), "success": True, "error": "",
             "no_tasks": True}
            for _ in range(15)
        ] + [
            {"timestamp": time.time(), "success": True, "error": ""}
            for _ in range(5)
        ]

        recs = tuner._analyze_loop_interval(records, config)
        # Should produce a recommendation since 15/20 cycles had no tasks
        assert len(recs) == 1
        assert "interval" in recs[0].field.lower() or "loop" in recs[0].field.lower()

    def test_no_task_count_zero_without_field_or_error(self):
        """Records without 'no_tasks' field and without 'no tasks' in error
        should not count as no-task cycles."""
        import time
        config = self._make_config()
        tuner = ConfigTuner("/tmp/test")

        # Normal successful records — no no_tasks field, no error string
        records = [
            {"timestamp": time.time(), "success": True, "error": ""}
            for _ in range(20)
        ]

        recs = tuner._analyze_loop_interval(records, config)
        # No recommendation to increase interval
        no_task_recs = [r for r in recs if "interval" in r.field.lower()]
        # Should recommend decreasing if current > 60, but our config is 60
        assert all("increas" not in r.reason.lower() for r in no_task_recs)
