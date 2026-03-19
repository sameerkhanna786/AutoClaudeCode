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
