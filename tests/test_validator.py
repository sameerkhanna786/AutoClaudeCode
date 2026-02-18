"""Tests for validator module."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from config_schema import Config
from validator import ValidationResult, Validator


@pytest.fixture
def validator(default_config):
    return Validator(default_config)


class TestValidator:
    @patch("validator.subprocess.run")
    def test_all_pass(self, mock_run, validator):
        mock_run.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
        result = validator.validate("/tmp")
        assert result.passed is True

    @patch("validator.subprocess.run")
    def test_test_failure_short_circuits(self, mock_run, validator):
        mock_run.return_value = MagicMock(returncode=1, stdout="FAILED", stderr="")
        result = validator.validate("/tmp")
        assert result.passed is False
        # Should only have run tests, not lint or build (lint/build are empty by default)
        assert any(s.name == "tests" for s in result.steps)

    @patch("validator.subprocess.run")
    def test_timeout(self, mock_run, validator):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="pytest", timeout=120)
        result = validator.validate("/tmp")
        assert result.passed is False
        assert "Timed out" in result.steps[0].output

    def test_empty_commands_skip(self, default_config):
        # Default config has empty lint and build commands
        default_config.validation.test_command = ""
        v = Validator(default_config)
        with patch("validator.subprocess.run") as mock_run:
            result = v.validate("/tmp")
            assert result.passed is True
            mock_run.assert_not_called()

    def test_summary(self):
        result = ValidationResult(passed=True, steps=[])
        assert result.summary == "no validations run"

    @patch("validator.subprocess.run")
    def test_lint_failure_after_test_pass(self, mock_run, default_config):
        default_config.validation.lint_command = "ruff check ."
        v = Validator(default_config)

        def side_effect(cmd, **kwargs):
            if "pytest" in cmd:
                return MagicMock(returncode=0, stdout="passed", stderr="")
            else:
                return MagicMock(returncode=1, stdout="lint error", stderr="")

        mock_run.side_effect = side_effect
        result = v.validate("/tmp")
        assert result.passed is False
        assert len(result.steps) == 2  # tests passed, lint failed, build not run

    @patch("validator.subprocess.run")
    def test_unexpected_exception_in_run_command(self, mock_run, validator):
        """Unexpected exceptions from subprocess.run should fail validation, not propagate."""
        mock_run.side_effect = RuntimeError("unexpected failure")
        result = validator.validate("/tmp")
        assert result.passed is False
        assert result.steps[0].passed is False
        assert "Unexpected error" in result.steps[0].output
        assert "unexpected failure" in result.steps[0].output
        assert result.steps[0].return_code == -1

    @patch("validator.subprocess.run")
    def test_unexpected_exception_returns_validation_result(self, mock_run, validator):
        """Validate always returns a ValidationResult, even on unexpected errors."""
        mock_run.side_effect = MemoryError("out of memory")
        result = validator.validate("/tmp")
        assert isinstance(result, ValidationResult)
        assert result.passed is False
        assert len(result.steps) == 1
        assert "out of memory" in result.steps[0].output
