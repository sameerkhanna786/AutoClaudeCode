"""Tests for validator module."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from config_schema import Config
from process_utils import RunResult
from validator import ValidationResult, Validator


@pytest.fixture
def validator(default_config):
    return Validator(default_config)


class TestValidator:
    @patch("validator.run_with_group_kill")
    def test_all_pass(self, mock_run, validator):
        mock_run.return_value = RunResult(returncode=0, stdout="OK", stderr="", timed_out=False)
        result = validator.validate("/tmp")
        assert result.passed is True

    @patch("validator.run_with_group_kill")
    def test_test_failure_short_circuits(self, mock_run, validator):
        mock_run.return_value = RunResult(returncode=1, stdout="FAILED", stderr="", timed_out=False)
        result = validator.validate("/tmp")
        assert result.passed is False
        # Should only have run tests, not lint or build (lint/build are empty by default)
        assert any(s.name == "tests" for s in result.steps)

    @patch("validator.run_with_group_kill")
    def test_timeout(self, mock_run, validator):
        mock_run.return_value = RunResult(returncode=-1, stdout="", stderr="", timed_out=True)
        result = validator.validate("/tmp")
        assert result.passed is False
        # Lint is skipped (empty command), tests timeout
        test_steps = [s for s in result.steps if s.name == "tests"]
        assert len(test_steps) == 1
        assert "Timed out" in test_steps[0].output

    def test_empty_commands_skip(self, default_config):
        # Default config has empty lint and build commands
        default_config.validation.test_command = ""
        v = Validator(default_config)
        with patch("validator.run_with_group_kill") as mock_run:
            result = v.validate("/tmp")
            assert result.passed is True
            mock_run.assert_not_called()

    def test_summary(self):
        result = ValidationResult(passed=True, steps=[])
        assert result.summary == "no validations run"

    @patch("validator.run_with_group_kill")
    def test_lint_failure_short_circuits_before_tests(self, mock_run, default_config):
        """Lint runs before tests; lint failure short-circuits so tests never run."""
        default_config.validation.lint_command = "ruff check ."
        v = Validator(default_config)

        def side_effect(cmd, **kwargs):
            if "ruff" in (cmd if isinstance(cmd, str) else " ".join(cmd)):
                return RunResult(returncode=1, stdout="lint error", stderr="", timed_out=False)
            return RunResult(returncode=0, stdout="passed", stderr="", timed_out=False)

        mock_run.side_effect = side_effect
        result = v.validate("/tmp")
        assert result.passed is False
        assert len(result.steps) == 1  # Only lint ran, tests never started
        assert result.steps[0].name == "lint"

    @patch("validator.run_with_group_kill")
    def test_unexpected_exception_in_run_command(self, mock_run, validator):
        """Unexpected exceptions from subprocess.run should fail validation, not propagate."""
        mock_run.side_effect = RuntimeError("unexpected failure")
        result = validator.validate("/tmp")
        assert result.passed is False
        # Lint is skipped (empty command), tests hit the exception
        test_steps = [s for s in result.steps if s.name == "tests"]
        assert len(test_steps) == 1
        assert test_steps[0].passed is False
        assert "Unexpected error" in test_steps[0].output
        assert "unexpected failure" in test_steps[0].output
        assert test_steps[0].return_code == -1

    @patch("validator.run_with_group_kill")
    def test_unexpected_exception_returns_validation_result(self, mock_run, validator):
        """Validate always returns a ValidationResult, even on unexpected errors."""
        mock_run.side_effect = MemoryError("out of memory")
        result = validator.validate("/tmp")
        assert isinstance(result, ValidationResult)
        assert result.passed is False
        # Lint is skipped (empty command), tests hit the exception
        test_steps = [s for s in result.steps if s.name == "tests"]
        assert len(test_steps) == 1
        assert "out of memory" in test_steps[0].output
