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
