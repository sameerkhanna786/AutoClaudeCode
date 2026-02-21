"""Tests for validator module."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from config_schema import Config
from process_utils import RunResult
from validator import ValidationResult, ValidationStep, Validator


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

    @patch("validator.run_with_group_kill")
    def test_os_error_in_run_command(self, mock_run, validator):
        """OSError from subprocess should fail the step gracefully."""
        mock_run.side_effect = OSError("No such file or directory")
        result = validator.validate("/tmp")
        assert result.passed is False
        test_steps = [s for s in result.steps if s.name == "tests"]
        assert len(test_steps) == 1
        assert test_steps[0].passed is False
        assert "No such file or directory" in test_steps[0].output
        assert test_steps[0].return_code == -1

    def test_summary_with_mixed_results(self):
        """Summary should show PASS/FAIL for each step."""
        steps = [
            ValidationStep(name="lint", command="ruff check .", passed=True),
            ValidationStep(name="tests", command="pytest", passed=False, return_code=1),
        ]
        result = ValidationResult(passed=False, steps=steps)
        assert "lint: PASS" in result.summary
        assert "tests: FAIL" in result.summary

    @patch("validator.run_with_group_kill")
    def test_validate_uses_config_target_dir_when_no_working_dir(self, mock_run, default_config):
        """When working_dir is None, validate uses config.target_dir."""
        default_config.target_dir = "/my/project"
        default_config.validation.test_command = "pytest"
        default_config.validation.lint_command = ""
        default_config.validation.build_command = ""
        v = Validator(default_config)
        mock_run.return_value = RunResult(returncode=0, stdout="OK", stderr="", timed_out=False)
        v.validate()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["cwd"] == "/my/project"

    @patch("validator.run_with_group_kill")
    def test_build_failure_after_tests_pass(self, mock_run, default_config):
        """Build failure after tests pass should still fail overall."""
        default_config.validation.lint_command = ""
        default_config.validation.build_command = "make build"

        call_count = 0

        def side_effect(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if "make" in cmd:
                return RunResult(returncode=2, stdout="build failed", stderr="error", timed_out=False)
            return RunResult(returncode=0, stdout="OK", stderr="", timed_out=False)

        mock_run.side_effect = side_effect
        v = Validator(default_config)
        result = v.validate("/tmp")
        assert result.passed is False
        assert len(result.steps) == 3  # lint (skipped), tests, build
        assert result.steps[2].name == "build"
        assert result.steps[2].passed is False

    @patch("validator.run_with_group_kill")
    def test_all_commands_run_sequentially(self, mock_run, default_config):
        """When all commands are set, they run in order: lint, tests, build."""
        default_config.validation.lint_command = "ruff check ."
        default_config.validation.build_command = "make build"
        v = Validator(default_config)
        mock_run.return_value = RunResult(returncode=0, stdout="OK", stderr="", timed_out=False)
        result = v.validate("/tmp")
        assert result.passed is True
        assert len(result.steps) == 3
        assert result.steps[0].name == "lint"
        assert result.steps[1].name == "tests"
        assert result.steps[2].name == "build"

    @patch("validator.run_with_group_kill")
    def test_output_combines_stdout_and_stderr(self, mock_run, validator):
        """Step output should contain both stdout and stderr combined."""
        mock_run.return_value = RunResult(
            returncode=0, stdout="standard output\n", stderr="error output\n", timed_out=False
        )
        result = validator.validate("/tmp")
        test_steps = [s for s in result.steps if s.name == "tests"]
        assert "standard output" in test_steps[0].output
        assert "error output" in test_steps[0].output
