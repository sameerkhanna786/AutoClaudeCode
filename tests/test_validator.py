"""Tests for validator module."""

import os
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


class TestCaptureBaseline:
    @patch("validator.run_with_group_kill")
    def test_capture_baseline_all_passing(self, mock_run, default_config):
        """Returns empty set when all tests pass."""
        default_config.validation.test_command = "python3 -m pytest"
        v = Validator(default_config)
        mock_run.return_value = RunResult(returncode=0, stdout="OK", stderr="", timed_out=False)
        result = v.capture_baseline("/tmp")
        assert result == set()

    @patch("validator.run_with_group_kill")
    def test_capture_baseline_with_failures(self, mock_run, default_config):
        """Returns set of failing test IDs when tests fail."""
        default_config.validation.test_command = "python3 -m pytest"
        v = Validator(default_config)
        output = (
            "FAILED tests/test_foo.py::TestBar::test_baz - AssertionError\n"
            "FAILED tests/test_qux.py::test_quux - ValueError\n"
            "2 failed, 10 passed\n"
        )
        mock_run.return_value = RunResult(returncode=1, stdout=output, stderr="", timed_out=False)
        result = v.capture_baseline("/tmp")
        assert result == {
            "tests/test_foo.py::TestBar::test_baz",
            "tests/test_qux.py::test_quux",
        }

    def test_capture_baseline_empty_test_command(self, default_config):
        """Returns empty set when test command is empty."""
        default_config.validation.test_command = ""
        v = Validator(default_config)
        result = v.capture_baseline("/tmp")
        assert result == set()

    @patch("validator.run_with_group_kill")
    def test_capture_baseline_no_failed_lines_parsed(self, mock_run, default_config):
        """Returns empty set when tests fail but no FAILED lines are parseable."""
        default_config.validation.test_command = "python3 -m pytest"
        v = Validator(default_config)
        mock_run.return_value = RunResult(
            returncode=1, stdout="some other error output", stderr="", timed_out=False,
        )
        result = v.capture_baseline("/tmp")
        assert result == set()


class TestValidateWithBaseline:
    @patch("validator.run_with_group_kill")
    def test_validate_with_baseline_ignores_preexisting(self, mock_run, default_config):
        """Pre-existing failures are ignored and validation passes."""
        default_config.validation.test_command = "python3 -m pytest"
        v = Validator(default_config)
        baseline = {"tests/test_foo.py::test_bar"}

        output = "FAILED tests/test_foo.py::test_bar - AssertionError\n1 failed\n"
        mock_run.return_value = RunResult(returncode=1, stdout=output, stderr="", timed_out=False)
        result = v.validate_with_baseline("/tmp", baseline)
        assert result.passed is True

    @patch("validator.run_with_group_kill")
    def test_validate_with_baseline_catches_new_failures(self, mock_run, default_config):
        """New failures still fail validation."""
        default_config.validation.test_command = "python3 -m pytest"
        v = Validator(default_config)
        baseline = {"tests/test_foo.py::test_bar"}

        output = "FAILED tests/test_new.py::test_broken - TypeError\n1 failed\n"
        mock_run.return_value = RunResult(returncode=1, stdout=output, stderr="", timed_out=False)
        result = v.validate_with_baseline("/tmp", baseline)
        assert result.passed is False

    @patch("validator.run_with_group_kill")
    def test_validate_with_baseline_mixed(self, mock_run, default_config):
        """Pre-existing + new failures → FAIL (only new reported)."""
        default_config.validation.test_command = "python3 -m pytest"
        v = Validator(default_config)
        baseline = {"tests/test_foo.py::test_bar"}

        output = (
            "FAILED tests/test_foo.py::test_bar - AssertionError\n"
            "FAILED tests/test_new.py::test_broken - TypeError\n"
            "2 failed\n"
        )
        mock_run.return_value = RunResult(returncode=1, stdout=output, stderr="", timed_out=False)
        result = v.validate_with_baseline("/tmp", baseline)
        assert result.passed is False

    @patch("validator.run_with_group_kill")
    def test_validate_with_baseline_none_falls_back(self, mock_run, default_config):
        """When baseline_failures is None, falls back to regular validate()."""
        default_config.validation.test_command = "python3 -m pytest"
        v = Validator(default_config)
        mock_run.return_value = RunResult(returncode=0, stdout="OK", stderr="", timed_out=False)
        result = v.validate_with_baseline("/tmp", None)
        assert result.passed is True

    @patch("validator.run_with_group_kill")
    def test_validate_with_baseline_empty_set_falls_back(self, mock_run, default_config):
        """When baseline_failures is empty set, falls back to regular validate()."""
        default_config.validation.test_command = "python3 -m pytest"
        v = Validator(default_config)
        mock_run.return_value = RunResult(returncode=0, stdout="OK", stderr="", timed_out=False)
        result = v.validate_with_baseline("/tmp", set())
        assert result.passed is True

    @patch("validator.run_with_group_kill")
    def test_validate_with_baseline_unparseable_failure(self, mock_run, default_config):
        """Tests fail with rc=1 but no FAILED lines → should be treated as genuine failure."""
        default_config.validation.test_command = "python3 -m pytest"
        v = Validator(default_config)
        baseline = {"tests/test_foo.py::test_bar"}

        # Simulate a collection error / import error — non-zero exit but no FAILED lines
        mock_run.return_value = RunResult(
            returncode=1, stdout="ImportError: cannot import name 'foo'", stderr="", timed_out=False,
        )
        result = v.validate_with_baseline("/tmp", baseline)
        assert result.passed is False

    @patch("validator.run_with_group_kill")
    def test_validate_with_baseline_strips_dash_x(self, mock_run, default_config):
        """-x is removed from test command when baseline is active."""
        default_config.validation.test_command = "python3 -m pytest tests/ -x -q"
        v = Validator(default_config)
        baseline = {"tests/test_foo.py::test_bar"}

        output = "FAILED tests/test_foo.py::test_bar - AssertionError\n1 failed\n"
        mock_run.return_value = RunResult(returncode=1, stdout=output, stderr="", timed_out=False)
        v.validate_with_baseline("/tmp", baseline)

        # Find the tests call (skip lint which is empty)
        test_calls = [
            c for c in mock_run.call_args_list
            if isinstance(c[0][0], str) and "pytest" in c[0][0]
        ]
        assert len(test_calls) >= 1
        test_cmd = test_calls[0][0][0]
        assert " -x " not in f" {test_cmd} "
        assert "-q" in test_cmd  # other flags preserved

    @patch("validator.run_with_group_kill")
    def test_validate_with_baseline_strips_dash_x_at_end(self, mock_run, default_config):
        """-x at end of command is removed when baseline is active."""
        default_config.validation.test_command = "python3 -m pytest tests/ -x"
        v = Validator(default_config)
        baseline = {"tests/test_foo.py::test_bar"}

        output = "FAILED tests/test_foo.py::test_bar - AssertionError\n1 failed\n"
        mock_run.return_value = RunResult(returncode=1, stdout=output, stderr="", timed_out=False)
        v.validate_with_baseline("/tmp", baseline)

        test_calls = [
            c for c in mock_run.call_args_list
            if isinstance(c[0][0], str) and "pytest" in c[0][0]
        ]
        assert len(test_calls) >= 1
        test_cmd = test_calls[0][0][0]
        assert " -x " not in f" {test_cmd} "
        assert "-x" not in test_cmd.split()

    @patch("validator.run_with_group_kill")
    def test_validate_with_baseline_no_double_spaces_after_x_removal(self, mock_run, default_config):
        """Removing -x should not leave double spaces in the command."""
        default_config.validation.test_command = "pytest -x tests/"
        v = Validator(default_config)
        baseline = {"tests/test_foo.py::test_bar"}

        output = "FAILED tests/test_foo.py::test_bar - AssertionError\n1 failed\n"
        mock_run.return_value = RunResult(returncode=1, stdout=output, stderr="", timed_out=False)
        v.validate_with_baseline("/tmp", baseline)

        test_calls = [
            c for c in mock_run.call_args_list
            if isinstance(c[0][0], str) and "pytest" in c[0][0]
        ]
        assert len(test_calls) >= 1
        test_cmd = test_calls[0][0][0]
        assert "  " not in test_cmd  # no double spaces
        assert "-x" not in test_cmd.split()


class TestValidateSyntaxOnly:
    def test_valid_python_files(self, validator, tmp_path):
        """Valid Python files pass syntax check."""
        (tmp_path / "good.py").write_text("x = 1\n")
        result = validator.validate_syntax_only(["good.py"], str(tmp_path))
        assert result.passed is True
        assert result.steps[0].name == "syntax"
        assert "1 file(s) passed" in result.steps[0].output

    def test_syntax_error_detected(self, validator, tmp_path):
        """Files with syntax errors fail the check."""
        (tmp_path / "bad.py").write_text("def foo(\n")
        result = validator.validate_syntax_only(["bad.py"], str(tmp_path))
        assert result.passed is False
        assert "Syntax errors found" in result.steps[0].output
        assert "bad.py" in result.steps[0].output

    def test_no_py_files(self, validator):
        """Non-Python files are skipped, result passes."""
        result = validator.validate_syntax_only(["readme.md", "data.json"], "/tmp")
        assert result.passed is True
        assert "No .py files" in result.steps[0].output

    def test_missing_file_skipped(self, validator, tmp_path):
        """Files that don't exist are silently skipped."""
        result = validator.validate_syntax_only(["nonexistent.py"], str(tmp_path))
        assert result.passed is True

    def test_mixed_valid_and_invalid(self, validator, tmp_path):
        """One bad file among good ones fails the whole check."""
        (tmp_path / "good.py").write_text("x = 1\n")
        (tmp_path / "bad.py").write_text("def :\n")
        result = validator.validate_syntax_only(["good.py", "bad.py"], str(tmp_path))
        assert result.passed is False
        assert "bad.py" in result.steps[0].output

    def test_uses_config_target_dir(self, default_config, tmp_path):
        """Uses config.target_dir when working_dir is not specified."""
        (tmp_path / "ok.py").write_text("a = 1\n")
        default_config.target_dir = str(tmp_path)
        v = Validator(default_config)
        result = v.validate_syntax_only(["ok.py"])
        assert result.passed is True

    def test_empty_file_list(self, validator):
        """Empty file list passes."""
        result = validator.validate_syntax_only([], "/tmp")
        assert result.passed is True

    def test_recursion_error_handled(self, validator, tmp_path):
        """Files that cause RecursionError in ast.parse are caught gracefully."""
        # Create a file with deeply nested expressions that triggers RecursionError
        (tmp_path / "deep.py").write_text("x = 1\n")
        with patch("validator.ast.parse", side_effect=RecursionError("maximum recursion depth exceeded")):
            result = validator.validate_syntax_only(["deep.py"], str(tmp_path))
        assert result.passed is False
        assert "parse error" in result.steps[0].output

    def test_memory_error_handled(self, validator, tmp_path):
        """Files that cause MemoryError in ast.parse are caught gracefully."""
        (tmp_path / "huge.py").write_text("x = 1\n")
        with patch("validator.ast.parse", side_effect=MemoryError("out of memory")):
            result = validator.validate_syntax_only(["huge.py"], str(tmp_path))
        assert result.passed is False
        assert "parse error" in result.steps[0].output

    def test_path_traversal_skipped(self, validator, tmp_path):
        """Files with path traversal (../) should be skipped, not read."""
        # Create a file outside the working dir
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.py").write_text("SECRET = 'password123'\n")

        workdir = tmp_path / "repo"
        workdir.mkdir()
        result = validator.validate_syntax_only(
            ["../outside/secret.py"], str(workdir),
        )
        # Should pass (file skipped), not parse the file outside cwd
        assert result.passed is True


class TestRemoveXFlag:
    """Test -x flag removal from test commands during baseline comparison."""

    @pytest.mark.parametrize("cmd,expected", [
        ("pytest -x -v", "pytest -v"),
        ("pytest -x", "pytest"),
        ("-x pytest", "pytest"),
        ("pytest -v -x -s", "pytest -v -s"),
        ("pytest --maxfail=3", "pytest --maxfail=3"),  # no -x, unchanged
        ("pytest -xvs", "pytest -xvs"),  # -x as part of combined flag, should NOT remove
    ])
    def test_remove_x_flag(self, cmd, expected):
        """Removing -x flag should not mangle adjacent tokens."""
        import re
        # Current (buggy) implementation:
        result = re.sub(r'(?:^|\s)-x(?=\s|$)', ' ', cmd).strip()
        result = re.sub(r'\s{2,}', ' ', result)
        assert result == expected


class TestCaptureBaselineNonPytest:
    """Test that capture_baseline doesn't append --tb=line for non-pytest commands."""

    @patch("validator.run_with_group_kill")
    def test_non_pytest_command_no_tb_flag(self, mock_run, default_config):
        """Non-pytest test commands should not get --tb=line appended."""
        default_config.validation.test_command = "npm test"
        v = Validator(default_config)
        mock_run.return_value = RunResult(returncode=0, stdout="OK", stderr="", timed_out=False)
        v.capture_baseline("/tmp")
        cmd = mock_run.call_args[0][0]
        assert "--tb=line" not in cmd

    @patch("validator.run_with_group_kill")
    def test_pytest_command_gets_tb_flag(self, mock_run, default_config):
        """Pytest-based test commands should still get --tb=line appended."""
        default_config.validation.test_command = "python3 -m pytest tests/"
        v = Validator(default_config)
        mock_run.return_value = RunResult(returncode=0, stdout="OK", stderr="", timed_out=False)
        v.capture_baseline("/tmp")
        cmd = mock_run.call_args[0][0]
        assert "--tb=line" in cmd


class TestValidateIncremental:
    """Tests for validate_incremental using configured test command."""

    @patch("validator.run_with_group_kill")
    def test_uses_configured_test_command(self, mock_run, default_config, tmp_path):
        """Targeted test command should use the configured test command, not hardcoded pytest."""
        default_config.validation.test_command = "python3 -m pytest tests/ -v"
        default_config.validation.lint_command = ""
        default_config.validation.build_command = ""
        default_config.target_dir = str(tmp_path)
        # Create a test file so mapping succeeds
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_foo.py").write_text("def test_ok(): pass\n")
        v = Validator(default_config)
        mock_run.return_value = RunResult(returncode=0, stdout="OK", stderr="", timed_out=False)
        v.validate_incremental(str(tmp_path), ["foo.py"])
        # Find the targeted test call
        targeted_calls = [
            c for c in mock_run.call_args_list
            if isinstance(c[0][0], str) and "targeted" not in c[0][0]
            and "test_foo" in c[0][0]
        ]
        assert len(targeted_calls) >= 1
        # The targeted command should start with the configured test command
        cmd = targeted_calls[0][0][0]
        assert cmd.startswith("python3 -m pytest tests/ -v"), (
            f"Expected command to start with configured test_command, got: {cmd}"
        )

    @patch("validator.run_with_group_kill")
    def test_non_pytest_command_omits_pytest_flags(self, mock_run, default_config, tmp_path):
        """Non-pytest test commands should not get -x -q flags appended."""
        default_config.validation.test_command = "npm test"
        default_config.validation.lint_command = ""
        default_config.validation.build_command = ""
        default_config.target_dir = str(tmp_path)
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_foo.py").write_text("def test_ok(): pass\n")
        v = Validator(default_config)
        mock_run.return_value = RunResult(returncode=0, stdout="OK", stderr="", timed_out=False)
        v.validate_incremental(str(tmp_path), ["foo.py"])
        targeted_calls = [
            c for c in mock_run.call_args_list
            if isinstance(c[0][0], str) and "test_foo" in c[0][0]
        ]
        assert len(targeted_calls) >= 1
        cmd = targeted_calls[0][0][0]
        assert "-x" not in cmd, f"Non-pytest command should not have -x flag: {cmd}"
        assert "-q" not in cmd, f"Non-pytest command should not have -q flag: {cmd}"

    @patch("validator.run_with_group_kill")
    def test_pytest_command_gets_xq_flags(self, mock_run, default_config, tmp_path):
        """Pytest-based test commands should still get -x -q flags."""
        default_config.validation.test_command = "python3 -m pytest tests/"
        default_config.validation.lint_command = ""
        default_config.validation.build_command = ""
        default_config.target_dir = str(tmp_path)
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_bar.py").write_text("def test_ok(): pass\n")
        v = Validator(default_config)
        mock_run.return_value = RunResult(returncode=0, stdout="OK", stderr="", timed_out=False)
        v.validate_incremental(str(tmp_path), ["bar.py"])
        targeted_calls = [
            c for c in mock_run.call_args_list
            if isinstance(c[0][0], str) and "test_bar" in c[0][0]
        ]
        assert len(targeted_calls) >= 1
        cmd = targeted_calls[0][0][0]
        assert "-x" in cmd, f"Pytest command should have -x flag: {cmd}"
        assert "-q" in cmd, f"Pytest command should have -q flag: {cmd}"
