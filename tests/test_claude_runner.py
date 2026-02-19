"""Tests for claude_runner module."""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from claude_runner import ClaudeResult, ClaudeRunner
from config_schema import Config


@pytest.fixture
def runner(default_config):
    return ClaudeRunner(default_config)


class TestBuildCommand:
    def test_basic_command(self, runner):
        cmd = runner._build_command("Fix the bug")
        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert "Fix the bug" in cmd
        assert "--model" in cmd
        assert "opus" in cmd
        assert "--max-turns" in cmd
        assert "25" in cmd
        assert "--output-format" in cmd
        assert "json" in cmd

    def test_build_command_uses_resolved_model(self):
        config = Config()
        config.claude.resolved_model = "claude-opus-4-6"
        runner = ClaudeRunner(config)
        cmd = runner._build_command("test")
        model_idx = cmd.index("--model") + 1
        assert cmd[model_idx] == "claude-opus-4-6"

    def test_build_command_falls_back_to_alias(self):
        config = Config()
        config.claude.resolved_model = ""
        runner = ClaudeRunner(config)
        cmd = runner._build_command("test")
        model_idx = cmd.index("--model") + 1
        assert cmd[model_idx] == "opus"


class TestParseJsonResponse:
    def test_clean_json(self, runner):
        stdout = '{"result": "Fixed it", "cost_usd": 0.05}'
        data = runner._parse_json_response(stdout)
        assert data["result"] == "Fixed it"
        assert data["cost_usd"] == 0.05

    def test_json_with_banner_lines(self, runner):
        stdout = (
            "Claude Code v1.0\n"
            "Loading...\n"
            '{"result": "Done", "cost_usd": 0.01}\n'
        )
        data = runner._parse_json_response(stdout)
        assert data["result"] == "Done"

    def test_no_json_raises(self, runner):
        stdout = "No JSON here\nJust text\n"
        with pytest.raises(ValueError, match="No JSON"):
            runner._parse_json_response(stdout)

    def test_invalid_json_raises(self, runner):
        stdout = "{not valid json}"
        with pytest.raises(ValueError, match="No JSON"):
            runner._parse_json_response(stdout)

    def test_json_with_trailing_output(self, runner):
        stdout = '{"result": "Done"}\nSome warning log\nAnother line\n'
        data = runner._parse_json_response(stdout)
        assert data["result"] == "Done"

    def test_json_with_banner_and_trailing_output(self, runner):
        stdout = 'Banner\n{"result": "Done"}\nTrailing log\n'
        data = runner._parse_json_response(stdout)
        assert data["result"] == "Done"

    def test_json_with_nested_braces(self, runner):
        stdout = '{"result": "Done", "meta": {"key": "val"}}\nlog line\n'
        data = runner._parse_json_response(stdout)
        assert data["result"] == "Done"
        assert data["meta"] == {"key": "val"}

    def test_json_with_braces_in_strings(self, runner):
        stdout = '{"result": "Fixed {thing}"}\nwarning\n'
        data = runner._parse_json_response(stdout)
        assert data["result"] == "Fixed {thing}"

    def test_incomplete_json_raises(self, runner):
        stdout = '{"result": "Done"'
        with pytest.raises(ValueError, match="No JSON"):
            runner._parse_json_response(stdout)

    def test_json_with_array_of_objects(self, runner):
        stdout = '{"result": "Done", "items": [{"a": 1}, {"b": 2}]}\n'
        data = runner._parse_json_response(stdout)
        assert data["items"] == [{"a": 1}, {"b": 2}]

    def test_json_with_unicode_escape_quote(self, runner):
        stdout = '{"result": "value with \\u0022quoted\\u0022 text"}\n'
        data = runner._parse_json_response(stdout)
        assert data["result"] == 'value with "quoted" text'

    def test_json_with_deeply_nested_structure(self, runner):
        stdout = '{"a": {"b": {"c": {"d": [1, 2, {"e": 3}]}}}}\ntrailing\n'
        data = runner._parse_json_response(stdout)
        assert data["a"]["b"]["c"]["d"] == [1, 2, {"e": 3}]

    def test_json_with_braces_in_banner(self, runner):
        stdout = 'Info: config={debug: true}\n{"result": "Done"}\n'
        data = runner._parse_json_response(stdout)
        assert data["result"] == "Done"

    def test_json_after_array_line(self, runner):
        stdout = '[{"status": "ok"}, {"status": "done"}]\n{"result": "Done"}\n'
        data = runner._parse_json_response(stdout)
        assert data["result"] == "Done"

    def test_json_after_nested_object_in_banner(self, runner):
        stdout = 'Progress: {"step": 1, "total": 3}\n{"result": "Done", "cost_usd": 0.01}\n'
        data = runner._parse_json_response(stdout)
        assert data["result"] == "Done"

    def test_multiline_json_object(self, runner):
        stdout = 'Banner\n{\n  "result": "Done",\n  "cost_usd": 0.01\n}\nTrailing\n'
        data = runner._parse_json_response(stdout)
        assert data["result"] == "Done"

    def test_json_mid_line_with_prefix_text(self, runner):
        stdout = 'Some info: {"result": "Done", "cost_usd": 0.01}\n'
        data = runner._parse_json_response(stdout)
        assert data["result"] == "Done"
        assert data["cost_usd"] == 0.01

    def test_json_mid_line_with_banner_and_prefix(self, runner):
        stdout = 'Banner line\nProgress: {"result": "Done"}\nTrailing\n'
        data = runner._parse_json_response(stdout)
        assert data["result"] == "Done"

    def test_multiline_json_starting_mid_line(self, runner):
        stdout = 'Prefix text {\n  "result": "Done",\n  "cost_usd": 0.01\n}\n'
        data = runner._parse_json_response(stdout)
        assert data["result"] == "Done"
        assert data["cost_usd"] == 0.01

    def test_mid_line_json_prefers_valid_object(self, runner):
        stdout = 'bad {stuff\n{"result": "Done"}\n'
        data = runner._parse_json_response(stdout)
        assert data["result"] == "Done"

    def test_multiline_json_with_multi_line_preceding_text(self, runner):
        stdout = 'Banner line 1\nBanner line 2\nMore text {\n  "result": "Done",\n  "cost_usd": 0.01\n}\n'
        data = runner._parse_json_response(stdout)
        assert data["result"] == "Done"
        assert data["cost_usd"] == 0.01

    def test_json_with_object_literal_in_string_value(self, runner):
        """JSON with escaped braces and quotes inside string values."""
        stdout = '{"result": "Use {\\"key\\": \\"val\\"} syntax", "cost_usd": 0.01}\n'
        data = runner._parse_json_response(stdout)
        assert data["result"] == 'Use {"key": "val"} syntax'
        assert data["cost_usd"] == 0.01

    def test_multiline_json_with_braces_in_strings(self, runner):
        """Multiline JSON where string values contain unmatched braces."""
        stdout = (
            'Banner\n'
            '{\n'
            '  "result": "Fixed {broken} stuff",\n'
            '  "details": "Replaced { with }",\n'
            '  "cost_usd": 0.02\n'
            '}\n'
        )
        data = runner._parse_json_response(stdout)
        assert data["result"] == "Fixed {broken} stuff"
        assert data["details"] == "Replaced { with }"
        assert data["cost_usd"] == 0.02


def _make_popen_mock(returncode=0, stdout="", stderr="", communicate_effect=None):
    """Create a mock subprocess.Popen that behaves like the real thing."""
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    mock_proc.returncode = returncode
    if communicate_effect is not None:
        mock_proc.communicate.side_effect = communicate_effect
    else:
        mock_proc.communicate.return_value = (stdout, stderr)
    return mock_proc


class TestRun:
    @patch("claude_runner.subprocess.Popen")
    def test_successful_run(self, mock_popen, runner):
        mock_popen.return_value = _make_popen_mock(
            returncode=0,
            stdout='{"result": "Fixed bug", "cost_usd": 0.03, "duration_seconds": 12.5}',
        )
        result = runner.run("Fix the bug")
        assert result.success is True
        assert result.result_text == "Fixed bug"
        assert result.cost_usd == 0.03
        assert result.duration_seconds == 12.5

    @patch("claude_runner.subprocess.Popen")
    def test_timeout(self, mock_popen, runner):
        mock_popen.return_value = _make_popen_mock(
            communicate_effect=subprocess.TimeoutExpired(cmd="claude", timeout=300),
        )
        result = runner.run("Fix the bug")
        assert result.success is False
        assert "timed out" in result.error

    @patch("claude_runner.subprocess.Popen")
    def test_command_not_found(self, mock_popen, runner):
        mock_popen.side_effect = FileNotFoundError()
        result = runner.run("Fix the bug")
        assert result.success is False
        assert "not found" in result.error

    @patch("claude_runner.subprocess.Popen")
    def test_nonzero_exit(self, mock_popen, runner):
        mock_popen.return_value = _make_popen_mock(
            returncode=1,
            stderr="Error occurred",
        )
        result = runner.run("Fix the bug")
        assert result.success is False
        assert "exited with code 1" in result.error

    @patch("claude_runner.subprocess.Popen")
    def test_unparseable_output(self, mock_popen, runner):
        mock_popen.return_value = _make_popen_mock(
            returncode=0,
            stdout="Not JSON at all",
        )
        result = runner.run("Fix the bug")
        assert result.success is False
        assert "parse" in result.error.lower()


class TestRetryLogic:
    @patch("claude_runner.time.sleep")
    @patch("claude_runner.subprocess.Popen")
    def test_retry_on_timeout(self, mock_popen, mock_sleep, runner):
        """Retries on TimeoutExpired, succeeds on third attempt."""
        mock_popen.side_effect = [
            _make_popen_mock(communicate_effect=subprocess.TimeoutExpired(cmd="claude", timeout=300)),
            _make_popen_mock(communicate_effect=subprocess.TimeoutExpired(cmd="claude", timeout=300)),
            _make_popen_mock(returncode=0, stdout='{"result": "Done", "cost_usd": 0.01}'),
        ]
        result = runner.run("Fix the bug")
        assert result.success is True
        assert result.result_text == "Done"
        assert mock_popen.call_count == 3
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(2)
        mock_sleep.assert_any_call(8)

    @patch("claude_runner.time.sleep")
    @patch("claude_runner.subprocess.Popen")
    def test_retry_on_nonzero_exit(self, mock_popen, mock_sleep, runner):
        """Retries on non-zero exit code, succeeds on third attempt."""
        mock_popen.side_effect = [
            _make_popen_mock(returncode=1, stderr="rate limited"),
            _make_popen_mock(returncode=1, stderr="rate limited"),
            _make_popen_mock(returncode=0, stdout='{"result": "Fixed", "cost_usd": 0.02}'),
        ]
        result = runner.run("Fix the bug")
        assert result.success is True
        assert result.result_text == "Fixed"
        assert mock_popen.call_count == 3

    @patch("claude_runner.time.sleep")
    @patch("claude_runner.subprocess.Popen")
    def test_no_retry_on_file_not_found(self, mock_popen, mock_sleep, runner):
        """FileNotFoundError is not retryable — returns immediately."""
        mock_popen.side_effect = FileNotFoundError()
        result = runner.run("Fix the bug")
        assert result.success is False
        assert "not found" in result.error
        assert mock_popen.call_count == 1
        mock_sleep.assert_not_called()

    @patch("claude_runner.time.sleep")
    @patch("claude_runner.subprocess.Popen")
    def test_no_retry_on_json_parse_failure(self, mock_popen, mock_sleep, runner):
        """JSON parse failure is not retryable — CLI ran fine, output was bad."""
        mock_popen.return_value = _make_popen_mock(
            returncode=0,
            stdout="Not JSON at all",
        )
        result = runner.run("Fix the bug")
        assert result.success is False
        assert "parse" in result.error.lower()
        assert mock_popen.call_count == 1
        mock_sleep.assert_not_called()

    @patch("claude_runner.time.sleep")
    @patch("claude_runner.subprocess.Popen")
    def test_all_retries_exhausted(self, mock_popen, mock_sleep, runner):
        """All retries exhausted returns failure."""
        mock_popen.side_effect = [
            _make_popen_mock(communicate_effect=subprocess.TimeoutExpired(cmd="claude", timeout=300))
            for _ in range(4)
        ]
        result = runner.run("Fix the bug")
        assert result.success is False
        assert "timed out" in result.error
        assert mock_popen.call_count == 4  # 1 initial + 3 retries
        assert mock_sleep.call_count == 3

    @patch("claude_runner.time.sleep")
    @patch("claude_runner.subprocess.Popen")
    def test_retry_delays_are_exponential(self, mock_popen, mock_sleep, runner):
        """Verify exponential backoff delays: 2, 8, 32."""
        mock_popen.side_effect = [
            _make_popen_mock(communicate_effect=subprocess.TimeoutExpired(cmd="claude", timeout=300))
            for _ in range(4)
        ]
        runner.run("Fix the bug")
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == [2, 8, 32]

    @patch("claude_runner.time.sleep")
    @patch("claude_runner.subprocess.Popen")
    def test_retry_on_os_error(self, mock_popen, mock_sleep, runner):
        """Retries on OSError, succeeds on second attempt."""
        mock_popen.side_effect = [
            OSError("Connection reset"),
            _make_popen_mock(returncode=0, stdout='{"result": "Done", "cost_usd": 0.01}'),
        ]
        result = runner.run("Fix the bug")
        assert result.success is True
        assert mock_popen.call_count == 2
        mock_sleep.assert_called_once_with(2)


class TestRateLimitBackoff:
    @patch("claude_runner.time.sleep")
    @patch("claude_runner.subprocess.Popen")
    def test_rate_limit_exponential_backoff(self, mock_popen, mock_sleep, runner):
        """Rate limit errors should use exponential backoff (5, 15, 45) not fixed delays."""
        mock_popen.side_effect = [
            _make_popen_mock(returncode=1, stderr="rate limit exceeded"),
            _make_popen_mock(returncode=1, stderr="rate limit exceeded"),
            _make_popen_mock(returncode=1, stderr="rate limit exceeded"),
            _make_popen_mock(returncode=1, stderr="rate limit exceeded"),
        ]
        result = runner.run("Fix the bug")
        assert result.success is False
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        # Exponential: 5 * 3^0 = 5, 5 * 3^1 = 15, 5 * 3^2 = 45
        assert delays == [5, 15, 45]

    @patch("claude_runner.time.sleep")
    @patch("claude_runner.subprocess.Popen")
    def test_non_rate_limit_uses_fixed_delays(self, mock_popen, mock_sleep, runner):
        """Non-rate-limit errors should still use the fixed delays (2, 8, 32)."""
        mock_popen.side_effect = [
            _make_popen_mock(returncode=1, stderr="some other error"),
            _make_popen_mock(returncode=1, stderr="some other error"),
            _make_popen_mock(returncode=1, stderr="some other error"),
            _make_popen_mock(returncode=1, stderr="some other error"),
        ]
        result = runner.run("Fix the bug")
        assert result.success is False
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == [2, 8, 32]

    @patch("claude_runner.time.sleep")
    @patch("claude_runner.subprocess.Popen")
    def test_rate_limit_429_detected(self, mock_popen, mock_sleep, runner):
        """429 in stderr should trigger rate limit backoff."""
        mock_popen.side_effect = [
            _make_popen_mock(returncode=1, stderr="HTTP 429 Too Many Requests"),
            _make_popen_mock(returncode=0, stdout='{"result": "Done", "cost_usd": 0.01}'),
        ]
        result = runner.run("Fix the bug")
        assert result.success is True
        mock_sleep.assert_called_once_with(5)  # 5 * 3^0 = 5
