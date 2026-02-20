"""Tests for claude_runner module."""

import json
import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

from claude_runner import CircuitBreaker, ClaudeResult, ClaudeRunner
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
    def test_cwd_always_uses_target_dir(self, mock_popen, runner):
        """cwd should always be config.target_dir."""
        mock_popen.return_value = _make_popen_mock(
            returncode=0,
            stdout='{"result": "Done", "cost_usd": 0.01}',
        )
        runner.run("Fix the bug")
        popen_call = mock_popen.call_args
        assert popen_call.kwargs["cwd"] == runner.config.target_dir

    @patch("claude_runner.subprocess.Popen")
    def test_add_dirs_appends_flags(self, mock_popen, runner):
        """add_dirs should append --add-dir flags to the command."""
        mock_popen.return_value = _make_popen_mock(
            returncode=0,
            stdout='{"result": "Done", "cost_usd": 0.01}',
        )
        runner.run("Fix the bug", add_dirs=["/tmp/worktree-0", "/tmp/worktree-1"])
        cmd = mock_popen.call_args[0][0]
        # Find all --add-dir flags
        add_dir_indices = [i for i, x in enumerate(cmd) if x == "--add-dir"]
        assert len(add_dir_indices) == 2
        dirs = [cmd[i + 1] for i in add_dir_indices]
        assert "/tmp/worktree-0" in dirs[0]
        assert "/tmp/worktree-1" in dirs[1]

    @patch("claude_runner.subprocess.Popen")
    def test_add_dirs_none_no_flags(self, mock_popen, runner):
        """When add_dirs is None, no --add-dir flags should be added."""
        mock_popen.return_value = _make_popen_mock(
            returncode=0,
            stdout='{"result": "Done", "cost_usd": 0.01}',
        )
        runner.run("Fix the bug")
        cmd = mock_popen.call_args[0][0]
        assert "--add-dir" not in cmd

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


class TestCostParsing:
    """Test that cost_usd and duration_seconds are parsed correctly from CLI output."""

    @patch("claude_runner.subprocess.Popen")
    def test_new_field_names(self, mock_popen, runner):
        """total_cost_usd and duration_ms are the actual CLI field names."""
        mock_popen.return_value = _make_popen_mock(
            returncode=0,
            stdout='{"result": "Done", "total_cost_usd": 0.037, "duration_ms": 2467}',
        )
        result = runner.run("Fix")
        assert result.success is True
        assert result.cost_usd == 0.037
        assert abs(result.duration_seconds - 2.467) < 0.001

    @patch("claude_runner.subprocess.Popen")
    def test_old_field_names_fallback(self, mock_popen, runner):
        """Falls back to cost_usd and duration_seconds for backward compat."""
        mock_popen.return_value = _make_popen_mock(
            returncode=0,
            stdout='{"result": "Done", "cost_usd": 0.05, "duration_seconds": 12.5}',
        )
        result = runner.run("Fix")
        assert result.success is True
        assert result.cost_usd == 0.05
        assert result.duration_seconds == 12.5

    @patch("claude_runner.subprocess.Popen")
    def test_new_fields_take_precedence(self, mock_popen, runner):
        """When both old and new fields are present, new ones win."""
        mock_popen.return_value = _make_popen_mock(
            returncode=0,
            stdout='{"result": "Done", "total_cost_usd": 0.1, "cost_usd": 0.05, "duration_ms": 5000, "duration_seconds": 2.0}',
        )
        result = runner.run("Fix")
        assert result.success is True
        assert result.cost_usd == 0.1
        assert result.duration_seconds == 5.0

    @patch("claude_runner.subprocess.Popen")
    def test_missing_cost_fields_default_to_zero(self, mock_popen, runner):
        """When no cost/duration fields, defaults to 0."""
        mock_popen.return_value = _make_popen_mock(
            returncode=0,
            stdout='{"result": "Done"}',
        )
        result = runner.run("Fix")
        assert result.success is True
        assert result.cost_usd == 0.0
        assert result.duration_seconds == 0.0

    @patch("claude_runner.subprocess.Popen")
    def test_duration_ms_zero_falls_back_to_seconds(self, mock_popen, runner):
        """duration_ms=0 should fall back to duration_seconds."""
        mock_popen.return_value = _make_popen_mock(
            returncode=0,
            stdout='{"result": "Done", "duration_ms": 0, "duration_seconds": 3.5}',
        )
        result = runner.run("Fix")
        assert result.success is True
        assert result.duration_seconds == 3.5


class TestMissingResultField:
    """Test handling of JSON responses missing the 'result' field."""

    @patch("claude_runner.subprocess.Popen")
    def test_missing_result_field_logs_warning(self, mock_popen, runner, caplog):
        """JSON without 'result' key should log a warning."""
        import logging
        mock_popen.return_value = _make_popen_mock(
            returncode=0,
            stdout='{"cost_usd": 0.05}',
        )
        with caplog.at_level(logging.WARNING):
            result = runner.run("Fix")
        assert result.success is True
        assert result.result_text == ""
        assert any("missing 'result' field" in r.message for r in caplog.records)

    @patch("claude_runner.subprocess.Popen")
    def test_missing_result_field_returns_success(self, mock_popen, runner):
        """Result should still be success=True with empty result_text when 'result' is missing."""
        mock_popen.return_value = _make_popen_mock(
            returncode=0,
            stdout='{"total_cost_usd": 0.03, "duration_ms": 1000}',
        )
        result = runner.run("Fix")
        assert result.success is True
        assert result.result_text == ""
        assert result.cost_usd == 0.03
        assert result.duration_seconds == 1.0


class TestCircuitBreaker:
    """Test circuit breaker state transitions and recovery behavior."""

    def test_initial_state_is_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitBreaker.STATE_CLOSED

    def test_allow_request_when_closed(self):
        cb = CircuitBreaker()
        assert cb.allow_request() is True

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker(failure_threshold=5)
        for _ in range(4):
            cb.record_failure()
        assert cb.state == CircuitBreaker.STATE_CLOSED
        assert cb.allow_request() is True

    def test_opens_at_failure_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitBreaker.STATE_OPEN
        assert cb.allow_request() is False

    def test_blocks_requests_when_open(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreaker.STATE_OPEN
        assert cb.allow_request() is False
        assert cb.allow_request() is False

    def test_transitions_to_half_open_after_recovery_timeout(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreaker.STATE_OPEN
        # Wait for recovery timeout
        time.sleep(1.1)
        assert cb.state == CircuitBreaker.STATE_HALF_OPEN

    def test_half_open_allows_limited_calls(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0, half_open_max_calls=1)
        cb.record_failure()
        cb.record_failure()
        # Recovery timeout is 0, should immediately transition
        assert cb.state == CircuitBreaker.STATE_HALF_OPEN
        # First call allowed
        assert cb.allow_request() is True
        # Second call blocked (max=1)
        assert cb.allow_request() is False

    def test_half_open_success_resets_to_closed(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0)
        cb.record_failure()
        cb.record_failure()
        # Force transition to HALF_OPEN
        assert cb.state == CircuitBreaker.STATE_HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitBreaker.STATE_CLOSED
        assert cb.allow_request() is True

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0, half_open_max_calls=1)
        cb.record_failure()
        cb.record_failure()
        # Force transition to HALF_OPEN (recovery_timeout=0 means immediate transition)
        _ = cb.state  # triggers transition
        assert cb.allow_request() is True  # probe call
        cb.record_failure()
        # record_failure sets state to OPEN, but with recovery_timeout=0 the
        # .state property immediately transitions back to HALF_OPEN.
        # Verify the internal state was set to OPEN by record_failure:
        assert cb._state == CircuitBreaker.STATE_OPEN
        # And that accessing .state triggers the immediate recovery transition:
        assert cb.state == CircuitBreaker.STATE_HALF_OPEN

    def test_half_open_failure_reopens_with_timeout(self):
        """Verify that a failure during HALF_OPEN re-opens the circuit with a real timeout."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=300, half_open_max_calls=1)
        cb.record_failure()
        cb.record_failure()
        # Manually transition to HALF_OPEN for testing
        cb._state = CircuitBreaker.STATE_HALF_OPEN
        cb._half_open_calls = 0
        assert cb.allow_request() is True  # probe call
        cb.record_failure()
        assert cb.state == CircuitBreaker.STATE_OPEN

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=5)
        for _ in range(4):
            cb.record_failure()
        cb.record_success()
        assert cb.state == CircuitBreaker.STATE_CLOSED
        # Need 5 more failures to open again
        for _ in range(4):
            cb.record_failure()
        assert cb.state == CircuitBreaker.STATE_CLOSED

    def test_manual_reset(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreaker.STATE_OPEN
        cb.reset()
        assert cb.state == CircuitBreaker.STATE_CLOSED
        assert cb.allow_request() is True

    def test_custom_parameters(self):
        cb = CircuitBreaker(failure_threshold=10, recovery_timeout=600, half_open_max_calls=3)
        assert cb.failure_threshold == 10
        assert cb.recovery_timeout == 600
        assert cb.half_open_max_calls == 3


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with ClaudeRunner."""

    @patch("claude_runner.subprocess.Popen")
    def test_run_blocked_by_open_circuit_breaker(self, mock_popen, runner):
        """When circuit breaker is open, run() should return failure without calling subprocess."""
        runner.circuit_breaker._state = CircuitBreaker.STATE_OPEN
        runner.circuit_breaker._opened_at = time.monotonic()  # recent, so won't auto-transition
        result = runner.run("Fix the bug")
        assert result.success is False
        assert "Circuit breaker" in result.error
        mock_popen.assert_not_called()

    def test_is_circuit_breaker_error_patterns(self):
        """Verify all expected error patterns are detected."""
        assert ClaudeRunner._is_circuit_breaker_error("rate limit exceeded") is True
        assert ClaudeRunner._is_circuit_breaker_error("HTTP 429 Too Many Requests") is True
        assert ClaudeRunner._is_circuit_breaker_error("Error 500 Internal") is True
        assert ClaudeRunner._is_circuit_breaker_error("502 Bad Gateway") is True
        assert ClaudeRunner._is_circuit_breaker_error("503 Service Unavailable") is True
        assert ClaudeRunner._is_circuit_breaker_error("504 Gateway Timeout") is True
        assert ClaudeRunner._is_circuit_breaker_error("server is overloaded") is True
        assert ClaudeRunner._is_circuit_breaker_error("normal error") is False
        assert ClaudeRunner._is_circuit_breaker_error("success") is False
