"""Tests for model_resolver module."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from model_resolver import resolve_model_id


class TestResolveModelId:
    @patch("model_resolver.subprocess.run")
    def test_successful_resolution(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "", "modelUsage": {"claude-opus-4-6": {"inputTokens": 5}}}',
            stderr="",
        )
        result = resolve_model_id("opus")
        assert result == "claude-opus-4-6"

    @patch("model_resolver.subprocess.run")
    def test_timeout_returns_none(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=30)
        result = resolve_model_id("opus")
        assert result is None

    @patch("model_resolver.subprocess.run")
    def test_cli_not_found_returns_none(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        result = resolve_model_id("opus")
        assert result is None

    @patch("model_resolver.subprocess.run")
    def test_nonzero_exit_returns_none(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error: unknown model",
        )
        result = resolve_model_id("opus")
        assert result is None

    @patch("model_resolver.subprocess.run")
    def test_invalid_json_returns_none(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Not JSON at all\nJust text\n",
            stderr="",
        )
        result = resolve_model_id("opus")
        assert result is None

    @patch("model_resolver.subprocess.run")
    def test_missing_model_usage_returns_none(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "done", "cost_usd": 0.01}',
            stderr="",
        )
        result = resolve_model_id("opus")
        assert result is None

    @patch("model_resolver.subprocess.run")
    def test_empty_model_usage_returns_none(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "done", "modelUsage": {}}',
            stderr="",
        )
        result = resolve_model_id("opus")
        assert result is None

    @patch("model_resolver.subprocess.run")
    def test_banner_lines_before_json(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "Claude Code v1.0\n"
                "Loading...\n"
                '{"result": "", "modelUsage": {"claude-sonnet-4-20250514": {"inputTokens": 5}}}\n'
            ),
            stderr="",
        )
        result = resolve_model_id("sonnet")
        assert result == "claude-sonnet-4-20250514"

    @patch("model_resolver.subprocess.run")
    def test_os_error_returns_none(self, mock_run):
        mock_run.side_effect = OSError("Connection refused")
        result = resolve_model_id("opus")
        assert result is None

    @patch("model_resolver.subprocess.run")
    def test_custom_command_and_timeout(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "", "modelUsage": {"claude-opus-4-6": {"inputTokens": 5}}}',
            stderr="",
        )
        result = resolve_model_id("opus", claude_command="/usr/local/bin/claude", timeout=60)
        assert result == "claude-opus-4-6"
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/usr/local/bin/claude"
        assert mock_run.call_args[1]["timeout"] == 60

    @patch("model_resolver.subprocess.run")
    def test_model_usage_not_dict_returns_none(self, mock_run):
        """modelUsage present but not a dict should return None."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "", "modelUsage": "not-a-dict"}',
            stderr="",
        )
        result = resolve_model_id("opus")
        assert result is None

    @patch("model_resolver.subprocess.run")
    def test_multiple_json_lines_picks_first_with_model_usage(self, mock_run):
        """When stdout has multiple JSON lines, the first with modelUsage wins."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                '{"status": "starting"}\n'
                '{"result": "", "modelUsage": {"claude-haiku-3-5-20241022": {"inputTokens": 1}}}\n'
                '{"result": "", "modelUsage": {"claude-opus-4-6": {"inputTokens": 1}}}\n'
            ),
            stderr="",
        )
        result = resolve_model_id("haiku")
        assert result == "claude-haiku-3-5-20241022"

    @patch("model_resolver.subprocess.run")
    def test_cli_args_constructed_correctly(self, mock_run):
        """Verify the exact CLI arguments used for model resolution."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"modelUsage": {"claude-opus-4-6": {}}}',
            stderr="",
        )
        resolve_model_id("opus", claude_command="claude", timeout=30)
        cmd = mock_run.call_args[0][0]
        assert cmd == [
            "claude", "-p", "x",
            "--model", "opus",
            "--output-format", "json",
            "--max-turns", "1",
            "--tools", "",
        ]
        assert mock_run.call_args[1]["capture_output"] is True
        assert mock_run.call_args[1]["text"] is True
        assert mock_run.call_args[1]["timeout"] == 30

    @patch("model_resolver.subprocess.run")
    def test_json_line_not_starting_with_brace_skipped(self, mock_run):
        """Lines that don't start with '{' after stripping should be skipped."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                '  \n'                              # blank after strip
                'Loading claude...\n'                # not JSON
                '[{"array": true}]\n'                # starts with [
                '{"modelUsage": {"claude-opus-4-6": {"inputTokens": 1}}}\n'
            ),
            stderr="",
        )
        result = resolve_model_id("opus")
        assert result == "claude-opus-4-6"

    @patch("model_resolver.subprocess.run")
    def test_data_not_dict_skipped(self, mock_run):
        """JSON line that parses to a non-dict (e.g., string) should be skipped."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                '"just a string"\n'
                '{"modelUsage": {"claude-opus-4-6": {"inputTokens": 1}}}\n'
            ),
            stderr="",
        )
        result = resolve_model_id("opus")
        assert result == "claude-opus-4-6"
