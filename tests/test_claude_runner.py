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
        with pytest.raises(json.JSONDecodeError):
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
        with pytest.raises(ValueError, match="No complete JSON"):
            runner._parse_json_response(stdout)


class TestRun:
    @patch("claude_runner.subprocess.run")
    def test_successful_run(self, mock_run, runner):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "Fixed bug", "cost_usd": 0.03, "duration_seconds": 12.5}',
            stderr="",
        )
        result = runner.run("Fix the bug")
        assert result.success is True
        assert result.result_text == "Fixed bug"
        assert result.cost_usd == 0.03
        assert result.duration_seconds == 12.5

    @patch("claude_runner.subprocess.run")
    def test_timeout(self, mock_run, runner):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=300)
        result = runner.run("Fix the bug")
        assert result.success is False
        assert "timed out" in result.error

    @patch("claude_runner.subprocess.run")
    def test_command_not_found(self, mock_run, runner):
        mock_run.side_effect = FileNotFoundError()
        result = runner.run("Fix the bug")
        assert result.success is False
        assert "not found" in result.error

    @patch("claude_runner.subprocess.run")
    def test_nonzero_exit(self, mock_run, runner):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error occurred",
        )
        result = runner.run("Fix the bug")
        assert result.success is False
        assert "exited with code 1" in result.error

    @patch("claude_runner.subprocess.run")
    def test_unparseable_output(self, mock_run, runner):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Not JSON at all",
            stderr="",
        )
        result = runner.run("Fix the bug")
        assert result.success is False
        assert "parse" in result.error.lower()
