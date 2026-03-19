"""Tests for model_resolver module."""

import json
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from model_resolver import resolve_model_id, _read_cache, _write_cache, _CACHE_TTL_SECONDS


# All tests patch _read_cache to return None (bypassing disk cache) so that
# the subprocess.run mock is always exercised.
@patch("model_resolver._read_cache", return_value=None)
@patch("model_resolver._write_cache")
class TestResolveModelId:
    @patch("model_resolver.subprocess.run")
    def test_successful_resolution(self, mock_run, _wc, _rc):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "", "modelUsage": {"claude-opus-4-6": {"inputTokens": 5}}}',
            stderr="",
        )
        result = resolve_model_id("opus")
        assert result == "claude-opus-4-6"

    @patch("model_resolver.subprocess.run")
    def test_timeout_returns_none(self, mock_run, _wc, _rc):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=30)
        result = resolve_model_id("opus")
        assert result is None

    @patch("model_resolver.subprocess.run")
    def test_cli_not_found_returns_none(self, mock_run, _wc, _rc):
        mock_run.side_effect = FileNotFoundError()
        result = resolve_model_id("opus")
        assert result is None

    @patch("model_resolver.subprocess.run")
    def test_nonzero_exit_returns_none(self, mock_run, _wc, _rc):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error: unknown model",
        )
        result = resolve_model_id("opus")
        assert result is None

    @patch("model_resolver.subprocess.run")
    def test_invalid_json_returns_none(self, mock_run, _wc, _rc):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Not JSON at all\nJust text\n",
            stderr="",
        )
        result = resolve_model_id("opus")
        assert result is None

    @patch("model_resolver.subprocess.run")
    def test_missing_model_usage_returns_none(self, mock_run, _wc, _rc):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "done", "cost_usd": 0.01}',
            stderr="",
        )
        result = resolve_model_id("opus")
        assert result is None

    @patch("model_resolver.subprocess.run")
    def test_empty_model_usage_returns_none(self, mock_run, _wc, _rc):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "done", "modelUsage": {}}',
            stderr="",
        )
        result = resolve_model_id("opus")
        assert result is None

    @patch("model_resolver.subprocess.run")
    def test_banner_lines_before_json(self, mock_run, _wc, _rc):
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
    def test_os_error_returns_none(self, mock_run, _wc, _rc):
        mock_run.side_effect = OSError("Connection refused")
        result = resolve_model_id("opus")
        assert result is None

    @patch("model_resolver.subprocess.run")
    def test_custom_command_and_timeout(self, mock_run, _wc, _rc):
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
    def test_model_usage_not_dict_returns_none(self, mock_run, _wc, _rc):
        """modelUsage present but not a dict should return None."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "", "modelUsage": "not-a-dict"}',
            stderr="",
        )
        result = resolve_model_id("opus")
        assert result is None

    @patch("model_resolver.subprocess.run")
    def test_multiple_json_lines_picks_first_with_model_usage(self, mock_run, _wc, _rc):
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
    def test_cli_args_constructed_correctly(self, mock_run, _wc, _rc):
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
    def test_json_line_not_starting_with_brace_skipped(self, mock_run, _wc, _rc):
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
    def test_data_not_dict_skipped(self, mock_run, _wc, _rc):
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


class TestWriteCacheAtomicWrite:
    """Tests that _write_cache uses atomic tempfile + os.replace pattern."""

    def test_write_cache_creates_valid_json(self, tmp_path):
        cache_path = str(tmp_path / "model_cache.json")
        _write_cache("opus", "claude-opus-4-6", cache_path=cache_path)

        data = json.loads(Path(cache_path).read_text())
        assert "entries" in data
        assert data["entries"]["opus"]["resolved"] == "claude-opus-4-6"
        assert "timestamp" in data["entries"]["opus"]

    def test_write_cache_preserves_existing_entries(self, tmp_path):
        cache_path = str(tmp_path / "model_cache.json")
        _write_cache("opus", "claude-opus-4-6", cache_path=cache_path)
        _write_cache("sonnet", "claude-sonnet-4-20250514", cache_path=cache_path)

        data = json.loads(Path(cache_path).read_text())
        assert data["entries"]["opus"]["resolved"] == "claude-opus-4-6"
        assert data["entries"]["sonnet"]["resolved"] == "claude-sonnet-4-20250514"

    def test_write_cache_no_temp_files_left(self, tmp_path):
        cache_path = str(tmp_path / "model_cache.json")
        _write_cache("opus", "claude-opus-4-6", cache_path=cache_path)

        # No .tmp files should remain after a successful write
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []

    def test_write_cache_atomic_replaces_existing(self, tmp_path):
        cache_path = str(tmp_path / "model_cache.json")
        _write_cache("opus", "claude-opus-4-6", cache_path=cache_path)
        _write_cache("opus", "claude-opus-4-20260101", cache_path=cache_path)

        data = json.loads(Path(cache_path).read_text())
        assert data["entries"]["opus"]["resolved"] == "claude-opus-4-20260101"


class TestCacheEncodingUtf8:
    """Tests that _read_cache and _write_cache use utf-8 encoding for read_text/write."""

    def test_read_cache_handles_non_ascii(self, tmp_path):
        """_read_cache should handle non-ASCII model names when read_text uses utf-8."""
        cache_path = str(tmp_path / "model_cache.json")
        data = {
            "entries": {
                "modèle": {
                    "resolved": "claude-résolvé-42",
                    "timestamp": time.time(),
                }
            }
        }
        Path(cache_path).write_text(json.dumps(data), encoding="utf-8")
        result = _read_cache("modèle", cache_path=cache_path)
        assert result == "claude-résolvé-42"

    def test_write_cache_non_ascii_roundtrip(self, tmp_path):
        """_write_cache should write non-ASCII content that _read_cache can read back."""
        cache_path = str(tmp_path / "model_cache.json")
        _write_cache("日本語モデル", "claude-日本語-v1", cache_path=cache_path)
        result = _read_cache("日本語モデル", cache_path=cache_path)
        assert result == "claude-日本語-v1"

    def test_write_cache_produces_utf8_file(self, tmp_path):
        """Written cache file should be valid UTF-8."""
        cache_path = str(tmp_path / "model_cache.json")
        _write_cache("émoji", "claude-🎉-v1", cache_path=cache_path)
        content = Path(cache_path).read_text(encoding="utf-8")
        data = json.loads(content)
        assert data["entries"]["émoji"]["resolved"] == "claude-🎉-v1"
