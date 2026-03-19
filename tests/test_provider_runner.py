"""Tests for provider_runner.py — provider-agnostic LLM backend."""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

from claude_runner import ClaudeResult, ClaudeRunner
from provider_runner import (
    OpenAIRunner,
    GeminiRunner,
    create_runner,
    ProviderRunner,
)


def _make_config(provider="claude"):
    config = MagicMock()
    config.claude.model = "opus"
    config.claude.resolved_model = ""
    config.claude.max_turns = 25
    config.claude.timeout_seconds = 14400
    config.claude.command = "claude"
    config.claude.max_retries = 0
    config.claude.retry_delays = [2]
    config.claude.rate_limit_base_delay = 5
    config.claude.rate_limit_multiplier = 3
    config.claude.provider = provider
    config.claude.api_key_env = ""
    config.target_dir = "/tmp/test"
    config.paths.state_dir = "/tmp/state"
    return config


class TestCreateRunner(unittest.TestCase):

    def test_claude_provider(self):
        config = _make_config("claude")
        runner = create_runner(config)
        self.assertIsInstance(runner, ClaudeRunner)

    def test_openai_provider(self):
        config = _make_config("openai")
        runner = create_runner(config)
        self.assertIsInstance(runner, OpenAIRunner)

    def test_gemini_provider(self):
        config = _make_config("gemini")
        runner = create_runner(config)
        self.assertIsInstance(runner, GeminiRunner)

    def test_default_provider(self):
        config = _make_config("unknown")
        runner = create_runner(config)
        # Unknown provider falls back to Claude
        self.assertIsInstance(runner, ClaudeRunner)

    def test_case_insensitive(self):
        config = _make_config("OpenAI")
        runner = create_runner(config)
        self.assertIsInstance(runner, OpenAIRunner)


class TestOpenAIRunner(unittest.TestCase):

    def test_missing_api_key(self):
        config = _make_config("openai")
        config.claude.api_key_env = "NONEXISTENT_KEY_12345"
        runner = OpenAIRunner(config)
        result = runner.run("Hello")
        self.assertFalse(result.success)
        self.assertIn("API key", result.error)

    @patch("urllib.request.urlopen")
    def test_successful_call(self, mock_urlopen):
        config = _make_config("openai")
        response_data = {
            "choices": [{"message": {"content": "Hello back"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        runner = OpenAIRunner(config)
        runner._api_key = "test-key"
        result = runner.run("Hello")

        self.assertTrue(result.success)
        self.assertEqual(result.result_text, "Hello back")
        self.assertEqual(result.input_tokens, 10)
        self.assertEqual(result.output_tokens, 20)

    @patch("urllib.request.urlopen")
    def test_http_error(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 429, "rate limited", {}, None,
        )

        config = _make_config("openai")
        runner = OpenAIRunner(config)
        runner._api_key = "test-key"
        result = runner.run("Hello")

        self.assertFalse(result.success)
        self.assertIn("429", result.error)

    def test_terminate(self):
        config = _make_config("openai")
        runner = OpenAIRunner(config)
        # Should not raise
        runner.terminate()


class TestGeminiRunner(unittest.TestCase):

    def test_missing_api_key(self):
        config = _make_config("gemini")
        config.claude.api_key_env = "NONEXISTENT_KEY_12345"
        runner = GeminiRunner(config)
        result = runner.run("Hello")
        self.assertFalse(result.success)
        self.assertIn("API key", result.error)

    @patch("urllib.request.urlopen")
    def test_successful_call(self, mock_urlopen):
        config = _make_config("gemini")
        response_data = {
            "candidates": [{"content": {"parts": [{"text": "Gemini says hi"}]}}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 15},
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        runner = GeminiRunner(config)
        runner._api_key = "test-key"
        result = runner.run("Hello")

        self.assertTrue(result.success)
        self.assertEqual(result.result_text, "Gemini says hi")
        self.assertEqual(result.input_tokens, 5)
        self.assertEqual(result.output_tokens, 15)

    @patch("urllib.request.urlopen")
    def test_network_error(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        config = _make_config("gemini")
        runner = GeminiRunner(config)
        runner._api_key = "test-key"
        result = runner.run("Hello")

        self.assertFalse(result.success)
        self.assertIn("connection", result.error.lower())

    def test_terminate(self):
        config = _make_config("gemini")
        runner = GeminiRunner(config)
        runner.terminate()


class TestProviderProtocol(unittest.TestCase):

    def test_openai_implements_protocol(self):
        self.assertTrue(issubclass(OpenAIRunner, ProviderRunner))

    def test_gemini_implements_protocol(self):
        self.assertTrue(issubclass(GeminiRunner, ProviderRunner))

    def test_claude_implements_protocol(self):
        self.assertTrue(issubclass(ClaudeRunner, ProviderRunner))


class TestGeminiApiKeySanitization(unittest.TestCase):
    """Tests for API key sanitization in Gemini error messages."""

    def test_sanitize_error_strips_api_key(self):
        config = _make_config(provider="gemini")
        runner = GeminiRunner(config)
        runner._api_key = "sk-secret-key-12345"
        msg = runner._sanitize_error(
            "Gemini connection error: https://example.com?key=sk-secret-key-12345"
        )
        self.assertNotIn("sk-secret-key-12345", msg)
        self.assertIn("***", msg)

    def test_sanitize_error_no_key_returns_unchanged(self):
        config = _make_config(provider="gemini")
        runner = GeminiRunner(config)
        runner._api_key = ""
        msg = runner._sanitize_error("Some error message")
        self.assertEqual(msg, "Some error message")

    @patch("provider_runner.urllib.request.urlopen")
    def test_http_error_does_not_leak_api_key(self, mock_urlopen):
        import urllib.error
        config = _make_config(provider="gemini")
        runner = GeminiRunner(config)
        runner._api_key = "my-secret-gemini-key"
        err = urllib.error.HTTPError(
            url=runner._build_url(),
            code=400,
            msg="Bad Request",
            hdrs={},
            fp=None,
        )
        err.read = lambda: b"bad request body"
        mock_urlopen.side_effect = err
        result = runner.run("test prompt")
        self.assertFalse(result.success)
        self.assertNotIn("my-secret-gemini-key", result.error)

    @patch("provider_runner.urllib.request.urlopen")
    def test_url_error_does_not_leak_api_key(self, mock_urlopen):
        import urllib.error
        config = _make_config(provider="gemini")
        runner = GeminiRunner(config)
        runner._api_key = "another-secret-key"
        mock_urlopen.side_effect = urllib.error.URLError(
            reason=f"Failed to connect to url with key=another-secret-key"
        )
        result = runner.run("test prompt")
        self.assertFalse(result.success)
        self.assertNotIn("another-secret-key", result.error)


class TestGeminiApiKeyNotInUrl(unittest.TestCase):
    """Verify API key is sent via header, not URL query parameter."""

    def test_build_url_has_no_key_param(self):
        config = _make_config(provider="gemini")
        runner = GeminiRunner(config)
        runner._api_key = "secret-key-12345"
        url = runner._build_url()
        self.assertNotIn("secret-key-12345", url)
        self.assertNotIn("key=", url)

    @patch("provider_runner.urllib.request.urlopen")
    def test_api_key_sent_in_header(self, mock_urlopen):
        config = _make_config(provider="gemini")
        response_data = {
            "candidates": [{"content": {"parts": [{"text": "hi"}]}}],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        runner = GeminiRunner(config)
        runner._api_key = "test-header-key"
        runner.run("Hello")

        # Verify the request was made with the key in headers, not URL
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        self.assertEqual(req.get_header("X-goog-api-key"), "test-header-key")
        self.assertNotIn("key=", req.full_url)


class TestOpenAIApiKeySanitization(unittest.TestCase):
    """Tests for API key sanitization in OpenAI error messages."""

    def test_sanitize_error_strips_api_key(self):
        config = _make_config(provider="openai")
        runner = OpenAIRunner(config)
        runner._api_key = "sk-secret-openai-key-12345"
        msg = runner._sanitize_error(
            "OpenAI error: key=sk-secret-openai-key-12345 failed"
        )
        self.assertNotIn("sk-secret-openai-key-12345", msg)
        self.assertIn("***", msg)

    def test_sanitize_error_no_key_returns_unchanged(self):
        config = _make_config(provider="openai")
        runner = OpenAIRunner(config)
        runner._api_key = ""
        msg = runner._sanitize_error("Some error message")
        self.assertEqual(msg, "Some error message")

    @patch("provider_runner.urllib.request.urlopen")
    def test_http_error_does_not_leak_api_key(self, mock_urlopen):
        import urllib.error
        config = _make_config(provider="openai")
        runner = OpenAIRunner(config)
        runner._api_key = "sk-secret-openai-key"
        err = urllib.error.HTTPError(
            url="https://api.openai.com/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=None,
        )
        err.read = lambda: b"invalid api key: sk-secret-openai-key"
        mock_urlopen.side_effect = err
        result = runner.run("test prompt")
        self.assertFalse(result.success)
        self.assertNotIn("sk-secret-openai-key", result.error)


class TestSanitizeErrorPatterns(unittest.TestCase):
    def test_openai_key_pattern_scrubbed(self):
        """OpenAI-style API keys should be scrubbed even if not the configured key."""
        config = _make_config()
        runner = OpenAIRunner(config)
        runner._api_key = "different-key"
        msg = "Error: invalid key sk-abc123def456ghi789jkl012mno345pqr678"
        sanitized = runner._sanitize_error(msg)
        assert "sk-abc123" not in sanitized

    def test_bearer_token_scrubbed(self):
        """Bearer tokens in error messages should be redacted."""
        config = _make_config()
        runner = OpenAIRunner(config)
        runner._api_key = ""
        msg = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        sanitized = runner._sanitize_error(msg)
        assert "eyJhbGci" not in sanitized


if __name__ == "__main__":
    unittest.main()
