"""Tests for provider_runner.py — provider-agnostic LLM backend."""

from __future__ import annotations

import json
import os
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
    config.pricing.cost_per_million_input_tokens = {"opus": 15.0, "sonnet": 3.0, "haiku": 0.25}
    config.pricing.output_cost_multiplier = 5.0
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


class TestSanitizeErrorModuleFunction(unittest.TestCase):
    """Tests that the module-level _sanitize_error function works correctly."""

    def test_module_level_function_exists(self):
        """_sanitize_error should be a module-level function reused by both runners."""
        from provider_runner import _sanitize_error
        result = _sanitize_error("no key here", "")
        self.assertEqual(result, "no key here")

    def test_module_level_strips_api_key(self):
        from provider_runner import _sanitize_error
        result = _sanitize_error("error with my-secret-key in it", "my-secret-key")
        self.assertNotIn("my-secret-key", result)
        self.assertIn("***", result)

    def test_module_level_strips_patterns(self):
        from provider_runner import _sanitize_error
        result = _sanitize_error("key sk-abc123def456ghi789jkl012mno345pqr678", "")
        self.assertNotIn("sk-abc123", result)

    def test_both_runners_use_same_function(self):
        """Both OpenAIRunner and GeminiRunner should delegate to the same function."""
        config_openai = _make_config("openai")
        config_gemini = _make_config("gemini")
        runner_openai = OpenAIRunner(config_openai)
        runner_gemini = GeminiRunner(config_gemini)
        runner_openai._api_key = "test-key-123"
        runner_gemini._api_key = "test-key-123"
        msg = "error test-key-123"
        self.assertEqual(
            runner_openai._sanitize_error(msg),
            runner_gemini._sanitize_error(msg),
        )


class TestOpenAIRetryOnTransientError(unittest.TestCase):
    """OpenAIRunner should retry on transient HTTP errors (429, 500, 502, 503)."""

    @patch("provider_runner.urllib.request.urlopen")
    def test_retries_on_429(self, mock_urlopen):
        import urllib.error
        config = _make_config("openai")
        runner = OpenAIRunner(config)
        runner._api_key = "test-key"

        # First call: 429, second call: success
        err = urllib.error.HTTPError("url", 429, "rate limited", {}, None)
        err.read = lambda: b"rate limited"
        response_data = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [err, mock_response]
        result = runner.run("Hello")
        self.assertTrue(result.success)
        self.assertEqual(mock_urlopen.call_count, 2)

    @patch("provider_runner.urllib.request.urlopen")
    def test_no_retry_on_400(self, mock_urlopen):
        """Client errors (400) should not be retried."""
        import urllib.error
        config = _make_config("openai")
        runner = OpenAIRunner(config)
        runner._api_key = "test-key"

        err = urllib.error.HTTPError("url", 400, "bad request", {}, None)
        err.read = lambda: b"bad request"
        mock_urlopen.side_effect = err
        result = runner.run("Hello")
        self.assertFalse(result.success)
        self.assertEqual(mock_urlopen.call_count, 1)


class TestGeminiRetryOnTransientError(unittest.TestCase):
    """GeminiRunner should retry on transient HTTP errors."""

    @patch("provider_runner.urllib.request.urlopen")
    def test_retries_on_503(self, mock_urlopen):
        import urllib.error
        config = _make_config("gemini")
        runner = GeminiRunner(config)
        runner._api_key = "test-key"

        err = urllib.error.HTTPError("url", 503, "service unavailable", {}, None)
        err.read = lambda: b"unavailable"
        response_data = {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [err, mock_response]
        result = runner.run("Hello")
        self.assertTrue(result.success)
        self.assertEqual(mock_urlopen.call_count, 2)

    @patch("provider_runner.urllib.request.urlopen")
    def test_no_retry_on_401(self, mock_urlopen):
        """Client errors (401) should not be retried."""
        import urllib.error
        config = _make_config("gemini")
        runner = GeminiRunner(config)
        runner._api_key = "test-key"

        err = urllib.error.HTTPError("url", 401, "unauthorized", {}, None)
        err.read = lambda: b"unauthorized"
        mock_urlopen.side_effect = err
        result = runner.run("Hello")
        self.assertFalse(result.success)
        self.assertEqual(mock_urlopen.call_count, 1)


class TestResponseDataInitialized(unittest.TestCase):
    """Test that response_data is initialized before the retry loop."""

    def test_openai_response_data_initialized(self):
        """OpenAIRunner.run should initialize response_data before the loop."""
        import inspect
        source = inspect.getsource(OpenAIRunner.run)
        # response_data should be assigned before the for loop
        init_pos = source.find("response_data = {}")
        loop_pos = source.find("for attempt in range")
        self.assertGreater(init_pos, -1, "response_data should be initialized before loop")
        self.assertLess(init_pos, loop_pos, "response_data init should come before for loop")

    def test_gemini_response_data_initialized(self):
        """GeminiRunner.run should initialize response_data before the loop."""
        import inspect
        source = inspect.getsource(GeminiRunner.run)
        init_pos = source.find("response_data = {}")
        loop_pos = source.find("for attempt in range")
        self.assertGreater(init_pos, -1, "response_data should be initialized before loop")
        self.assertLess(init_pos, loop_pos, "response_data init should come before for loop")


class TestEmptyResponseHandling(unittest.TestCase):
    """Test that runners return failure on empty API responses."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test12345678901234567890"})
    def test_openai_empty_response_returns_failure(self):
        """OpenAI runner should return failure when response has no content."""
        config = _make_config("openai")
        config.claude.api_key_env = "OPENAI_API_KEY"
        runner = OpenAIRunner(config)

        empty_response = json.dumps({"choices": []}).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = empty_response
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = runner.run("test prompt")
        self.assertFalse(result.success)
        self.assertIn("no content", result.error)

    @patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaTestKey123456789012345678901"})
    def test_gemini_empty_response_returns_failure(self):
        """Gemini runner should return failure when response has no content."""
        config = _make_config("gemini")
        config.claude.api_key_env = "GEMINI_API_KEY"
        runner = GeminiRunner(config)

        empty_response = json.dumps({"candidates": []}).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = empty_response
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = runner.run("test prompt")
        self.assertFalse(result.success)
        self.assertIn("no content", result.error)


class TestCostEstimation(unittest.TestCase):
    """Test that OpenAI and Gemini runners estimate cost instead of returning 0."""

    @patch("urllib.request.urlopen")
    def test_openai_returns_nonzero_cost(self, mock_urlopen):
        """OpenAI runner should estimate cost based on token usage."""
        config = _make_config("openai")
        response_data = {
            "choices": [{"message": {"content": "response"}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
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
        self.assertGreater(result.cost_usd, 0.0)

    @patch("urllib.request.urlopen")
    def test_gemini_returns_nonzero_cost(self, mock_urlopen):
        """Gemini runner should estimate cost based on token usage."""
        config = _make_config("gemini")
        response_data = {
            "candidates": [{"content": {"parts": [{"text": "response"}]}}],
            "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 500},
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
        self.assertGreater(result.cost_usd, 0.0)

    def test_estimate_cost_function(self):
        """_estimate_cost should compute cost from tokens and pricing config."""
        from provider_runner import _estimate_cost
        config = _make_config("openai")
        # Config model is "opus" at 15.0 per million input tokens
        cost = _estimate_cost(config, "openai", 1_000_000, 0)
        self.assertAlmostEqual(cost, 15.0, places=2)

    def test_estimate_cost_with_output(self):
        """_estimate_cost should account for output tokens at multiplied rate."""
        from provider_runner import _estimate_cost
        config = _make_config("openai")
        # 1M input + 1M output at 15.0/M input (opus), 5x output multiplier
        cost = _estimate_cost(config, "openai", 1_000_000, 1_000_000)
        self.assertAlmostEqual(cost, 15.0 + 75.0, places=2)

    def test_estimate_cost_zero_tokens(self):
        """_estimate_cost with zero tokens should return 0."""
        from provider_runner import _estimate_cost
        config = _make_config("openai")
        cost = _estimate_cost(config, "openai", 0, 0)
        self.assertEqual(cost, 0.0)


class TestAddDirsIncludedInPrompt(unittest.TestCase):
    """Regression: add_dirs must be included in the prompt for non-Claude providers."""

    def test_openai_add_dirs_prepended_to_prompt(self):
        """OpenAI runner should prepend add_dirs to the prompt text."""
        config = _make_config("openai")
        runner = OpenAIRunner(config)
        runner._api_key = "test-key"

        captured_payloads = []
        original_prompt = "Fix the bug in main.py"

        def mock_urlopen(req, **kwargs):
            body = json.loads(req.data.decode("utf-8"))
            captured_payloads.append(body)
            resp = MagicMock()
            resp.read.return_value = json.dumps({
                "choices": [{"message": {"content": "done"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }).encode()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("provider_runner.urllib.request.urlopen", side_effect=mock_urlopen):
            runner.run(original_prompt, add_dirs=["/tmp/worktree-0"])

        self.assertEqual(len(captured_payloads), 1)
        sent_content = captured_payloads[0]["messages"][0]["content"]
        self.assertIn("/tmp/worktree-0", sent_content)
        self.assertIn(original_prompt, sent_content)

    def test_gemini_add_dirs_prepended_to_prompt(self):
        """Gemini runner should prepend add_dirs to the prompt text."""
        config = _make_config("gemini")
        runner = GeminiRunner(config)
        runner._api_key = "test-key"

        captured_payloads = []
        original_prompt = "Fix the bug"

        def mock_urlopen(req, **kwargs):
            body = json.loads(req.data.decode("utf-8"))
            captured_payloads.append(body)
            resp = MagicMock()
            resp.read.return_value = json.dumps({
                "candidates": [{"content": {"parts": [{"text": "done"}]}}],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
            }).encode()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("provider_runner.urllib.request.urlopen", side_effect=mock_urlopen):
            runner.run(original_prompt, add_dirs=["/workspace/repo"])

        self.assertEqual(len(captured_payloads), 1)
        sent_text = captured_payloads[0]["contents"][0]["parts"][0]["text"]
        self.assertIn("/workspace/repo", sent_text)
        self.assertIn(original_prompt, sent_text)

    def test_openai_no_add_dirs_prompt_unchanged(self):
        """When add_dirs is None, prompt should not be modified."""
        config = _make_config("openai")
        runner = OpenAIRunner(config)
        runner._api_key = "test-key"

        captured_payloads = []

        def mock_urlopen(req, **kwargs):
            body = json.loads(req.data.decode("utf-8"))
            captured_payloads.append(body)
            resp = MagicMock()
            resp.read.return_value = json.dumps({
                "choices": [{"message": {"content": "done"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }).encode()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("provider_runner.urllib.request.urlopen", side_effect=mock_urlopen):
            runner.run("Fix bug", add_dirs=None)

        sent_content = captured_payloads[0]["messages"][0]["content"]
        self.assertEqual(sent_content, "Fix bug")


class TestProviderRunnerCircuitBreaker(unittest.TestCase):
    """Test that non-Claude runners have circuit breaker protection."""

    def test_openai_runner_has_circuit_breaker(self):
        """Verify OpenAI runner has circuit breaker protection."""
        config = _make_config("openai")
        runner = OpenAIRunner(config)
        self.assertTrue(hasattr(runner, 'circuit_breaker'))

    def test_gemini_runner_has_circuit_breaker(self):
        """Verify Gemini runner has circuit breaker protection."""
        config = _make_config("gemini")
        runner = GeminiRunner(config)
        self.assertTrue(hasattr(runner, 'circuit_breaker'))

    def test_openai_circuit_breaker_blocks_after_failures(self):
        """OpenAI runner circuit breaker opens after consecutive failures."""
        config = _make_config("openai")
        runner = OpenAIRunner(config)
        for _ in range(runner.circuit_breaker.failure_threshold):
            runner.circuit_breaker.record_failure()
        self.assertFalse(runner.circuit_breaker.allow_request())

    def test_gemini_circuit_breaker_blocks_after_failures(self):
        """Gemini runner circuit breaker opens after consecutive failures."""
        config = _make_config("gemini")
        runner = GeminiRunner(config)
        for _ in range(runner.circuit_breaker.failure_threshold):
            runner.circuit_breaker.record_failure()
        self.assertFalse(runner.circuit_breaker.allow_request())


class TestBoundedResponseRead(unittest.TestCase):
    """HTTP response reads must pass a size limit to prevent OOM."""

    @patch("urllib.request.urlopen")
    def test_openai_read_passes_size_limit(self, mock_urlopen):
        """OpenAI runner should call resp.read() with a max size argument."""
        config = _make_config("openai")
        response_data = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        runner = OpenAIRunner(config)
        runner._api_key = "test-key"
        runner.run("Hello")

        # Verify read() was called with a size limit (10 MB)
        mock_response.read.assert_called_once_with(10 * 1024 * 1024)

    @patch("urllib.request.urlopen")
    def test_gemini_read_passes_size_limit(self, mock_urlopen):
        """Gemini runner should call resp.read() with a max size argument."""
        config = _make_config("gemini")
        response_data = {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        runner = GeminiRunner(config)
        runner._api_key = "test-key"
        runner.run("Hello")

        # Verify read() was called with a size limit (10 MB)
        mock_response.read.assert_called_once_with(10 * 1024 * 1024)


class TestEstimateCostProviderDefaults(unittest.TestCase):
    """_estimate_cost should use provider-specific defaults for known models."""

    def test_openai_gpt4o_uses_correct_pricing(self):
        from provider_runner import _estimate_cost
        config = _make_config("openai")
        config.claude.model = "gpt-4o"
        config.claude.resolved_model = "gpt-4o"
        cost = _estimate_cost(config, "openai", 1_000_000, 0)
        self.assertAlmostEqual(cost, 2.5, places=2)

    def test_gemini_flash_uses_correct_pricing(self):
        from provider_runner import _estimate_cost
        config = _make_config("gemini")
        config.claude.model = "gemini-2.0-flash"
        config.claude.resolved_model = "gemini-2.0-flash"
        cost = _estimate_cost(config, "gemini", 1_000_000, 0)
        self.assertAlmostEqual(cost, 0.075, places=3)

    def test_unknown_model_returns_zero_with_warning(self):
        from provider_runner import _estimate_cost
        config = _make_config("openai")
        config.claude.model = "unknown-model-xyz"
        config.claude.resolved_model = "unknown-model-xyz"
        import logging
        with self.assertLogs("provider_runner", level="WARNING") as cm:
            cost = _estimate_cost(config, "openai", 1_000_000, 0)
        self.assertEqual(cost, 0.0)
        self.assertTrue(any("No pricing found" in msg for msg in cm.output))

    def test_user_config_overrides_provider_default(self):
        from provider_runner import _estimate_cost
        config = _make_config("openai")
        config.claude.model = "gpt-4o"
        config.claude.resolved_model = "gpt-4o"
        # User sets custom pricing for gpt-4o
        config.pricing.cost_per_million_input_tokens["gpt-4o"] = 99.0
        cost = _estimate_cost(config, "openai", 1_000_000, 0)
        self.assertAlmostEqual(cost, 99.0, places=2)


if __name__ == "__main__":
    unittest.main()
