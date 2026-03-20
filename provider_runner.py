"""Provider-agnostic LLM backend: supports Claude CLI, OpenAI, and Gemini APIs."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from typing import List, Optional, Protocol, runtime_checkable

from claude_runner import CircuitBreaker, ClaudeResult, ClaudeRunner
from config_schema import Config

logger = logging.getLogger(__name__)

# HTTP status codes that indicate transient errors worth retrying
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503}

# Patterns for scrubbing API keys from error messages
_API_KEY_PATTERNS = [
    re.compile(r'sk-[A-Za-z0-9]{20,}'),               # OpenAI keys
    re.compile(r'Bearer\s+[A-Za-z0-9\-_.=]{20,}'),     # Bearer tokens
    re.compile(r'AIza[A-Za-z0-9\-_]{30,}'),             # Google API keys
    re.compile(r'key-[A-Za-z0-9]{20,}'),                # Generic key- prefixed
]

# Default context window sizes for token tracking
_CONTEXT_WINDOWS = {
    "claude": 200000,
    "openai": 128000,
    "gemini": 1000000,
}


def _sanitize_error(message: str, api_key: str) -> str:
    """Strip API keys from error messages to prevent leakage in logs."""
    if api_key:
        message = message.replace(api_key, "***")
    for pattern in _API_KEY_PATTERNS:
        message = pattern.sub("***", message)
    return message


def _estimate_cost(config: Config, provider: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD based on token usage and pricing config.

    Without this, cost-limit safety checks (check_cost_limit,
    _cost_limit_exceeded) are completely bypassed for non-Claude providers.
    """
    pricing = config.pricing.cost_per_million_input_tokens
    model = (config.claude.resolved_model or config.claude.model).lower()

    # Map provider models to pricing tiers
    if provider == "openai":
        cost_per_m_input = pricing.get(model, pricing.get("sonnet", 3.0))
    elif provider == "gemini":
        cost_per_m_input = pricing.get(model, pricing.get("haiku", 0.25))
    else:
        cost_per_m_input = pricing.get(model, 0.0)

    if cost_per_m_input <= 0:
        return 0.0

    output_multiplier = config.pricing.output_cost_multiplier
    input_cost = (input_tokens / 1_000_000) * cost_per_m_input
    output_cost = (output_tokens / 1_000_000) * cost_per_m_input * output_multiplier
    return input_cost + output_cost


@runtime_checkable
class ProviderRunner(Protocol):
    """Protocol for LLM provider runners."""

    def run(self, prompt: str, add_dirs: Optional[List[str]] = None) -> ClaudeResult:
        """Run the LLM with the given prompt."""
        ...

    def terminate(self) -> None:
        """Terminate any running subprocess."""
        ...


class OpenAIRunner:
    """Runner for OpenAI chat completions API (stdlib urllib, no external deps)."""

    def __init__(self, config: Config):
        self.config = config
        api_key_env = config.claude.api_key_env or "OPENAI_API_KEY"
        self._api_key = os.environ.get(api_key_env, "")
        self._model = config.claude.resolved_model or config.claude.model
        self._timeout = config.claude.timeout_seconds
        self._base_url = "https://api.openai.com/v1/chat/completions"
        self.circuit_breaker = CircuitBreaker()

    def _sanitize_error(self, message: str) -> str:
        return _sanitize_error(message, self._api_key)

    def run(self, prompt: str, add_dirs: Optional[List[str]] = None) -> ClaudeResult:
        """Run OpenAI chat completion with retry on transient errors."""
        if not self._api_key:
            return ClaudeResult(
                success=False,
                error="OpenAI API key not set (check api_key_env config)",
            )

        if not self.circuit_breaker.allow_request():
            return ClaudeResult(
                success=False,
                error="Circuit breaker is open: too many consecutive API failures. "
                      "Will automatically retry after recovery timeout.",
            )

        if add_dirs:
            dir_context = "Working directories:\n" + "\n".join(f"- {d}" for d in add_dirs)
            prompt = f"{dir_context}\n\n{prompt}"

        start = time.time()
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
        }
        data = json.dumps(payload).encode("utf-8")
        max_attempts = 3
        last_error = ""
        response_data = {}

        for attempt in range(max_attempts):
            try:
                req = urllib.request.Request(
                    self._base_url,
                    data=data,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self._api_key}",
                    },
                )
                _MAX_RESPONSE_BYTES = 10 * 1024 * 1024  # 10 MB
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    response_data = json.loads(resp.read(_MAX_RESPONSE_BYTES).decode("utf-8"))
                break  # Success

            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8")[:500]
                except Exception:
                    pass
                last_error = self._sanitize_error(f"OpenAI API error {e.code}: {body}")
                if e.code in _RETRYABLE_STATUS_CODES and attempt < max_attempts - 1:
                    delay = (2 ** attempt)
                    logger.debug("OpenAI transient error %d, retrying in %ds", e.code, delay)
                    time.sleep(delay)
                    continue
                self.circuit_breaker.record_failure()
                return ClaudeResult(
                    success=False, error=last_error,
                    duration_seconds=time.time() - start,
                )
            except urllib.error.URLError as e:
                last_error = self._sanitize_error(f"OpenAI connection error: {e.reason}")
                if attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)
                    continue
                self.circuit_breaker.record_failure()
                return ClaudeResult(
                    success=False, error=last_error,
                    duration_seconds=time.time() - start,
                )
            except Exception as e:
                self.circuit_breaker.record_failure()
                return ClaudeResult(
                    success=False,
                    error=self._sanitize_error(f"OpenAI request failed: {e}"),
                    duration_seconds=time.time() - start,
                )

        duration = time.time() - start

        # Parse response
        if not response_data:
            self.circuit_breaker.record_failure()
            return ClaudeResult(
                success=False,
                error="OpenAI returned empty response",
                duration_seconds=time.time() - start,
            )

        choices = response_data.get("choices", [])
        result_text = ""
        if choices:
            result_text = choices[0].get("message", {}).get("content", "")

        if not result_text:
            self.circuit_breaker.record_failure()
            return ClaudeResult(
                success=False,
                error="OpenAI returned no content in response",
                duration_seconds=time.time() - start,
                raw_json=response_data,
            )

        self.circuit_breaker.record_success()

        usage = response_data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_tokens = input_tokens + output_tokens
        context_window = _CONTEXT_WINDOWS.get("openai", 128000)
        context_pct = (total_tokens / context_window) * 100 if total_tokens else 0.0

        cost_usd = _estimate_cost(self.config, "openai", input_tokens, output_tokens)

        return ClaudeResult(
            success=True,
            result_text=result_text,
            cost_usd=cost_usd,
            duration_seconds=duration,
            raw_json=response_data,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            context_window_pct=context_pct,
        )

    def terminate(self) -> None:
        """No subprocess to terminate for API-based runner."""
        pass


class GeminiRunner:
    """Runner for Google Gemini generateContent API (stdlib urllib, no external deps)."""

    def __init__(self, config: Config):
        self.config = config
        api_key_env = config.claude.api_key_env or "GEMINI_API_KEY"
        self._api_key = os.environ.get(api_key_env, "")
        self._model = config.claude.resolved_model or config.claude.model
        self._timeout = config.claude.timeout_seconds
        self.circuit_breaker = CircuitBreaker()

    def _build_url(self) -> str:
        return (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent"
        )

    def _sanitize_error(self, message: str) -> str:
        return _sanitize_error(message, self._api_key)

    def run(self, prompt: str, add_dirs: Optional[List[str]] = None) -> ClaudeResult:
        """Run Gemini generateContent with retry on transient errors."""
        if not self._api_key:
            return ClaudeResult(
                success=False,
                error="Gemini API key not set (check api_key_env config)",
            )

        if not self.circuit_breaker.allow_request():
            return ClaudeResult(
                success=False,
                error="Circuit breaker is open: too many consecutive API failures. "
                      "Will automatically retry after recovery timeout.",
            )

        if add_dirs:
            dir_context = "Working directories:\n" + "\n".join(f"- {d}" for d in add_dirs)
            prompt = f"{dir_context}\n\n{prompt}"

        start = time.time()
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
        }
        data = json.dumps(payload).encode("utf-8")
        max_attempts = 3
        last_error = ""
        response_data = {}

        for attempt in range(max_attempts):
            try:
                req = urllib.request.Request(
                    self._build_url(),
                    data=data,
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": self._api_key,
                    },
                )
                _MAX_RESPONSE_BYTES = 10 * 1024 * 1024  # 10 MB
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    response_data = json.loads(resp.read(_MAX_RESPONSE_BYTES).decode("utf-8"))
                break  # Success

            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8")[:500]
                except Exception:
                    pass
                last_error = self._sanitize_error(f"Gemini API error {e.code}: {body}")
                if e.code in _RETRYABLE_STATUS_CODES and attempt < max_attempts - 1:
                    delay = (2 ** attempt)
                    logger.debug("Gemini transient error %d, retrying in %ds", e.code, delay)
                    time.sleep(delay)
                    continue
                self.circuit_breaker.record_failure()
                return ClaudeResult(
                    success=False, error=last_error,
                    duration_seconds=time.time() - start,
                )
            except urllib.error.URLError as e:
                last_error = self._sanitize_error(f"Gemini connection error: {e.reason}")
                if attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)
                    continue
                self.circuit_breaker.record_failure()
                return ClaudeResult(
                    success=False, error=last_error,
                    duration_seconds=time.time() - start,
                )
            except Exception as e:
                self.circuit_breaker.record_failure()
                return ClaudeResult(
                    success=False,
                    error=self._sanitize_error(f"Gemini request failed: {e}"),
                    duration_seconds=time.time() - start,
                )

        duration = time.time() - start

        # Parse response
        if not response_data:
            self.circuit_breaker.record_failure()
            return ClaudeResult(
                success=False,
                error="Gemini returned empty response",
                duration_seconds=time.time() - start,
            )

        result_text = ""
        candidates = response_data.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts:
                result_text = parts[0].get("text", "")

        if not result_text:
            self.circuit_breaker.record_failure()
            return ClaudeResult(
                success=False,
                error="Gemini returned no content in response",
                duration_seconds=time.time() - start,
                raw_json=response_data,
            )

        self.circuit_breaker.record_success()

        usage = response_data.get("usageMetadata", {})
        input_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)
        total_tokens = input_tokens + output_tokens
        context_window = _CONTEXT_WINDOWS.get("gemini", 1000000)
        context_pct = (total_tokens / context_window) * 100 if total_tokens else 0.0

        cost_usd = _estimate_cost(self.config, "gemini", input_tokens, output_tokens)

        return ClaudeResult(
            success=True,
            result_text=result_text,
            cost_usd=cost_usd,
            duration_seconds=duration,
            raw_json=response_data,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            context_window_pct=context_pct,
        )

    def terminate(self) -> None:
        """No subprocess to terminate for API-based runner."""
        pass


def create_runner(config: Config) -> ProviderRunner:
    """Factory: create the appropriate runner based on config.claude.provider."""
    provider = config.claude.provider.lower()

    if provider == "openai":
        return OpenAIRunner(config)
    elif provider == "gemini":
        return GeminiRunner(config)
    else:
        # Default: Claude CLI
        return ClaudeRunner(config)
