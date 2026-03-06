"""Provider-agnostic LLM backend: supports Claude CLI, OpenAI, and Gemini APIs."""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from claude_runner import ClaudeResult, ClaudeRunner
from config_schema import Config

logger = logging.getLogger(__name__)

# Default context window sizes for token tracking
_CONTEXT_WINDOWS = {
    "claude": 200000,
    "openai": 128000,
    "gemini": 1000000,
}


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

    def run(self, prompt: str, add_dirs: Optional[List[str]] = None) -> ClaudeResult:
        """Run OpenAI chat completion."""
        if not self._api_key:
            return ClaudeResult(
                success=False,
                error="OpenAI API key not set (check api_key_env config)",
            )

        start = time.time()
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self._base_url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                response_data = json.loads(resp.read().decode("utf-8"))

        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")[:500]
            except Exception:
                pass
            return ClaudeResult(
                success=False,
                error=f"OpenAI API error {e.code}: {body}",
                duration_seconds=time.time() - start,
            )
        except urllib.error.URLError as e:
            return ClaudeResult(
                success=False,
                error=f"OpenAI connection error: {e.reason}",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            return ClaudeResult(
                success=False,
                error=f"OpenAI request failed: {e}",
                duration_seconds=time.time() - start,
            )

        duration = time.time() - start

        # Parse response
        choices = response_data.get("choices", [])
        result_text = ""
        if choices:
            result_text = choices[0].get("message", {}).get("content", "")

        usage = response_data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_tokens = input_tokens + output_tokens
        context_window = _CONTEXT_WINDOWS.get("openai", 128000)
        context_pct = (total_tokens / context_window) * 100 if total_tokens else 0.0

        return ClaudeResult(
            success=True,
            result_text=result_text,
            cost_usd=0.0,  # Cost calculation not available without pricing config
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

    def _build_url(self) -> str:
        return (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent?key={self._api_key}"
        )

    def run(self, prompt: str, add_dirs: Optional[List[str]] = None) -> ClaudeResult:
        """Run Gemini generateContent."""
        if not self._api_key:
            return ClaudeResult(
                success=False,
                error="Gemini API key not set (check api_key_env config)",
            )

        start = time.time()
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self._build_url(),
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                response_data = json.loads(resp.read().decode("utf-8"))

        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")[:500]
            except Exception:
                pass
            return ClaudeResult(
                success=False,
                error=f"Gemini API error {e.code}: {body}",
                duration_seconds=time.time() - start,
            )
        except urllib.error.URLError as e:
            return ClaudeResult(
                success=False,
                error=f"Gemini connection error: {e.reason}",
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            return ClaudeResult(
                success=False,
                error=f"Gemini request failed: {e}",
                duration_seconds=time.time() - start,
            )

        duration = time.time() - start

        # Parse response
        result_text = ""
        candidates = response_data.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts:
                result_text = parts[0].get("text", "")

        usage = response_data.get("usageMetadata", {})
        input_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)
        total_tokens = input_tokens + output_tokens
        context_window = _CONTEXT_WINDOWS.get("gemini", 1000000)
        context_pct = (total_tokens / context_window) * 100 if total_tokens else 0.0

        return ClaudeResult(
            success=True,
            result_text=result_text,
            cost_usd=0.0,
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
