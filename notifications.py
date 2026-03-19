"""Webhook notifications for critical events (Slack, Discord, generic HTTP)."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from config_schema import WebhookConfig, NotificationEventsConfig, NotificationsConfig

logger = logging.getLogger(__name__)


# Map event names to NotificationEventsConfig field names
_EVENT_FIELD_MAP = {
    "cycle_success": "on_cycle_success",
    "cycle_failure": "on_cycle_failure",
    "consecutive_failure_threshold": "on_consecutive_failure_threshold",
    "cost_limit_exceeded": "on_cost_limit_exceeded",
    "safety_error": "on_safety_error",
    "periodic_summary": "on_periodic_summary",
}


class NaturalLanguageSummarizer:
    """Generates natural language summaries of completed work."""

    _TEMPLATES = {
        "test_failure": "I fixed a failing test: {desc}",
        "lint": "I resolved a lint issue: {desc}",
        "todo": "I addressed a TODO: {desc}",
        "coverage": "I added test coverage: {desc}",
        "quality": "I improved code quality: {desc}",
        "feedback": "I completed a developer request: {desc}",
        "claude_idea": "I implemented an improvement: {desc}",
    }

    _FAILURE_TEMPLATES = {
        "test_failure": "I tried to fix a failing test but encountered issues: {desc}",
        "lint": "I tried to resolve a lint issue but encountered issues: {desc}",
        "todo": "I tried to address a TODO but encountered issues: {desc}",
        "coverage": "I tried to add test coverage but encountered issues: {desc}",
        "quality": "I tried to improve code quality but encountered issues: {desc}",
        "feedback": "I tried to complete a developer request but encountered issues: {desc}",
        "claude_idea": "I tried to implement an improvement but encountered issues: {desc}",
    }

    def summarize(self, tasks, success: bool, cost_usd: float = 0.0) -> str:
        """Generate a natural language summary. Uses templates for single tasks, batch summary for multi."""
        if not tasks:
            return "No tasks were processed."
        if len(tasks) == 1:
            task = tasks[0]
            source = task.get("source", "") if isinstance(task, dict) else getattr(task, "source", "unknown")
            desc = task.get("description", "") if isinstance(task, dict) else getattr(task, "description", "")
            if success:
                template = self._TEMPLATES.get(source, "I worked on: {desc}")
            else:
                template = self._FAILURE_TEMPLATES.get(
                    source, "I worked on a task but encountered issues: {desc}",
                )
            return template.format(desc=desc)

        source_counts: Dict[str, int] = {}
        for t in tasks:
            source = t.get("source", "") if isinstance(t, dict) else getattr(t, "source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1

        parts: List[str] = []
        _LABELS = {"test_failure": "fixed {n} test failure(s)", "lint": "resolved {n} lint issue(s)",
                    "todo": "addressed {n} TODO(s)", "coverage": "added test coverage for {n} module(s)",
                    "quality": "improved {n} module(s)", "feedback": "completed {n} developer request(s)"}
        for source, count in source_counts.items():
            label = _LABELS.get(source, "made {n} improvement(s)")
            parts.append(label.format(n=count))

        if success:
            summary = "I completed a batch: " + ", ".join(parts)
        else:
            summary = "I attempted a batch but encountered issues: " + ", ".join(parts)
        if cost_usd > 0:
            summary += f" (cost: ${cost_usd:.4f})"
        return summary


class NotificationManager:
    """Sends webhook notifications for critical orchestrator events.

    All sends run in background threads to avoid blocking the orchestrator.
    Failures are logged but never propagated — notification errors must not
    crash the main loop.

    Includes simple rate-limiting: identical (event, details) pairs within
    a 60-second window are deduplicated.
    """

    RATE_LIMIT_SECONDS = 60

    def __init__(self, config: NotificationsConfig):
        self._config = config
        self._recent: Dict[str, float] = {}  # dedup key -> timestamp
        self._lock = threading.Lock()
        self._webhook_pool = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="webhook",
        )

    def notify(self, event: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Send a notification for the given event to all configured webhooks.

        Args:
            event: Event name (e.g. "cycle_success", "safety_error").
            details: Additional context about the event.
        """
        if not self._config.enabled:
            return

        if not self._config.webhooks:
            return

        # Check if this event type is enabled
        event_field = _EVENT_FIELD_MAP.get(event)
        if event_field and not getattr(self._config.events, event_field, True):
            return

        details = details or {}

        # Rate-limit: deduplicate identical events within the window.
        # Hash the details to bound key size — large payloads would otherwise
        # create arbitrarily long dict keys consuming excessive memory.
        details_hash = hashlib.md5(
            json.dumps(details, sort_keys=True, default=str).encode()
        ).hexdigest()
        dedup_key = f"{event}:{details_hash}"
        now = time.time()
        with self._lock:
            last_sent = self._recent.get(dedup_key, 0)
            if now - last_sent < self.RATE_LIMIT_SECONDS:
                logger.debug("Rate-limited notification for event=%s", event)
                return
            self._recent[dedup_key] = now

            # Clean up old entries
            cutoff = now - self.RATE_LIMIT_SECONDS * 2
            self._recent = {k: v for k, v in self._recent.items() if v > cutoff}

        # Send to all webhooks via a bounded thread pool to prevent
        # unbounded thread creation under rapid notification bursts.
        for webhook in self._config.webhooks:
            if not webhook.url:
                continue
            self._webhook_pool.submit(self._send_webhook, webhook, event, details)

    def _send_webhook(
        self, webhook: WebhookConfig, event: str, details: Dict[str, Any],
    ) -> None:
        """Send a notification to a single webhook endpoint.

        Retries up to 3 times with exponential backoff (1s, 2s, 4s) on
        transient network errors. Since sends already run in daemon threads,
        the delay doesn't block the orchestrator.
        """
        max_attempts = 3
        base_delay = 1.0

        if webhook.type == "slack":
            payload = self._format_slack_payload(event, details)
        elif webhook.type == "discord":
            payload = self._format_discord_payload(event, details)
        else:
            payload = self._format_generic_payload(event, details)

        data = json.dumps(payload).encode("utf-8")

        for attempt in range(max_attempts):
            try:
                req = urllib.request.Request(
                    webhook.url,
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    resp.read()  # consume response

                logger.debug(
                    "Notification sent: event=%s webhook=%s",
                    event, webhook.name or webhook.url[:40],
                )
                return  # Success
            except (urllib.error.URLError, OSError, ValueError) as e:
                if attempt < max_attempts - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.debug(
                        "Webhook send failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, max_attempts, delay, e,
                    )
                    time.sleep(delay)
                else:
                    logger.warning(
                        "Failed to send notification to %s after %d attempts: %s",
                        webhook.name or webhook.url[:40], max_attempts, e,
                    )
            except Exception:
                logger.exception(
                    "Unexpected error sending notification to %s",
                    webhook.name or webhook.url[:40],
                )
                return  # Don't retry unexpected errors

    def _format_slack_payload(self, event: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """Format a Slack-compatible webhook payload."""
        title = f"Auto Claude Code: {event.replace('_', ' ').title()}"
        lines = [f"*{title}*"]
        # Prepend NL summary if enabled and event is cycle_success or cycle_failure
        if self._config.nl_summaries and event in ("cycle_success", "cycle_failure"):
            tasks = details.get("tasks", [])
            # Build lightweight task dicts for the summarizer
            task_objs = [{"source": "unknown", "description": t} if isinstance(t, str) else t for t in tasks]
            success = event == "cycle_success"
            cost = details.get("cost_usd", 0.0)
            nl_summary = NaturalLanguageSummarizer().summarize(task_objs, success=success, cost_usd=cost)
            lines.append(f"_{nl_summary}_")
        for key, value in details.items():
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            lines.append(f"\u2022 {key}: {value}")
        return {"text": "\n".join(lines)}

    def _format_discord_payload(self, event: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """Format a Discord-compatible webhook payload."""
        title = f"**Auto Claude Code: {event.replace('_', ' ').title()}**"
        lines = [title]
        # Prepend NL summary if enabled and event is cycle_success or cycle_failure
        if self._config.nl_summaries and event in ("cycle_success", "cycle_failure"):
            tasks = details.get("tasks", [])
            task_objs = [{"source": "unknown", "description": t} if isinstance(t, str) else t for t in tasks]
            success = event == "cycle_success"
            cost = details.get("cost_usd", 0.0)
            nl_summary = NaturalLanguageSummarizer().summarize(task_objs, success=success, cost_usd=cost)
            lines.append(f"*{nl_summary}*")
        for key, value in details.items():
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            lines.append(f"\u2022 {key}: {value}")
        return {"content": "\n".join(lines)}

    def shutdown(self) -> None:
        """Shut down the webhook thread pool, waiting for pending sends."""
        self._webhook_pool.shutdown(wait=True)

    @staticmethod
    def _format_generic_payload(
        event: str, details: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Format a generic JSON webhook payload."""
        return {
            "event": event,
            "source": "auto_claude_code",
            "details": details,
            "timestamp": time.time(),
        }
