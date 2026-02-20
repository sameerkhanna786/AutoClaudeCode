"""Webhook notifications for critical events (Slack, Discord, generic HTTP)."""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WebhookConfig:
    """Configuration for a single webhook endpoint."""
    url: str = ""
    type: str = "generic"  # "slack", "discord", "generic"
    name: str = ""


@dataclass
class NotificationEventsConfig:
    """Which events trigger notifications."""
    on_cycle_success: bool = True
    on_cycle_failure: bool = True
    on_consecutive_failure_threshold: bool = True
    on_cost_limit_exceeded: bool = True
    on_safety_error: bool = True


@dataclass
class NotificationsConfig:
    """Top-level notification configuration."""
    enabled: bool = False
    webhooks: List[WebhookConfig] = field(default_factory=list)
    events: NotificationEventsConfig = field(default_factory=NotificationEventsConfig)


# Map event names to NotificationEventsConfig field names
_EVENT_FIELD_MAP = {
    "cycle_success": "on_cycle_success",
    "cycle_failure": "on_cycle_failure",
    "consecutive_failure_threshold": "on_consecutive_failure_threshold",
    "cost_limit_exceeded": "on_cost_limit_exceeded",
    "safety_error": "on_safety_error",
}


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

        # Rate-limit: deduplicate identical events within the window
        dedup_key = f"{event}:{json.dumps(details, sort_keys=True, default=str)}"
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

        # Send to all webhooks in background threads
        for webhook in self._config.webhooks:
            if not webhook.url:
                continue
            thread = threading.Thread(
                target=self._send_webhook,
                args=(webhook, event, details),
                daemon=True,
            )
            thread.start()

    def _send_webhook(
        self, webhook: WebhookConfig, event: str, details: Dict[str, Any],
    ) -> None:
        """Send a notification to a single webhook endpoint."""
        try:
            if webhook.type == "slack":
                payload = self._format_slack_payload(event, details)
            elif webhook.type == "discord":
                payload = self._format_discord_payload(event, details)
            else:
                payload = self._format_generic_payload(event, details)

            data = json.dumps(payload).encode("utf-8")
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
        except (urllib.error.URLError, OSError, ValueError) as e:
            logger.warning(
                "Failed to send notification to %s: %s",
                webhook.name or webhook.url[:40], e,
            )
        except Exception:
            logger.exception(
                "Unexpected error sending notification to %s",
                webhook.name or webhook.url[:40],
            )

    @staticmethod
    def _format_slack_payload(event: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """Format a Slack-compatible webhook payload."""
        title = f"Auto Claude Code: {event.replace('_', ' ').title()}"
        lines = [f"*{title}*"]
        for key, value in details.items():
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            lines.append(f"• {key}: {value}")
        return {"text": "\n".join(lines)}

    @staticmethod
    def _format_discord_payload(event: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """Format a Discord-compatible webhook payload."""
        title = f"**Auto Claude Code: {event.replace('_', ' ').title()}**"
        lines = [title]
        for key, value in details.items():
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            lines.append(f"• {key}: {value}")
        return {"content": "\n".join(lines)}

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
