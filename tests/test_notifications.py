"""Tests for notifications.py — webhook notification system."""

from __future__ import annotations

import json
import time
import unittest
from unittest.mock import MagicMock, patch, call

from notifications import (
    NotificationEventsConfig,
    NotificationManager,
    NotificationsConfig,
    WebhookConfig,
)


def _make_config(enabled=True, webhooks=None, events=None):
    """Helper to create a NotificationsConfig."""
    if webhooks is None:
        webhooks = [WebhookConfig(url="https://hooks.example.com/test", type="generic", name="test")]
    if events is None:
        events = NotificationEventsConfig()
    return NotificationsConfig(enabled=enabled, webhooks=webhooks, events=events)


class TestNotificationManagerDisabled(unittest.TestCase):

    @patch("notifications.urllib.request.urlopen")
    def test_disabled_does_nothing(self, mock_urlopen):
        config = _make_config(enabled=False)
        mgr = NotificationManager(config)
        mgr.notify("cycle_success", {"tasks": ["test"]})
        mock_urlopen.assert_not_called()


class TestNotifySlackFormat(unittest.TestCase):

    @patch("notifications.urllib.request.urlopen")
    def test_slack_payload(self, mock_urlopen):
        mock_urlopen.return_value.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"")))
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        webhook = WebhookConfig(url="https://hooks.slack.com/test", type="slack", name="slack-test")
        config = _make_config(webhooks=[webhook])
        mgr = NotificationManager(config)
        mgr.notify("cycle_success", {"tasks": ["fix tests"]})

        # Wait for background thread
        time.sleep(0.2)

        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode())
        self.assertIn("text", payload)
        self.assertIn("Cycle Success", payload["text"])


class TestNotifyDiscordFormat(unittest.TestCase):

    @patch("notifications.urllib.request.urlopen")
    def test_discord_payload(self, mock_urlopen):
        mock_urlopen.return_value.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"")))
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        webhook = WebhookConfig(url="https://discord.com/api/webhooks/test", type="discord")
        config = _make_config(webhooks=[webhook])
        mgr = NotificationManager(config)
        mgr.notify("cycle_failure", {"error": "tests failed"})

        time.sleep(0.2)

        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode())
        self.assertIn("content", payload)
        self.assertIn("Cycle Failure", payload["content"])


class TestNotifyGenericFormat(unittest.TestCase):

    @patch("notifications.urllib.request.urlopen")
    def test_generic_payload(self, mock_urlopen):
        mock_urlopen.return_value.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"")))
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        config = _make_config()
        mgr = NotificationManager(config)
        mgr.notify("safety_error", {"error": "disk full"})

        time.sleep(0.2)

        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode())
        self.assertEqual(payload["event"], "safety_error")
        self.assertEqual(payload["source"], "auto_claude_code")
        self.assertIn("details", payload)


class TestNotifyHandlesHTTPError(unittest.TestCase):

    @patch("notifications.urllib.request.urlopen")
    def test_error_does_not_propagate(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Connection refused")

        config = _make_config()
        mgr = NotificationManager(config)
        # Should not raise
        mgr.notify("cycle_failure", {"error": "failed"})

        time.sleep(0.2)
        # Verify the call was attempted
        mock_urlopen.assert_called_once()


class TestNotifyRespectsEventConfig(unittest.TestCase):

    @patch("notifications.urllib.request.urlopen")
    def test_disabled_event_not_sent(self, mock_urlopen):
        events = NotificationEventsConfig(on_cycle_success=False)
        config = _make_config(events=events)
        mgr = NotificationManager(config)
        mgr.notify("cycle_success", {"tasks": ["test"]})

        time.sleep(0.2)
        mock_urlopen.assert_not_called()


class TestNotifyRateLimiting(unittest.TestCase):

    @patch("notifications.urllib.request.urlopen")
    def test_duplicate_event_rate_limited(self, mock_urlopen):
        mock_urlopen.return_value.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"")))
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        config = _make_config()
        mgr = NotificationManager(config)

        # Send the same event twice
        mgr.notify("cycle_success", {"tasks": ["test"]})
        mgr.notify("cycle_success", {"tasks": ["test"]})

        time.sleep(0.2)
        # Only one call should have been made
        self.assertEqual(mock_urlopen.call_count, 1)


class TestWebhookConfigValidation(unittest.TestCase):

    @patch("notifications.urllib.request.urlopen")
    def test_empty_url_skipped(self, mock_urlopen):
        webhooks = [
            WebhookConfig(url="", type="generic"),
            WebhookConfig(url="https://valid.example.com", type="generic"),
        ]
        config = _make_config(webhooks=webhooks)
        mgr = NotificationManager(config)
        mgr.notify("safety_error", {"error": "test"})

        time.sleep(0.2)
        # Only one call should be made (the valid webhook)
        self.assertEqual(mock_urlopen.call_count, 1)


from notifications import NaturalLanguageSummarizer


class TestNaturalLanguageSummarizerFailure(unittest.TestCase):
    """Tests that failure summaries don't produce broken grammar like double 'I'."""

    def test_single_task_failure_no_double_i(self):
        summarizer = NaturalLanguageSummarizer()
        tasks = [{"source": "test_failure", "description": "test_foo fails"}]
        result = summarizer.summarize(tasks, success=False)
        # Should NOT contain "I ... I" (double I from template nesting)
        self.assertNotIn("I but encountered issues I", result)
        self.assertIn("encountered issues", result)
        self.assertIn("test_foo fails", result)

    def test_single_task_success_uses_template(self):
        summarizer = NaturalLanguageSummarizer()
        tasks = [{"source": "lint", "description": "unused import"}]
        result = summarizer.summarize(tasks, success=True)
        self.assertEqual(result, "I resolved a lint issue: unused import")

    def test_batch_failure_no_double_i(self):
        summarizer = NaturalLanguageSummarizer()
        tasks = [
            {"source": "test_failure", "description": "t1"},
            {"source": "lint", "description": "t2"},
        ]
        result = summarizer.summarize(tasks, success=False)
        self.assertNotIn("I but encountered issues", result)
        self.assertIn("attempted a batch but encountered issues", result)

    def test_batch_success_message(self):
        summarizer = NaturalLanguageSummarizer()
        tasks = [
            {"source": "test_failure", "description": "t1"},
            {"source": "lint", "description": "t2"},
        ]
        result = summarizer.summarize(tasks, success=True)
        self.assertIn("I completed a batch", result)
        self.assertNotIn("encountered issues", result)

    def test_single_task_failure_unknown_source(self):
        summarizer = NaturalLanguageSummarizer()
        tasks = [{"source": "unknown_type", "description": "some work"}]
        result = summarizer.summarize(tasks, success=False)
        self.assertNotIn("I but", result)
        self.assertIn("some work", result)
        self.assertIn("encountered issues", result)


if __name__ == "__main__":
    unittest.main()
