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

    def test_single_task_failure_default_template_grammar(self):
        """Default failure template should read naturally with desc at the end."""
        summarizer = NaturalLanguageSummarizer()
        tasks = [{"source": "new_source", "description": "fix the widget"}]
        result = summarizer.summarize(tasks, success=False)
        # Description should come after "issues:" not in the middle of the sentence
        self.assertTrue(
            result.endswith("fix the widget"),
            f"Expected description at end, got: {result}",
        )


class TestNotifyDedupWithHashedKey(unittest.TestCase):
    """Tests that rate-limiting dedup uses hashed keys (not raw details)."""

    @patch("notifications.urllib.request.urlopen")
    def test_large_details_still_deduped(self, mock_urlopen):
        """Large payloads should be deduped via hash, not stored as raw keys."""
        mock_urlopen.return_value.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"")))
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        config = _make_config()
        mgr = NotificationManager(config)

        large_details = {"data": "x" * 10000}
        mgr.notify("cycle_success", large_details)
        mgr.notify("cycle_success", large_details)

        time.sleep(0.2)
        # Only one call — second should be rate-limited
        self.assertEqual(mock_urlopen.call_count, 1)

    @patch("notifications.urllib.request.urlopen")
    def test_different_details_not_deduped(self, mock_urlopen):
        """Different details produce different hashes, so both should send."""
        mock_urlopen.return_value.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"")))
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        config = _make_config()
        mgr = NotificationManager(config)

        mgr.notify("cycle_success", {"task": "a"})
        mgr.notify("cycle_success", {"task": "b"})

        time.sleep(0.2)
        self.assertEqual(mock_urlopen.call_count, 2)

    def test_dedup_keys_are_bounded_size(self):
        """Dedup keys should use MD5 hashes, not raw JSON strings."""
        config = _make_config()
        mgr = NotificationManager(config)

        large_details = {"data": "x" * 100000}
        mgr.notify("cycle_success", large_details)

        # The keys in _recent should be bounded (event name + ":" + 32-char hex)
        for key in mgr._recent:
            # MD5 hex is 32 chars, plus "cycle_success:" prefix = ~46 chars
            self.assertLess(len(key), 100)


class TestPeriodicSummaryEvent(unittest.TestCase):

    @patch("notifications.urllib.request.urlopen")
    def test_periodic_summary_respects_config(self, mock_urlopen):
        """periodic_summary should be filterable via on_periodic_summary config."""
        events = NotificationEventsConfig(on_periodic_summary=False)
        config = _make_config(events=events)
        mgr = NotificationManager(config)
        mgr.notify("periodic_summary", {"summary": "test"})
        mock_urlopen.assert_not_called()

    @patch("notifications.urllib.request.urlopen")
    def test_periodic_summary_enabled_sends(self, mock_urlopen):
        """periodic_summary sends notification when enabled."""
        mock_urlopen.return_value.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"")))
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)
        events = NotificationEventsConfig(on_periodic_summary=True)
        config = _make_config(events=events)
        mgr = NotificationManager(config)
        mgr.notify("periodic_summary", {"summary": "test"})
        time.sleep(0.2)
        mock_urlopen.assert_called_once()


if __name__ == "__main__":
    unittest.main()
