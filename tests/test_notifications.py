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

    @patch("notifications._resolve_and_check_ip", return_value=(False, "93.184.216.34"))
    @patch("notifications.urllib.request.urlopen")
    def test_slack_payload(self, mock_urlopen, _mock_ip):
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

    @patch("notifications._resolve_and_check_ip", return_value=(False, "93.184.216.34"))
    @patch("notifications.urllib.request.urlopen")
    def test_discord_payload(self, mock_urlopen, _mock_ip):
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

    @patch("notifications._resolve_and_check_ip", return_value=(False, "93.184.216.34"))
    @patch("notifications.urllib.request.urlopen")
    def test_generic_payload(self, mock_urlopen, _mock_ip):
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

    @patch("notifications._resolve_and_check_ip", return_value=(False, "93.184.216.34"))
    @patch("notifications.urllib.request.urlopen")
    def test_error_does_not_propagate(self, mock_urlopen, _mock_ip):
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

    @patch("notifications._resolve_and_check_ip", return_value=(False, "93.184.216.34"))
    @patch("notifications.urllib.request.urlopen")
    def test_duplicate_event_rate_limited(self, mock_urlopen, _mock_ip):
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

    @patch("notifications._resolve_and_check_ip", return_value=(False, "93.184.216.34"))
    @patch("notifications.urllib.request.urlopen")
    def test_empty_url_skipped(self, mock_urlopen, _mock_ip):
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

    @patch("notifications._resolve_and_check_ip", return_value=(False, "93.184.216.34"))
    @patch("notifications.urllib.request.urlopen")
    def test_large_details_still_deduped(self, mock_urlopen, _mock_ip):
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

    @patch("notifications._resolve_and_check_ip", return_value=(False, "93.184.216.34"))
    @patch("notifications.urllib.request.urlopen")
    def test_different_details_not_deduped(self, mock_urlopen, _mock_ip):
        """Different details produce different hashes, so both should send."""
        mock_urlopen.return_value.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"")))
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        config = _make_config()
        mgr = NotificationManager(config)

        mgr.notify("cycle_success", {"task": "a"})
        mgr.notify("cycle_success", {"task": "b"})

        time.sleep(0.2)
        self.assertEqual(mock_urlopen.call_count, 2)

    @patch("notifications._resolve_and_check_ip", return_value=(False, "93.184.216.34"))
    def test_dedup_keys_are_bounded_size(self, _mock_ip):
        """Dedup keys should use MD5 hashes, not raw JSON strings."""
        config = _make_config()
        mgr = NotificationManager(config)

        large_details = {"data": "x" * 100000}
        mgr.notify("cycle_success", large_details)

        # The keys in _recent should be bounded (event name + ":" + 32-char hex)
        for key in mgr._recent:
            # MD5 hex is 32 chars, plus "cycle_success:" prefix = ~46 chars
            self.assertLess(len(key), 100)
        mgr.shutdown()


class TestPeriodicSummaryEvent(unittest.TestCase):

    @patch("notifications.urllib.request.urlopen")
    def test_periodic_summary_respects_config(self, mock_urlopen):
        """periodic_summary should be filterable via on_periodic_summary config."""
        events = NotificationEventsConfig(on_periodic_summary=False)
        config = _make_config(events=events)
        mgr = NotificationManager(config)
        mgr.notify("periodic_summary", {"summary": "test"})
        mock_urlopen.assert_not_called()

    @patch("notifications._resolve_and_check_ip", return_value=(False, "93.184.216.34"))
    @patch("notifications.urllib.request.urlopen")
    def test_periodic_summary_enabled_sends(self, mock_urlopen, _mock_ip):
        """periodic_summary sends notification when enabled."""
        mock_urlopen.return_value.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"")))
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)
        events = NotificationEventsConfig(on_periodic_summary=True)
        config = _make_config(events=events)
        mgr = NotificationManager(config)
        mgr.notify("periodic_summary", {"summary": "test"})
        time.sleep(0.2)
        mock_urlopen.assert_called_once()


class TestWebhookThreadPool(unittest.TestCase):

    def test_uses_thread_pool_instead_of_raw_threads(self):
        """NotificationManager should use a bounded ThreadPoolExecutor."""
        config = _make_config()
        mgr = NotificationManager(config)
        from concurrent.futures import ThreadPoolExecutor
        self.assertIsInstance(mgr._webhook_pool, ThreadPoolExecutor)

    @patch("notifications._resolve_and_check_ip", return_value=(False, "93.184.216.34"))
    @patch("notifications.urllib.request.urlopen")
    def test_thread_pool_sends_webhook(self, mock_urlopen, _mock_ip):
        """Webhooks should still be sent via the thread pool."""
        mock_urlopen.return_value.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"")))
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)
        events = NotificationEventsConfig(on_cycle_success=True)
        config = _make_config(events=events)
        mgr = NotificationManager(config)
        mgr.notify("cycle_success", {"test": True})
        time.sleep(0.3)
        mock_urlopen.assert_called_once()


class TestNotificationManagerShutdown(unittest.TestCase):
    """Tests for the shutdown method."""

    def test_shutdown_does_not_raise(self):
        """shutdown() should complete without error."""
        config = _make_config()
        mgr = NotificationManager(config)
        mgr.shutdown()  # Should not raise

    def test_shutdown_prevents_new_sends(self):
        """After shutdown, the pool should reject new work."""
        config = _make_config()
        mgr = NotificationManager(config)
        mgr.shutdown()
        # After shutdown, submitting new work should raise RuntimeError
        import concurrent.futures
        with self.assertRaises(RuntimeError):
            mgr._webhook_pool.submit(lambda: None)


class TestNotificationShutdownWaits(unittest.TestCase):
    """shutdown() must use wait=True to drain pending notifications."""

    def test_shutdown_waits_for_pending(self):
        """shutdown() should call pool.shutdown(wait=True)."""
        import inspect
        source = inspect.getsource(NotificationManager.shutdown)
        assert "wait=True" in source, (
            "shutdown() should use wait=True to drain pending notifications"
        )


class TestDisabledManagerNoThreadPool(unittest.TestCase):
    """When notifications are disabled, no ThreadPoolExecutor should be created."""

    def test_disabled_no_pool_created(self):
        """A disabled NotificationManager should not allocate a thread pool."""
        config = _make_config(enabled=False)
        mgr = NotificationManager(config)
        # _webhook_pool should be None when disabled
        self.assertIsNone(mgr._webhook_pool)

    def test_disabled_shutdown_safe(self):
        """shutdown() on a disabled manager should not raise."""
        config = _make_config(enabled=False)
        mgr = NotificationManager(config)
        mgr.shutdown()  # Should not raise even with no pool


class TestSSRFProtection(unittest.TestCase):
    """Webhook URLs targeting private/loopback IPs must be rejected."""

    @patch("notifications.urllib.request.urlopen")
    @patch("notifications.socket.getaddrinfo")
    def test_rejects_loopback_ip(self, mock_getaddrinfo, mock_urlopen):
        """Webhooks pointing to 127.0.0.1 must be blocked."""
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("127.0.0.1", 0)),
        ]
        webhook = WebhookConfig(
            url="http://localhost/steal", type="generic", name="evil",
        )
        config = _make_config(webhooks=[webhook])
        mgr = NotificationManager(config)
        mgr.notify("cycle_success", {"tasks": ["test"]})
        time.sleep(0.3)
        mock_urlopen.assert_not_called()

    @patch("notifications.urllib.request.urlopen")
    @patch("notifications.socket.getaddrinfo")
    def test_rejects_private_ip(self, mock_getaddrinfo, mock_urlopen):
        """Webhooks pointing to 10.x.x.x must be blocked."""
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("10.0.0.1", 0)),
        ]
        webhook = WebhookConfig(
            url="http://internal.corp/api", type="generic", name="internal",
        )
        config = _make_config(webhooks=[webhook])
        mgr = NotificationManager(config)
        mgr.notify("cycle_success", {"tasks": ["test"]})
        time.sleep(0.3)
        mock_urlopen.assert_not_called()

    @patch("notifications.urllib.request.urlopen")
    @patch("notifications.socket.getaddrinfo")
    def test_rejects_link_local_metadata(self, mock_getaddrinfo, mock_urlopen):
        """Webhooks pointing to 169.254.169.254 (cloud metadata) must be blocked."""
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("169.254.169.254", 0)),
        ]
        webhook = WebhookConfig(
            url="http://169.254.169.254/latest/meta-data/", type="generic", name="metadata",
        )
        config = _make_config(webhooks=[webhook])
        mgr = NotificationManager(config)
        mgr.notify("cycle_success", {"tasks": ["test"]})
        time.sleep(0.3)
        mock_urlopen.assert_not_called()

    @patch("notifications.urllib.request.urlopen")
    @patch("notifications.socket.getaddrinfo")
    def test_allows_public_ip(self, mock_getaddrinfo, mock_urlopen):
        """Webhooks pointing to public IPs should succeed."""
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("151.101.1.69", 0)),
        ]
        mock_urlopen.return_value.__enter__ = MagicMock(
            return_value=MagicMock(read=MagicMock(return_value=b"")),
        )
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        webhook = WebhookConfig(
            url="https://hooks.slack.com/test", type="generic", name="public",
        )
        config = _make_config(webhooks=[webhook])
        mgr = NotificationManager(config)
        mgr.notify("cycle_success", {"tasks": ["test"]})
        time.sleep(0.3)
        mock_urlopen.assert_called_once()


class TestIsPrivateIpFunction(unittest.TestCase):
    """Unit tests for _is_private_ip helper."""

    @patch("notifications.socket.getaddrinfo")
    def test_private_range_detected(self, mock_getaddrinfo):
        from notifications import _is_private_ip
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("192.168.1.1", 0)),
        ]
        self.assertTrue(_is_private_ip("evil.local"))

    @patch("notifications.socket.getaddrinfo")
    def test_public_range_allowed(self, mock_getaddrinfo):
        from notifications import _is_private_ip
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("8.8.8.8", 0)),
        ]
        self.assertFalse(_is_private_ip("dns.google"))

    @patch("notifications.socket.getaddrinfo")
    def test_dns_failure_returns_true(self, mock_getaddrinfo):
        """DNS failure should block the request to prevent SSRF via unresolvable hosts."""
        from notifications import _is_private_ip
        import socket
        mock_getaddrinfo.side_effect = socket.gaierror("DNS lookup failed")
        self.assertTrue(_is_private_ip("nonexistent.local"))

    @patch("notifications.socket.getaddrinfo")
    def test_os_error_returns_true(self, mock_getaddrinfo):
        """OSError during DNS should also block to prevent SSRF."""
        from notifications import _is_private_ip
        mock_getaddrinfo.side_effect = OSError("Network unreachable")
        self.assertTrue(_is_private_ip("unreachable.host"))


class TestResolveAndCheckIp(unittest.TestCase):
    """Tests for _resolve_and_check_ip which resolves DNS once to prevent rebinding."""

    @patch("notifications.socket.getaddrinfo")
    def test_returns_resolved_ip(self, mock_getaddrinfo):
        from notifications import _resolve_and_check_ip
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 0)),
        ]
        is_private, resolved_ip = _resolve_and_check_ip("example.com")
        self.assertFalse(is_private)
        self.assertEqual(resolved_ip, "93.184.216.34")

    @patch("notifications.socket.getaddrinfo")
    def test_detects_private_ip(self, mock_getaddrinfo):
        from notifications import _resolve_and_check_ip
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("192.168.1.1", 0)),
        ]
        is_private, resolved_ip = _resolve_and_check_ip("internal.local")
        self.assertTrue(is_private)
        self.assertEqual(resolved_ip, "192.168.1.1")

    @patch("notifications.socket.getaddrinfo")
    def test_dns_failure_returns_private(self, mock_getaddrinfo):
        from notifications import _resolve_and_check_ip
        import socket
        mock_getaddrinfo.side_effect = socket.gaierror("DNS failed")
        is_private, resolved_ip = _resolve_and_check_ip("nonexistent.host")
        self.assertTrue(is_private)
        self.assertEqual(resolved_ip, "")

    @patch("notifications.socket.getaddrinfo")
    def test_ipv6_mapped_ipv4_loopback_blocked(self, mock_getaddrinfo):
        """IPv6-mapped IPv4 loopback (::ffff:127.0.0.1) must be blocked."""
        from notifications import _resolve_and_check_ip
        mock_getaddrinfo.return_value = [
            (10, 1, 6, "", ("::ffff:127.0.0.1", 0, 0, 0)),
        ]
        is_private, resolved_ip = _resolve_and_check_ip("evil.com")
        self.assertTrue(is_private)
        self.assertEqual(resolved_ip, "::ffff:127.0.0.1")

    @patch("notifications.socket.getaddrinfo")
    def test_ipv6_mapped_ipv4_private_blocked(self, mock_getaddrinfo):
        """IPv6-mapped IPv4 private (::ffff:192.168.1.1) must be blocked."""
        from notifications import _resolve_and_check_ip
        mock_getaddrinfo.return_value = [
            (10, 1, 6, "", ("::ffff:192.168.1.1", 0, 0, 0)),
        ]
        is_private, resolved_ip = _resolve_and_check_ip("evil.com")
        self.assertTrue(is_private)

    @patch("notifications.socket.getaddrinfo")
    def test_ipv6_mapped_ipv4_public_allowed(self, mock_getaddrinfo):
        """IPv6-mapped IPv4 public (::ffff:8.8.8.8) should be allowed."""
        from notifications import _resolve_and_check_ip
        mock_getaddrinfo.return_value = [
            (10, 1, 6, "", ("::ffff:8.8.8.8", 0, 0, 0)),
        ]
        is_private, resolved_ip = _resolve_and_check_ip("dns.google")
        self.assertFalse(is_private)

    @patch("notifications.socket.getaddrinfo")
    def test_backward_compatible_with_is_private_ip(self, mock_getaddrinfo):
        """_is_private_ip still works correctly using _resolve_and_check_ip."""
        from notifications import _is_private_ip
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("10.0.0.5", 0)),
        ]
        self.assertTrue(_is_private_ip("internal.corp"))


class TestHttpWebhookWarning(unittest.TestCase):
    """Verify a warning is logged for HTTP (non-HTTPS) webhooks."""

    @patch("notifications.urllib.request.urlopen")
    @patch("notifications.logger")
    def test_http_webhook_logs_warning(self, mock_logger, mock_urlopen):
        mock_urlopen.return_value.__enter__ = MagicMock(
            return_value=MagicMock(read=MagicMock(return_value=b""))
        )
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)
        config = _make_config(
            webhooks=[WebhookConfig(url="http://hooks.example.com/test", type="generic", name="test")]
        )
        mgr = NotificationManager(config)
        mgr.notify("cycle_complete", {"task": "test", "success": True})
        mock_logger.warning.assert_any_call(
            "Webhook uses insecure HTTP; consider using HTTPS: %s",
            "http://hooks.example.com/test",
        )

    @patch("notifications.urllib.request.urlopen")
    @patch("notifications.logger")
    def test_https_webhook_no_http_warning(self, mock_logger, mock_urlopen):
        mock_urlopen.return_value.__enter__ = MagicMock(
            return_value=MagicMock(read=MagicMock(return_value=b""))
        )
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)
        config = _make_config()  # default uses https
        mgr = NotificationManager(config)
        mgr.notify("cycle_complete", {"task": "test", "success": True})
        # Should NOT have logged the HTTP warning
        for call_args in mock_logger.warning.call_args_list:
            self.assertNotIn("insecure HTTP", str(call_args))


class TestWebhookPoolNoneGuard(unittest.TestCase):
    """When NotificationManager is created with enabled=False, _webhook_pool is None.

    Calling notify() after mutating _config.enabled to True should not crash
    with AttributeError when trying to call None.submit().
    """

    def test_notify_with_none_pool_does_not_crash(self):
        config = _make_config(enabled=False)
        mgr = NotificationManager(config)
        assert mgr._webhook_pool is None

        # Mutate enabled to True to simulate config reload
        mgr._config.enabled = True
        # This should not raise AttributeError
        mgr.notify("cycle_success", {"tasks": ["test"]})


class TestNotificationManagerContextManager(unittest.TestCase):
    """Tests for NotificationManager __enter__/__exit__ context manager protocol."""

    def test_context_manager_returns_self(self):
        config = _make_config(enabled=False)
        mgr = NotificationManager(config)
        with mgr as ctx:
            assert ctx is mgr

    def test_context_manager_calls_shutdown(self):
        config = _make_config(enabled=False)
        mgr = NotificationManager(config)
        with unittest.mock.patch.object(mgr, "shutdown") as mock_shutdown:
            with mgr:
                pass
            mock_shutdown.assert_called_once()

    def test_context_manager_exit_returns_false(self):
        config = _make_config(enabled=False)
        mgr = NotificationManager(config)
        result = mgr.__exit__(None, None, None)
        assert result is False


class TestEnabledNoWebhooksNoPool(unittest.TestCase):
    """Regression: enabled=True but empty webhooks should not create thread pool."""

    def test_no_pool_when_enabled_but_no_webhooks(self):
        config = _make_config(enabled=True)
        config.webhooks = []
        mgr = NotificationManager(config)
        assert mgr._webhook_pool is None, (
            "Thread pool should not be created when enabled but no webhooks configured"
        )
        mgr.shutdown()

    def test_pool_created_when_enabled_with_webhooks(self):
        config = _make_config(enabled=True)
        config.webhooks = [WebhookConfig(url="https://hooks.slack.com/test")]
        mgr = NotificationManager(config)
        assert mgr._webhook_pool is not None, (
            "Thread pool should be created when enabled with webhooks"
        )
        mgr.shutdown()


class TestNotificationManagerDel(unittest.TestCase):
    """Tests for NotificationManager.__del__ thread pool cleanup."""

    def test_del_shuts_down_pool(self):
        """__del__ calls shutdown(wait=False) on the thread pool."""
        config = _make_config(enabled=True)
        config.webhooks = [WebhookConfig(url="https://hooks.slack.com/test")]
        mgr = NotificationManager(config)
        assert mgr._webhook_pool is not None
        mock_pool = MagicMock()
        mgr._webhook_pool = mock_pool
        mgr.__del__()
        mock_pool.shutdown.assert_called_once_with(wait=False)

    def test_del_no_pool_no_error(self):
        """__del__ does not raise when _webhook_pool is None."""
        config = _make_config(enabled=False)
        mgr = NotificationManager(config)
        assert mgr._webhook_pool is None
        # Should not raise
        mgr.__del__()

    def test_del_suppresses_shutdown_exception(self):
        """__del__ suppresses exceptions from shutdown to avoid errors during GC."""
        config = _make_config(enabled=True)
        config.webhooks = [WebhookConfig(url="https://hooks.slack.com/test")]
        mgr = NotificationManager(config)
        mock_pool = MagicMock()
        mock_pool.shutdown.side_effect = RuntimeError("shutdown failed")
        mgr._webhook_pool = mock_pool
        # Should not raise
        mgr.__del__()


class TestDnsRebindingUrlReplacement(unittest.TestCase):
    """URL hostname-to-IP replacement must only modify the netloc, not the path."""

    def test_hostname_in_path_not_corrupted(self):
        """When hostname appears in the URL path, only the netloc is replaced."""
        from notifications import NotificationManager
        webhook = WebhookConfig(
            url="https://hooks.slack.com/services/slack.com/abc",
            type="generic",
            name="test",
        )
        config = _make_config(webhooks=[webhook])
        mgr = NotificationManager(config)

        # Call _send_webhook with a mocked urlopen and IP resolution
        with patch("notifications._resolve_and_check_ip", return_value=(False, "1.2.3.4")), \
             patch("notifications.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__ = MagicMock(
                return_value=MagicMock(read=MagicMock(return_value=b""))
            )
            mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)
            mgr._send_webhook(webhook, "test_event", {})

            # The URL passed to urlopen should have the IP only in the netloc
            req = mock_urlopen.call_args[0][0]
            url = req.full_url
            # The path should still contain "slack.com" unchanged
            self.assertIn("/services/slack.com/abc", url)
            # The netloc should have the resolved IP
            from urllib.parse import urlparse
            parsed = urlparse(url)
            self.assertIn("1.2.3.4", parsed.netloc)

    def test_simple_url_replacement_works(self):
        """Standard URLs (hostname not in path) still work correctly."""
        from notifications import NotificationManager
        webhook = WebhookConfig(
            url="https://hooks.example.com/webhook",
            type="generic",
            name="test",
        )
        config = _make_config(webhooks=[webhook])
        mgr = NotificationManager(config)

        with patch("notifications._resolve_and_check_ip", return_value=(False, "93.184.216.34")), \
             patch("notifications.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__ = MagicMock(
                return_value=MagicMock(read=MagicMock(return_value=b""))
            )
            mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)
            mgr._send_webhook(webhook, "test_event", {})

            req = mock_urlopen.call_args[0][0]
            url = req.full_url
            from urllib.parse import urlparse
            parsed = urlparse(url)
            self.assertIn("93.184.216.34", parsed.netloc)
            self.assertEqual(parsed.path, "/webhook")


class TestIPv6AddressBracketing(unittest.TestCase):
    """IPv6 resolved addresses must be wrapped in brackets in URL netloc."""

    def test_ipv6_address_gets_brackets_in_url(self):
        """When DNS resolves to an IPv6 address, it must be bracketed in the URL."""
        from notifications import NotificationManager
        webhook = WebhookConfig(
            url="https://hooks.example.com/webhook",
            type="generic",
            name="test",
        )
        config = _make_config(webhooks=[webhook])
        mgr = NotificationManager(config)

        with patch("notifications._resolve_and_check_ip", return_value=(False, "2001:db8::1")), \
             patch("notifications.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__ = MagicMock(
                return_value=MagicMock(read=MagicMock(return_value=b""))
            )
            mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)
            mgr._send_webhook(webhook, "test_event", {})

            req = mock_urlopen.call_args[0][0]
            url = req.full_url
            from urllib.parse import urlparse
            parsed = urlparse(url)
            # IPv6 must be bracketed in netloc
            self.assertIn("[2001:db8::1]", parsed.netloc)
            self.assertEqual(parsed.path, "/webhook")

    def test_ipv4_address_not_bracketed(self):
        """IPv4 addresses should NOT get brackets."""
        from notifications import NotificationManager
        webhook = WebhookConfig(
            url="https://hooks.example.com/webhook",
            type="generic",
            name="test",
        )
        config = _make_config(webhooks=[webhook])
        mgr = NotificationManager(config)

        with patch("notifications._resolve_and_check_ip", return_value=(False, "93.184.216.34")), \
             patch("notifications.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__ = MagicMock(
                return_value=MagicMock(read=MagicMock(return_value=b""))
            )
            mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)
            mgr._send_webhook(webhook, "test_event", {})

            req = mock_urlopen.call_args[0][0]
            url = req.full_url
            # IPv4 should not have brackets
            self.assertNotIn("[", url)
            self.assertIn("93.184.216.34", url)


class TestWebhookResponseBounded(unittest.TestCase):
    """Webhook response reads must be bounded to prevent OOM."""

    @patch("notifications._resolve_and_check_ip", return_value=(False, "93.184.216.34"))
    @patch("notifications.urllib.request.urlopen")
    def test_response_read_has_size_limit(self, mock_urlopen, _mock_ip):
        mock_resp = MagicMock()
        mock_resp.read = MagicMock(return_value=b"ok")
        mock_urlopen.return_value.__enter__ = MagicMock(return_value=mock_resp)
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        webhook = WebhookConfig(url="https://example.com/hook", type="generic", name="test")
        config = _make_config(webhooks=[webhook])
        mgr = NotificationManager(config)
        mgr._send_webhook(webhook, "test_event", {})

        # Verify read() was called with a size limit argument
        mock_resp.read.assert_called_once()
        args = mock_resp.read.call_args
        assert args is not None
        # First positional arg should be the max bytes limit
        assert len(args[0]) > 0, "read() must be called with a size limit"
        assert args[0][0] == 1024 * 1024  # 1 MB


class TestGenericPayloadNLSummary(unittest.TestCase):
    """Verify that generic webhook payloads include NL summaries when enabled."""

    @patch("notifications._resolve_and_check_ip", return_value=(False, "93.184.216.34"))
    @patch("notifications.urllib.request.urlopen")
    def test_generic_payload_includes_nl_summary_when_enabled(self, mock_urlopen, _mock_ip):
        mock_urlopen.return_value.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"")))
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        config = NotificationsConfig(
            enabled=True,
            webhooks=[WebhookConfig(url="https://hooks.example.com/test", type="generic", name="test")],
            events=NotificationEventsConfig(),
            nl_summaries=True,
        )
        mgr = NotificationManager(config)
        mgr.notify("cycle_success", {"tasks": ["fix tests"]})

        time.sleep(0.2)

        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode())
        self.assertIn("summary", payload)
        self.assertIsInstance(payload["summary"], str)
        self.assertTrue(len(payload["summary"]) > 0)

    @patch("notifications._resolve_and_check_ip", return_value=(False, "93.184.216.34"))
    @patch("notifications.urllib.request.urlopen")
    def test_generic_payload_no_summary_when_disabled(self, mock_urlopen, _mock_ip):
        mock_urlopen.return_value.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"")))
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        config = NotificationsConfig(
            enabled=True,
            webhooks=[WebhookConfig(url="https://hooks.example.com/test", type="generic", name="test")],
            events=NotificationEventsConfig(),
            nl_summaries=False,
        )
        mgr = NotificationManager(config)
        mgr.notify("cycle_success", {"tasks": ["fix tests"]})

        time.sleep(0.2)

        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode())
        self.assertNotIn("summary", payload)

    @patch("notifications._resolve_and_check_ip", return_value=(False, "93.184.216.34"))
    @patch("notifications.urllib.request.urlopen")
    def test_generic_payload_no_summary_for_non_cycle_events(self, mock_urlopen, _mock_ip):
        mock_urlopen.return_value.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"")))
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        config = NotificationsConfig(
            enabled=True,
            webhooks=[WebhookConfig(url="https://hooks.example.com/test", type="generic", name="test")],
            events=NotificationEventsConfig(),
            nl_summaries=True,
        )
        mgr = NotificationManager(config)
        mgr.notify("safety_error", {"error": "disk full"})

        time.sleep(0.2)

        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode())
        self.assertNotIn("summary", payload)


class TestNetlocAuthPreservation(unittest.TestCase):
    """URL hostname replacement must not corrupt auth credentials in netloc."""

    def test_hostname_in_password_not_corrupted(self):
        """When password contains the hostname string, only the host is replaced."""
        from notifications import NotificationManager
        # URL with auth: user:example.com@example.com/path
        webhook = WebhookConfig(
            url="https://user:example.com@example.com/webhook",
            type="generic",
            name="test",
        )
        config = _make_config(webhooks=[webhook])
        mgr = NotificationManager(config)

        with patch("notifications._resolve_and_check_ip", return_value=(False, "1.2.3.4")), \
             patch("notifications.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__ = MagicMock(
                return_value=MagicMock(read=MagicMock(return_value=b""))
            )
            mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)
            mgr._send_webhook(webhook, "test_event", {})

            req = mock_urlopen.call_args[0][0]
            url = req.full_url
            from urllib.parse import urlparse
            parsed = urlparse(url)
            # The host should be the resolved IP
            self.assertEqual(parsed.hostname, "1.2.3.4")
            # The password should still be "example.com" (not replaced)
            self.assertEqual(parsed.password, "example.com")
            self.assertEqual(parsed.username, "user")

    def test_url_with_port_preserved(self):
        """Port number must be preserved when replacing hostname with IP."""
        from notifications import NotificationManager
        webhook = WebhookConfig(
            url="https://hooks.example.com:8443/webhook",
            type="generic",
            name="test",
        )
        config = _make_config(webhooks=[webhook])
        mgr = NotificationManager(config)

        with patch("notifications._resolve_and_check_ip", return_value=(False, "1.2.3.4")), \
             patch("notifications.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__ = MagicMock(
                return_value=MagicMock(read=MagicMock(return_value=b""))
            )
            mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)
            mgr._send_webhook(webhook, "test_event", {})

            req = mock_urlopen.call_args[0][0]
            url = req.full_url
            from urllib.parse import urlparse
            parsed = urlparse(url)
            self.assertEqual(parsed.hostname, "1.2.3.4")
            self.assertEqual(parsed.port, 8443)


class TestProviderErrorBodyBounded(unittest.TestCase):
    """Provider runner error body reads must be bounded."""

    def test_openai_error_body_read_bounded(self):
        """OpenAI HTTPError body should be read with a size limit."""
        import urllib.error
        from provider_runner import OpenAIRunner

        config = MagicMock()
        config.claude.model = "gpt-4"
        config.claude.resolved_model = ""
        config.claude.timeout_seconds = 30
        config.claude.api_key_env = ""
        config.claude.provider = "openai"
        config.pricing.cost_per_million_input_tokens = {"sonnet": 3.0}
        config.pricing.output_cost_multiplier = 5.0

        runner = OpenAIRunner(config)
        runner._api_key = "test-key"

        mock_error = urllib.error.HTTPError(
            url="https://api.openai.com/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs={},
            fp=None,
        )
        mock_error.read = MagicMock(return_value=b"error body")

        with patch("provider_runner.urllib.request.urlopen", side_effect=mock_error):
            result = runner.run("test")

        # Verify read was called with a size limit (not unlimited)
        mock_error.read.assert_called_once_with(512)


if __name__ == "__main__":
    unittest.main()
