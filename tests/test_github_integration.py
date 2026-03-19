"""Tests for github_integration.py — GitHub API client."""

from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock

from config_schema import GitHubConfig
from github_integration import GitHubClient


class TestTokenSanitization(unittest.TestCase):
    """Verify GitHub tokens are not leaked in error log output."""

    @patch("github_integration.urllib.request.urlopen")
    def test_token_scrubbed_from_error_body(self, mock_urlopen):
        """If the error body contains the token, it must be replaced with ***."""
        import urllib.error

        token = "ghp_s3cr3tT0k3nValue123"
        config = GitHubConfig(
            enabled=True,
            token=token,
            repo_owner="test",
            repo_name="repo",
        )
        client = GitHubClient(config)

        # Simulate an HTTPError whose body echoes the token
        error_body = f"Bad credentials: {token}".encode("utf-8")
        http_error = urllib.error.HTTPError(
            url="https://api.github.com/repos/test/repo/pulls",
            code=401,
            msg="Unauthorized",
            hdrs=MagicMock(),
            fp=MagicMock(),
        )
        http_error.read = MagicMock(return_value=error_body)
        mock_urlopen.side_effect = http_error

        with patch("github_integration.logger") as mock_logger:
            with self.assertRaises(urllib.error.HTTPError):
                client._request("GET", "/repos/test/repo/pulls")

            # Verify logger.error was called and token is NOT in the args
            mock_logger.error.assert_called_once()
            log_args = str(mock_logger.error.call_args)
            self.assertNotIn(token, log_args)
            self.assertIn("***", log_args)


class TestPushAndCreatePrUsesGroupKill(unittest.TestCase):
    """push_and_create_pr should use run_with_group_kill to prevent orphaned processes."""

    @patch("github_integration.run_with_group_kill")
    @patch.object(GitHubClient, "create_pull_request", return_value={"number": 1})
    def test_push_uses_run_with_group_kill(self, mock_create_pr, mock_group_kill):
        """git push should use run_with_group_kill instead of subprocess.run."""
        mock_group_kill.return_value = MagicMock(returncode=0, stderr="")
        config = GitHubConfig(
            enabled=True, create_prs=True,
            token="ghp_test", repo_owner="test", repo_name="repo",
        )
        client = GitHubClient(config)
        client.push_and_create_pr(
            branch_name="feature",
            title="Test PR",
            body="body",
            target_dir="/tmp/test",
        )
        mock_group_kill.assert_called_once()
        args = mock_group_kill.call_args
        cmd = args[0][0] if args[0] else args[1].get("cmd")
        self.assertEqual(cmd, ["git", "push", "-u", "origin", "feature"])


if __name__ == "__main__":
    unittest.main()
