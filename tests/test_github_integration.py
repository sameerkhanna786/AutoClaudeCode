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


if __name__ == "__main__":
    unittest.main()
