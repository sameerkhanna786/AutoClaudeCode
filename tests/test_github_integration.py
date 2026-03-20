"""Tests for github_integration.py — GitHub API client."""

from __future__ import annotations

import os
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

    @patch("github_integration.run_with_group_kill")
    @patch.object(GitHubClient, "create_pull_request", return_value={"number": 1})
    def test_push_does_not_pass_capture_output(self, mock_create_pr, mock_group_kill):
        """run_with_group_kill does not accept capture_output; must not be passed."""
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
        _, kwargs = mock_group_kill.call_args
        self.assertNotIn("capture_output", kwargs,
                         "capture_output is not a valid parameter for run_with_group_kill")


class TestTokenFromEnvVar(unittest.TestCase):
    """GitHub token should be resolvable from an environment variable."""

    def test_token_env_resolves_from_environment(self):
        """When token_env is set, token should be resolved from the env var."""
        config = GitHubConfig(
            enabled=True, token_env="TEST_GH_TOKEN_12345",
            repo_owner="test", repo_name="repo",
        )
        import os
        os.environ["TEST_GH_TOKEN_12345"] = "ghp_from_env"
        try:
            client = GitHubClient(config)
            self.assertEqual(client._resolved_token, "ghp_from_env")
        finally:
            del os.environ["TEST_GH_TOKEN_12345"]

    def test_token_env_takes_precedence_over_token(self):
        """token_env should take precedence over plaintext token field."""
        import os
        os.environ["TEST_GH_TOKEN_PREC"] = "ghp_env_value"
        try:
            config = GitHubConfig(
                enabled=True, token="ghp_plaintext",
                token_env="TEST_GH_TOKEN_PREC",
                repo_owner="test", repo_name="repo",
            )
            client = GitHubClient(config)
            self.assertEqual(client._resolved_token, "ghp_env_value")
        finally:
            del os.environ["TEST_GH_TOKEN_PREC"]

    def test_falls_back_to_plaintext_token(self):
        """When token_env is not set, falls back to plaintext token."""
        config = GitHubConfig(
            enabled=True, token="ghp_plaintext",
            repo_owner="test", repo_name="repo",
        )
        client = GitHubClient(config)
        self.assertEqual(client._resolved_token, "ghp_plaintext")


class TestPlaintextTokenWarning(unittest.TestCase):
    """Verify a warning is logged when plaintext token is used."""

    @patch("github_integration.logger")
    def test_plaintext_token_logs_warning(self, mock_logger):
        config = GitHubConfig(
            enabled=True, token="ghp_secret123",
            repo_owner="test", repo_name="repo",
        )
        GitHubClient(config)
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("plaintext", warning_msg)

    @patch("github_integration.logger")
    def test_env_token_no_warning(self, mock_logger):
        config = GitHubConfig(
            enabled=True, token_env="GITHUB_TOKEN",
            repo_owner="test", repo_name="repo",
        )
        with patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_from_env"}):
            GitHubClient(config)
        mock_logger.warning.assert_not_called()

    @patch("github_integration.logger")
    def test_empty_token_no_warning(self, mock_logger):
        config = GitHubConfig(
            enabled=True, token="",
            repo_owner="test", repo_name="repo",
        )
        GitHubClient(config)
        mock_logger.warning.assert_not_called()


class TestBoundedResponseRead(unittest.TestCase):
    """GitHub API response reads must pass a size limit to prevent OOM."""

    @patch("github_integration.urllib.request.urlopen")
    def test_read_passes_size_limit(self, mock_urlopen):
        """_request should call resp.read() with a max size argument."""
        import json as _json
        config = GitHubConfig(
            enabled=True,
            token="ghp_testtoken",
            repo_owner="test",
            repo_name="repo",
        )
        client = GitHubClient(config)

        mock_response = MagicMock()
        mock_response.read.return_value = _json.dumps({"id": 1}).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client._request("GET", "/repos/test/repo")

        # Verify read() was called with a size limit (10 MB)
        mock_response.read.assert_called_once_with(10 * 1024 * 1024)


class TestErrorBodyReadBounded(unittest.TestCase):
    """HTTP error body reads must be bounded to prevent OOM."""

    @patch("github_integration.urllib.request.urlopen")
    def test_error_body_read_has_size_limit(self, mock_urlopen):
        import urllib.error

        config = GitHubConfig(
            enabled=True,
            token="ghp_test123",
            repo_owner="test",
            repo_name="repo",
        )
        client = GitHubClient(config)

        mock_error_fp = MagicMock()
        mock_error_fp.read = MagicMock(return_value=b"error details")
        http_error = urllib.error.HTTPError(
            url="https://api.github.com/test",
            code=500,
            msg="Server Error",
            hdrs=MagicMock(),
            fp=mock_error_fp,
        )
        http_error.read = mock_error_fp.read
        mock_urlopen.side_effect = http_error

        with self.assertRaises(urllib.error.HTTPError):
            client._request("GET", "/repos/test/repo")

        # Verify read() was called with a size limit
        mock_error_fp.read.assert_called_once()
        args = mock_error_fp.read.call_args
        assert args is not None
        assert len(args[0]) > 0, "read() must be called with a size limit"
        assert args[0][0] == 1024 * 1024  # 1 MB


if __name__ == "__main__":
    unittest.main()
