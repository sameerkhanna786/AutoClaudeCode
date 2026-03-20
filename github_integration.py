"""GitHub PR integration for Auto Claude Code (stdlib only)."""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from config_schema import GitHubConfig
from process_utils import run_with_group_kill

logger = logging.getLogger(__name__)


class GitHubClient:
    """Minimal GitHub API client using only stdlib (urllib)."""

    API_BASE = "https://api.github.com"

    def __init__(self, config: GitHubConfig):
        self._config = config
        # Resolve token: prefer env var (token_env) over plaintext (token)
        if config.token_env:
            self._resolved_token = os.environ.get(config.token_env, "")
        else:
            self._resolved_token = config.token
            if config.token:
                logger.warning(
                    "GitHub token is set as plaintext in config. "
                    "Use 'token_env' to reference an environment variable instead."
                )

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        *,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """Make an authenticated GitHub API request.

        Args:
            method: HTTP method (GET, POST, PUT, PATCH, DELETE).
            path: API path (e.g. "/repos/{owner}/{repo}/pulls").
            body: Optional JSON body for POST/PUT/PATCH requests.
            timeout: Request timeout in seconds.

        Returns:
            Parsed JSON response as a dict.

        Raises:
            urllib.error.URLError: On network errors.
            ValueError: On JSON parse errors or missing token.
        """
        if not self._resolved_token:
            raise ValueError("GitHub token is not configured")

        url = f"{self.API_BASE}{path}"
        data = json.dumps(body).encode("utf-8") if body else None

        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"token {self._resolved_token}",
                "Accept": "application/vnd.github.v3+json",
                "Content-Type": "application/json",
                "User-Agent": "auto-claude-code",
            },
        )

        _MAX_RESPONSE_BYTES = 10 * 1024 * 1024  # 10 MB
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                response_data = resp.read(_MAX_RESPONSE_BYTES).decode("utf-8")
                if response_data:
                    return json.loads(response_data)
                return {}
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read(8192).decode("utf-8")
            except Exception:
                logger.debug("Failed to read HTTP error body", exc_info=True)
            # Sanitize error body to prevent token leakage in logs
            if self._resolved_token and self._resolved_token in error_body:
                error_body = error_body.replace(self._resolved_token, "***")
            logger.error(
                "GitHub API error: %s %s -> %d: %s",
                method, path, e.code, error_body,
            )
            raise

    def create_pull_request(
        self,
        title: str,
        body: str,
        head: str,
        base: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a pull request on GitHub.

        Args:
            title: PR title.
            body: PR description/body.
            head: The branch containing the changes.
            base: The branch to merge into (defaults to config base_branch).

        Returns:
            GitHub API response dict containing PR details.
        """
        base = base or self._config.base_branch
        owner = self._config.repo_owner
        repo = self._config.repo_name

        pr_data = {
            "title": title,
            "body": body,
            "head": head,
            "base": base,
        }

        result = self._request("POST", f"/repos/{owner}/{repo}/pulls", pr_data)
        pr_number = result.get("number")
        logger.info("Created PR #%s: %s", pr_number, title)

        # Add label if configured
        if self._config.label and pr_number:
            try:
                self.add_labels(pr_number, [self._config.label])
            except Exception as e:
                logger.warning("Failed to add label to PR #%s: %s", pr_number, e)

        return result

    def comment_on_pr(self, pr_number: int, body: str) -> Dict[str, Any]:
        """Add a comment to a pull request.

        Args:
            pr_number: The PR number.
            body: Comment text.

        Returns:
            GitHub API response dict.
        """
        owner = self._config.repo_owner
        repo = self._config.repo_name

        return self._request(
            "POST",
            f"/repos/{owner}/{repo}/issues/{pr_number}/comments",
            {"body": body},
        )

    def add_labels(self, pr_number: int, labels: List[str]) -> Dict[str, Any]:
        """Add labels to a pull request.

        Args:
            pr_number: The PR number.
            labels: List of label names to add.

        Returns:
            GitHub API response dict.
        """
        owner = self._config.repo_owner
        repo = self._config.repo_name

        return self._request(
            "POST",
            f"/repos/{owner}/{repo}/issues/{pr_number}/labels",
            {"labels": labels},
        )

    def push_and_create_pr(
        self,
        branch_name: str,
        title: str,
        body: str,
        target_dir: str,
        base: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Push the current branch and create a PR.

        This is a convenience method that pushes the local branch to the
        remote and then creates a pull request.

        Args:
            branch_name: Name of the branch to push.
            title: PR title.
            body: PR description.
            target_dir: Working directory for git commands.
            base: Base branch for the PR.

        Returns:
            GitHub API response dict, or None on failure.
        """
        if not self._config.enabled or not self._config.create_prs:
            logger.debug("GitHub PR creation is disabled")
            return None

        # Push the branch using run_with_group_kill to prevent orphaned processes
        try:
            result = run_with_group_kill(
                ["git", "push", "-u", "origin", branch_name],
                cwd=target_dir,
                timeout=120,
            )
            if result.returncode != 0:
                logger.error("Failed to push branch %s: %s", branch_name, result.stderr)
                return None
        except OSError as e:
            logger.error("Git push failed: %s", e)
            return None

        # Create the PR
        try:
            pr = self.create_pull_request(title, body, branch_name, base)
            return pr
        except Exception as e:
            logger.error("Failed to create PR: %s", e)
            return None
