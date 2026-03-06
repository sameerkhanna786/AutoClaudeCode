"""AI-powered merge conflict resolution using Claude."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

from claude_runner import ClaudeRunner, ClaudeResult
from config_schema import Config

logger = logging.getLogger(__name__)

# Regex to detect unresolved conflict markers in file content
CONFLICT_MARKER_RE = re.compile(r'^<{7}\s|^={7}$|^>{7}\s', re.MULTILINE)


class ConflictResolver:
    """Resolves merge conflicts by invoking Claude to produce clean merged files."""

    def __init__(self, config: Config):
        self.config = config
        self.runner = ClaudeRunner(config)
        self._total_cost = 0.0

    def resolve_conflicts(
        self,
        repo_dir: str,
        conflicted_files: List[str],
        worker_branch: str,
        main_branch: str,
    ) -> Tuple[bool, float]:
        """Resolve merge conflicts in the given files using Claude.

        Args:
            repo_dir: Path to the repository root.
            conflicted_files: List of file paths (relative to repo_dir) with conflicts.
            worker_branch: Name of the branch being merged in.
            main_branch: Name of the target branch (e.g. main).

        Returns:
            (success, cost_usd) tuple. success is True only if all files were
            resolved without remaining conflict markers.
        """
        self._total_cost = 0.0
        max_cost = self.config.parallel.conflict_resolution_max_cost

        for filepath in conflicted_files:
            # Cost guard
            if self._total_cost >= max_cost:
                logger.warning(
                    "Conflict resolution cost limit reached (%.2f >= %.2f), aborting",
                    self._total_cost, max_cost,
                )
                return False, self._total_cost

            abs_path = Path(repo_dir) / filepath
            if not abs_path.exists():
                logger.warning("Conflicted file not found: %s", abs_path)
                return False, self._total_cost

            conflicted_content = abs_path.read_text(errors="replace")
            prompt = self._build_resolve_prompt(
                filepath, conflicted_content, worker_branch, main_branch,
            )

            result: ClaudeResult = self.runner.run(prompt)
            self._total_cost += result.cost_usd

            if not result.success or not result.result_text:
                logger.warning(
                    "Claude failed to resolve conflicts in %s: %s",
                    filepath, result.error or "empty response",
                )
                return False, self._total_cost

            resolved = self._extract_resolved_content(result.result_text)
            if resolved is None:
                logger.warning(
                    "Could not extract resolved content for %s", filepath,
                )
                return False, self._total_cost

            # Validate no conflict markers remain
            if CONFLICT_MARKER_RE.search(resolved):
                logger.warning(
                    "Resolved content for %s still contains conflict markers",
                    filepath,
                )
                return False, self._total_cost

            # Write resolved content back
            abs_path.write_text(resolved)
            logger.info("Resolved conflicts in %s", filepath)

        return True, self._total_cost

    @staticmethod
    def _build_resolve_prompt(
        filepath: str,
        conflicted_content: str,
        worker_branch: str,
        main_branch: str,
    ) -> str:
        """Build the prompt sent to Claude for conflict resolution."""
        return (
            f"You are resolving a merge conflict in the file `{filepath}`.\n"
            f"The file is being merged from branch `{worker_branch}` into `{main_branch}`.\n\n"
            f"Below is the file content with conflict markers. The sections between\n"
            f"`<<<<<<< HEAD` and `=======` are from `{main_branch}`, and the sections\n"
            f"between `=======` and `>>>>>>> {worker_branch}` are from `{worker_branch}`.\n\n"
            f"```\n{conflicted_content}\n```\n\n"
            f"Produce the fully resolved file content. Combine both sides logically,\n"
            f"keeping all meaningful changes from both branches. Do NOT leave any\n"
            f"conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`) in the output.\n\n"
            f"Return ONLY the resolved file content inside a single code block."
        )

    @staticmethod
    def _extract_resolved_content(response_text: str) -> Optional[str]:
        """Extract file content from Claude's response.

        Looks for a fenced code block first; falls back to the raw text
        if no code block is found.  Returns None for empty responses.
        """
        if not response_text or not response_text.strip():
            return None

        # Try to find a fenced code block (``` ... ```)
        # Match opening ``` with optional language tag, then content, then closing ```
        code_block_re = re.compile(
            r'```[^\n]*\n(.*?)```', re.DOTALL,
        )
        match = code_block_re.search(response_text)
        if match:
            return match.group(1)

        # No code block found — return the raw text
        return response_text.strip()
