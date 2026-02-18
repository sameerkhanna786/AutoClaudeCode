"""Auto-discover tasks from the target project (test failures, lint, TODOs, etc.)."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from config_schema import Config

logger = logging.getLogger(__name__)

# Map file extensions to their comment prefix styles
_COMMENT_PREFIXES = {
    ".py": ("#",), ".rb": ("#",),
    ".js": ("//", "/*"), ".ts": ("//", "/*"),
    ".jsx": ("//", "/*"), ".tsx": ("//", "/*"),
    ".go": ("//", "/*"), ".rs": ("//", "/*"),
    ".java": ("//", "/*"),
}

# Regex to match string literals (double-quoted and single-quoted, with escapes)
_STRING_LITERAL_RE = re.compile(r'''"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*' '''.strip())

MAX_TASK_DESCRIPTION_LENGTH = 2000


def _sanitize_description(desc: str) -> str:
    """Sanitize a task description: strip, collapse newlines, and truncate."""
    desc = desc.strip()
    desc = desc.replace("\n", " ").replace("\r", " ")
    if len(desc) > MAX_TASK_DESCRIPTION_LENGTH:
        desc = desc[:MAX_TASK_DESCRIPTION_LENGTH] + "..."
    return desc


def _extract_comment_text(line: str, ext: str) -> Optional[str]:
    """Strip string literals from *line*, then return the comment body if a
    comment prefix for *ext* is found.  Returns ``None`` when no comment is
    detected."""
    prefixes = _COMMENT_PREFIXES.get(ext)
    if not prefixes:
        return None

    # Remove string literals so that a '#' or '//' inside a string is ignored
    stripped = _STRING_LITERAL_RE.sub("", line)

    # Find the earliest comment prefix in the stripped line
    earliest_pos = None
    earliest_prefix = None
    for pfx in prefixes:
        pos = stripped.find(pfx)
        if pos != -1 and (earliest_pos is None or pos < earliest_pos):
            earliest_pos = pos
            earliest_prefix = pfx

    if earliest_pos is None:
        return None

    # Return the text after the comment prefix
    return stripped[earliest_pos + len(earliest_prefix):]


@dataclass
class Task:
    description: str
    priority: int  # 1 = highest (feedback), 5 = lowest (quality review)
    source: str  # "test_failure", "lint", "todo", "coverage", "quality", "feedback"
    source_file: Optional[str] = None  # file path for feedback tasks
    line_number: Optional[int] = None

    def __post_init__(self):
        self.description = _sanitize_description(self.description)


class TaskDiscovery:
    def __init__(self, config: Config):
        self.config = config
        self.target_dir = config.target_dir

    def discover_all(self) -> List[Task]:
        """Run all enabled discovery strategies and return sorted tasks."""
        tasks: List[Task] = []
        dc = self.config.discovery

        if dc.enable_test_failures:
            tasks.extend(self._discover_test_failures())

        if dc.enable_lint_errors:
            tasks.extend(self._discover_lint_errors())

        if dc.enable_todos:
            tasks.extend(self._discover_todos())

        if dc.enable_coverage:
            tasks.extend(self._discover_coverage_gaps())

        if dc.enable_claude_ideas:
            tasks.extend(self._discover_claude_ideas())

        if dc.enable_quality_review:
            tasks.extend(self._discover_quality_issues())

        # Sort by priority (lower number = higher priority)
        tasks.sort(key=lambda t: t.priority)
        return tasks

    def _discover_test_failures(self) -> List[Task]:
        """Run pytest and parse failures."""
        test_cmd = self.config.validation.test_command
        if not test_cmd.strip():
            return []

        try:
            proc = subprocess.run(
                test_cmd,
                shell=True,
                cwd=self.target_dir,
                capture_output=True,
                text=True,
                timeout=self.config.validation.test_timeout,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("Test discovery failed: %s", e)
            return []

        if proc.returncode == 0:
            return []

        # Parse pytest output for FAILED lines
        tasks = []
        for line in proc.stdout.split("\n"):
            line = line.strip()
            if line.startswith("FAILED"):
                # e.g. "FAILED tests/test_foo.py::test_bar - AssertionError: ..."
                desc = f"Fix test failure: {line}"
                tasks.append(Task(
                    description=desc,
                    priority=2,
                    source="test_failure",
                ))

        # If we got failures but couldn't parse individual ones, create a generic task
        if proc.returncode != 0 and not tasks:
            tasks.append(Task(
                description=f"Fix test failures (exit code {proc.returncode})",
                priority=2,
                source="test_failure",
            ))

        return tasks

    def _discover_lint_errors(self) -> List[Task]:
        """Run lint command and parse errors."""
        lint_cmd = self.config.validation.lint_command
        if not lint_cmd.strip():
            return []

        try:
            proc = subprocess.run(
                lint_cmd,
                shell=True,
                cwd=self.target_dir,
                capture_output=True,
                text=True,
                timeout=self.config.validation.lint_timeout,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("Lint discovery failed: %s", e)
            return []

        if proc.returncode == 0:
            return []

        # Try to parse ruff JSON output
        output = proc.stdout.strip()
        try:
            errors = json.loads(output)
            if isinstance(errors, list):
                tasks = []
                for err in errors[:10]:  # Cap at 10 lint errors
                    filename = err.get("filename", "unknown")
                    message = err.get("message", "lint error")
                    code = err.get("code", "")
                    desc = f"Fix lint error in {filename}: [{code}] {message}"
                    tasks.append(Task(
                        description=desc,
                        priority=2,
                        source="lint",
                        source_file=filename,
                    ))
                return tasks
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: generic lint error task
        return [Task(
            description="Fix lint errors",
            priority=2,
            source="lint",
        )]

    def _discover_todos(self) -> List[Task]:
        """Scan source files for TODO/FIXME/HACK comments."""
        tasks = []
        patterns = self.config.discovery.todo_patterns
        exclude_dirs = set(self.config.discovery.exclude_dirs)

        if not patterns:
            return []

        keyword_pat = r"\b(" + "|".join(re.escape(p) for p in patterns) + r")\b"
        keyword_re = re.compile(keyword_pat, re.IGNORECASE)

        target = Path(self.target_dir)
        for root, dirs, files in os.walk(target):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for fname in files:
                ext = Path(fname).suffix
                if ext not in _COMMENT_PREFIXES:
                    continue

                fpath = Path(root) / fname
                rel_path = str(fpath.relative_to(target))

                try:
                    content = fpath.read_text(errors="ignore")
                except OSError:
                    continue

                for i, line in enumerate(content.split("\n"), 1):
                    comment_text = _extract_comment_text(line, ext)
                    if comment_text is None:
                        continue
                    match = keyword_re.search(comment_text)
                    if match:
                        comment = line.strip()
                        if len(comment) > 120:
                            comment = comment[:120] + "..."
                        desc = f"Address {match.group(1)} in {rel_path}:{i}: {comment}"
                        tasks.append(Task(
                            description=desc,
                            priority=3,
                            source="todo",
                            source_file=rel_path,
                            line_number=i,
                        ))

        return tasks[:self.config.discovery.max_todo_tasks]

    def _discover_claude_ideas(self) -> List[Task]:
        """Use Claude to analyze the codebase and suggest improvement ideas.

        Invokes Claude CLI in read-only analysis mode with a low max-turns
        to keep cost down. Parses the response for actionable improvement tasks.
        """
        cc = self.config.claude
        prompt = (
            "Analyze the codebase in the current directory. Identify up to 5 concrete, "
            "actionable improvements. Focus on:\n"
            "- Bug risks or edge cases that could cause failures\n"
            "- Missing error handling\n"
            "- Performance improvements\n"
            "- Code clarity or maintainability improvements\n"
            "- Missing tests for important functionality\n"
            "- Design improvements\n\n"
            "For each improvement, output EXACTLY one line in this format:\n"
            "IDEA: <one-sentence description of the improvement>\n\n"
            "Do NOT make any changes. Do NOT run git commands. Only analyze and output IDEA lines."
        )

        cmd = [
            cc.command, "-p", prompt,
            "--model", self.config.discovery.discovery_model,
            "--max-turns", "5",
            "--output-format", "json",
        ]

        try:
            proc = subprocess.run(
                cmd,
                cwd=self.target_dir,
                capture_output=True,
                text=True,
                timeout=self.config.discovery.discovery_timeout,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.warning("Claude idea discovery failed: %s", e)
            return []

        if proc.returncode != 0:
            logger.warning("Claude idea discovery exited with code %d", proc.returncode)
            return []

        # Parse JSON response to get result text
        result_text = proc.stdout
        try:
            lines = result_text.strip().split("\n")
            for i, line in enumerate(lines):
                if line.strip().startswith("{"):
                    try:
                        data = json.loads("\n".join(lines[i:]))
                        if (isinstance(data, dict)
                                and "result" in data
                                and isinstance(data.get("result"), str)):
                            result_text = data["result"]
                            break
                    except (json.JSONDecodeError, TypeError):
                        continue
        except Exception:
            pass

        # Extract IDEA lines
        MIN_IDEA_LENGTH = 10
        tasks = []
        for line in result_text.split("\n"):
            line = line.strip()
            if line.upper().startswith("IDEA:"):
                desc = line[5:].strip()
                if desc and len(desc) >= MIN_IDEA_LENGTH:
                    tasks.append(Task(
                        description=desc,
                        priority=4,
                        source="claude_idea",
                    ))
                elif desc:
                    logger.debug("Skipping short IDEA line: %r", desc)

        logger.info("Claude discovered %d improvement ideas", len(tasks))
        return tasks[:5]

    def _discover_coverage_gaps(self) -> List[Task]:
        """Discover files with low test coverage (optional, requires pytest-cov)."""
        try:
            proc = subprocess.run(
                "python3 -m pytest --cov --cov-report=json --cov-report=term -q",
                shell=True,
                cwd=self.target_dir,
                capture_output=True,
                text=True,
                timeout=self.config.validation.test_timeout,
            )
        except (subprocess.TimeoutExpired, OSError):
            return []

        cov_file = Path(self.target_dir) / "coverage.json"
        if not cov_file.exists():
            return []

        try:
            data = json.loads(cov_file.read_text())
            files = data.get("files", {})
            tasks = []
            for fname, info in files.items():
                pct = info.get("summary", {}).get("percent_covered", 100)
                if pct < 50:
                    desc = f"Improve test coverage for {fname} (currently {pct:.0f}%)"
                    tasks.append(Task(
                        description=desc,
                        priority=4,
                        source="coverage",
                        source_file=fname,
                    ))
            return tasks[:10]
        except (json.JSONDecodeError, OSError):
            return []

    def _discover_quality_issues(self) -> List[Task]:
        """Review source files for general quality issues."""
        tasks = []
        target = Path(self.target_dir)
        exclude_dirs = set(self.config.discovery.exclude_dirs)

        for root, dirs, files in os.walk(target):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for fname in files:
                if not fname.endswith(".py"):
                    continue

                fpath = Path(root) / fname
                rel_path = str(fpath.relative_to(target))

                try:
                    content = fpath.read_text(errors="ignore")
                except OSError:
                    continue

                lines = content.split("\n")
                # Flag very long files
                if len(lines) > 500:
                    tasks.append(Task(
                        description=f"Review and potentially refactor {rel_path} ({len(lines)} lines)",
                        priority=5,
                        source="quality",
                        source_file=rel_path,
                    ))

        return tasks[:5]
