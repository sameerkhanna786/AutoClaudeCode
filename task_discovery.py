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


@dataclass
class Task:
    description: str
    priority: int  # 1 = highest (feedback), 5 = lowest (quality review)
    source: str  # "test_failure", "lint", "todo", "coverage", "quality", "feedback"
    source_file: Optional[str] = None  # file path for feedback tasks
    line_number: Optional[int] = None


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

        # Build a regex pattern
        pattern = re.compile(r"\b(" + "|".join(re.escape(p) for p in patterns) + r")\b.*", re.IGNORECASE)

        target = Path(self.target_dir)
        for root, dirs, files in os.walk(target):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for fname in files:
                if not fname.endswith((".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".rb")):
                    continue

                fpath = Path(root) / fname
                rel_path = str(fpath.relative_to(target))

                try:
                    content = fpath.read_text(errors="ignore")
                except OSError:
                    continue

                for i, line in enumerate(content.split("\n"), 1):
                    match = pattern.search(line)
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

        return tasks[:20]  # Cap TODO tasks

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
