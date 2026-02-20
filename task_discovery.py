"""Auto-discover tasks from the target project (test failures, lint, TODOs, etc.)."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from config_schema import Config
from process_utils import run_with_group_kill

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
MAX_TASK_CONTEXT_LENGTH = 12000

# Default timeout for _discover_todos file walk (seconds)
TODO_SCAN_TIMEOUT = 60

_FILE_REF_RE = re.compile(
    r'`([a-zA-Z0-9_/.\-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|rb|sh|yaml|yml|json|md|txt))'
    r'(?::(\d+))?(?:-\d+)?`'
)
_FILE_REF_FALLBACK_RE = re.compile(
    r'(?:in\s+|for\s+)([a-zA-Z0-9_/.\-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|rb|sh|yaml|yml|json|md|txt))'
    r'(?::(\d+))?'
)


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
    context: str = ""  # rich context: tracebacks, file snippets, error details

    def __post_init__(self):
        self.description = _sanitize_description(self.description)
        if len(self.context) > MAX_TASK_CONTEXT_LENGTH:
            self.context = self.context[:MAX_TASK_CONTEXT_LENGTH] + "\n... (truncated)"

    @property
    def task_key(self) -> str:
        """Stable dedup key — same underlying issue produces the same key."""
        if self.source == "todo" and self.source_file:
            if self.line_number is not None:
                return f"todo:{self.source_file}:{self.line_number}"
            return f"todo:{self.source_file}"

        if self.source in ("lint", "test_failure", "quality", "coverage") and self.source_file:
            return f"{self.source}:{self.source_file}"

        if self.source == "coverage":
            match = re.search(r'for\s+(\S+)', self.description)
            if match:
                return f"coverage:{match.group(1)}"

        if self.source == "claude_idea":
            match = _FILE_REF_RE.search(self.description)
            if match:
                return f"claude_idea:{match.group(1)}"
            match = _FILE_REF_FALLBACK_RE.search(self.description)
            if match:
                return f"claude_idea:{match.group(1)}"
            return f"claude_idea:{self.description[:60]}"

        if self.source == "feedback" and self.source_file:
            return f"feedback:{self.source_file}"

        if self.source == "test_failure":
            match = re.search(r'FAILED\s+(\S+)', self.description)
            if match:
                return f"test_failure:{match.group(1)}"

        return f"{self.source}:{self.description}"


class TaskDiscovery:
    def __init__(self, config: Config, state_manager=None):
        self.config = config
        self.target_dir = config.target_dir
        self.state_manager = state_manager

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

    def _extract_test_traceback(self, full_output: str, test_id: str) -> str:
        """Extract the traceback section for a specific test from pytest output."""
        if not test_id:
            return ""
        lines = full_output.split("\n")
        collecting = False
        result_lines: List[str] = []
        for line in lines:
            if test_id in line and ("FAILED" in line or "ERROR" in line or "___" in line):
                collecting = True
                result_lines = [line]
                continue
            if collecting:
                if (line.startswith("___") and test_id not in line) or \
                   line.startswith("=") or \
                   (line.startswith("FAILED ") and test_id not in line):
                    break
                result_lines.append(line)
        return "\n".join(result_lines)

    def _read_file_snippet(self, filepath: str, line_num: int, context_lines: int = 5) -> str:
        """Read a snippet of a file around the given line number."""
        try:
            fpath = Path(filepath)
            if not fpath.is_absolute():
                fpath = Path(self.target_dir) / filepath
            if not fpath.exists():
                return ""
            content = fpath.read_text(errors="ignore")
            lines = content.split("\n")
            start = max(0, line_num - context_lines - 1)
            end = min(len(lines), line_num + context_lines)
            numbered = []
            for i, line in enumerate(lines[start:end], start + 1):
                marker = " >> " if i == line_num else "    "
                numbered.append(f"{marker}{i:4d} | {line}")
            return "\n".join(numbered)
        except OSError:
            return ""

    def _discover_test_failures(self) -> List[Task]:
        """Run pytest and parse failures."""
        test_cmd = self.config.validation.test_command
        if not test_cmd.strip():
            return []

        try:
            result = run_with_group_kill(
                test_cmd,
                shell=True,
                cwd=self.target_dir,
                timeout=self.config.validation.test_timeout,
            )
        except OSError as e:
            logger.warning("Test discovery failed: %s", e)
            return []

        if result.timed_out:
            logger.warning("Test discovery timed out after %ds", self.config.validation.test_timeout)
            return []

        if result.returncode == 0:
            return []

        # Capture full output for context
        full_output = result.stdout
        if result.stderr.strip():
            full_output += "\n" + result.stderr

        # Parse pytest output for FAILED lines
        tasks = []
        for line in result.stdout.split("\n"):
            line = line.strip()
            if line.startswith("FAILED"):
                # e.g. "FAILED tests/test_foo.py::test_bar - AssertionError: ..."
                desc = f"Fix test failure: {line}"
                # Extract test identifier for per-test traceback
                parts = line.split()
                test_id = parts[1] if len(parts) > 1 else ""
                per_test_ctx = self._extract_test_traceback(full_output, test_id)
                tasks.append(Task(
                    description=desc,
                    priority=2,
                    source="test_failure",
                    context=per_test_ctx if per_test_ctx else full_output,
                ))

        # If we got failures but couldn't parse individual ones, create a generic task
        if result.returncode != 0 and not tasks:
            tasks.append(Task(
                description=f"Fix test failures (exit code {result.returncode})",
                priority=2,
                source="test_failure",
                context=full_output,
            ))

        return tasks

    def _discover_lint_errors(self) -> List[Task]:
        """Run lint command and parse errors."""
        lint_cmd = self.config.validation.lint_command
        if not lint_cmd.strip():
            return []

        try:
            result = run_with_group_kill(
                lint_cmd,
                shell=True,
                cwd=self.target_dir,
                timeout=self.config.validation.lint_timeout,
            )
        except OSError as e:
            logger.warning("Lint discovery failed: %s", e)
            return []

        if result.timed_out:
            logger.warning("Lint discovery timed out after %ds", self.config.validation.lint_timeout)
            return []

        if result.returncode == 0:
            return []

        # Try to parse ruff JSON output
        output = result.stdout.strip()
        try:
            errors = json.loads(output)
            if isinstance(errors, list):
                tasks = []
                for err in errors[:10]:  # Cap at 10 lint errors
                    filename = err.get("filename", "unknown")
                    message = err.get("message", "lint error")
                    code = err.get("code", "")
                    line_num = err.get("location", {}).get("row", 0)
                    desc = f"Fix lint error in {filename}: [{code}] {message}"
                    # Build context: snippet around the error line
                    context = ""
                    if line_num > 0:
                        context = self._read_file_snippet(filename, line_num, context_lines=5)
                    tasks.append(Task(
                        description=desc,
                        priority=2,
                        source="lint",
                        source_file=filename,
                        line_number=line_num if line_num > 0 else None,
                        context=context,
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

    def _discover_todos(self, timeout: int = TODO_SCAN_TIMEOUT) -> List[Task]:
        """Scan source files for TODO/FIXME/HACK comments.

        Args:
            timeout: Maximum time in seconds for the file walk operation.
                     If exceeded, returns whatever tasks were found so far.
        """
        tasks = []
        patterns = self.config.discovery.todo_patterns
        exclude_dirs = set(self.config.discovery.exclude_dirs)

        if not patterns:
            return []

        escaped = "|".join(re.escape(p) for p in patterns)
        keyword_pat = r"^\s*(" + escaped + r")\b"
        keyword_re = re.compile(keyword_pat, re.IGNORECASE)

        target = Path(self.target_dir)
        deadline = time.monotonic() + timeout
        timed_out = False
        files_scanned = 0
        for root, dirs, files in os.walk(target):
            # Check deadline at the start of each directory
            if time.monotonic() > deadline:
                timed_out = True
                break

            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for fname in files:
                # Check deadline periodically (every file)
                if time.monotonic() > deadline:
                    timed_out = True
                    break

                ext = Path(fname).suffix
                if ext not in _COMMENT_PREFIXES:
                    continue

                fpath = Path(root) / fname
                rel_path = str(fpath.relative_to(target))

                try:
                    content = fpath.read_text(errors="ignore")
                except OSError:
                    continue

                files_scanned += 1

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
                        context = self._read_file_snippet(str(fpath), i, context_lines=5)
                        tasks.append(Task(
                            description=desc,
                            priority=3,
                            source="todo",
                            source_file=rel_path,
                            line_number=i,
                            context=context,
                        ))

            if timed_out:
                break

        if timed_out:
            logger.warning(
                "TODO scan timed out after %ds (scanned %d files, found %d tasks)",
                timeout, files_scanned, len(tasks),
            )

        return tasks[:self.config.discovery.max_todo_tasks]

    def _discover_claude_ideas(self) -> List[Task]:
        """Use Claude to analyze the codebase and suggest improvement ideas.

        Invokes Claude CLI in read-only analysis mode with a low max-turns
        to keep cost down. Parses the response for actionable improvement tasks.
        """
        cc = self.config.claude
        if self.config.discovery.discovery_prompt:
            focus_section = self.config.discovery.discovery_prompt
        else:
            focus_section = (
                "- New features or functionality that would add value\n"
                "- Missing tests for important functionality\n"
                "- Bug risks or edge cases that could cause failures\n"
                "- Performance improvements\n"
                "- Code clarity or maintainability improvements\n"
            )

        history_section = ""
        if self.state_manager:
            recent = self.state_manager.get_recent_task_summaries(
                lookback_seconds=86400, max_items=15,
            )
            if recent:
                history_section = (
                    "\nRECENTLY COMPLETED TASKS (do NOT suggest similar improvements):\n"
                    + "\n".join(recent)
                    + "\n\nFocus on NEW areas that haven't been addressed recently.\n"
                )

        prompt = (
            "Analyze the codebase in the current directory. Identify up to 5 concrete, "
            "actionable improvements. Focus on:\n"
            + focus_section
            + history_section
            + "\nFor each improvement, output EXACTLY one line in this format:\n"
            "IDEA: <one-sentence description of the improvement>\n\n"
            "Do NOT make any changes. Do NOT run git commands. Only analyze and output IDEA lines."
        )

        cmd = [
            cc.command, "-p", prompt,
            "--model", self.config.discovery.discovery_model or self.config.claude.resolved_model,
            "--max-turns", str(self.config.discovery.discovery_max_turns),
            "--output-format", "json",
        ]

        try:
            result = run_with_group_kill(
                cmd,
                cwd=self.target_dir,
                timeout=self.config.discovery.discovery_timeout,
            )
        except (FileNotFoundError, OSError) as e:
            logger.warning("Claude idea discovery failed: %s", e)
            return []

        if result is None:
            logger.warning("Claude idea discovery returned no result")
            return []

        if result.timed_out:
            logger.warning("Claude idea discovery timed out after %ds",
                           self.config.discovery.discovery_timeout)
            return []

        if result.returncode != 0:
            logger.warning("Claude idea discovery exited with code %d", result.returncode)
            return []

        # Parse JSON response to get result text
        result_text = result.stdout
        try:
            # Helper: extract text content from a parsed JSON dict.
            # The Claude CLI uses "result" for successful runs and may use
            # "result_text" in some versions.
            def _extract_text(d: dict) -> str:
                for key in ("result", "result_text"):
                    val = d.get(key)
                    if isinstance(val, str) and val.strip():
                        return val
                return ""

            # Strategy 1: Try each line individually as a complete JSON object
            for line in result_text.strip().split("\n"):
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        # Check for error_max_turns — Claude ran out of turns
                        if data.get("subtype") == "error_max_turns":
                            logger.warning(
                                "Claude idea discovery hit max turns (%d); "
                                "increase discovery.discovery_max_turns",
                                self.config.discovery.discovery_max_turns,
                            )
                            text = _extract_text(data)
                            if text:
                                result_text = text
                            break
                        text = _extract_text(data)
                        if text:
                            result_text = text
                            break
                except (json.JSONDecodeError, TypeError):
                    continue
            else:
                # Strategy 2: Try multi-line JSON joining from each '{'-starting line
                lines = result_text.strip().split("\n")
                for i, line in enumerate(lines):
                    if not line.strip().startswith("{"):
                        continue
                    try:
                        data = json.loads("\n".join(lines[i:]))
                        if isinstance(data, dict):
                            if data.get("subtype") == "error_max_turns":
                                logger.warning(
                                    "Claude idea discovery hit max turns (%d); "
                                    "increase discovery.discovery_max_turns",
                                    self.config.discovery.discovery_max_turns,
                                )
                                text = _extract_text(data)
                                if text:
                                    result_text = text
                                break
                            text = _extract_text(data)
                            if text:
                                result_text = text
                                break
                    except (json.JSONDecodeError, TypeError):
                        continue
                else:
                    # Strategy 3: Use raw_decode to find JSON anywhere in the output
                    decoder = json.JSONDecoder()
                    for i, ch in enumerate(result_text):
                        if ch != "{":
                            continue
                        try:
                            obj, end = decoder.raw_decode(result_text, i)
                            if isinstance(obj, dict):
                                text = _extract_text(obj)
                                if text:
                                    result_text = text
                                    break
                        except (json.JSONDecodeError, TypeError):
                            continue
        except Exception:
            pass

        # Warn if we couldn't extract text from a JSON response
        if result_text is result.stdout and result.stdout.strip()[:1] in ("{", "["):
            logger.warning(
                "Claude idea response contained JSON but no text content "
                "could be extracted — falling back to raw text extraction"
            )

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
        if not tasks and result_text.strip():
            # Attempt fallback extraction from analysis-format text
            lines = [l.strip() for l in result_text.split("\n") if l.strip()]
            for line in lines:
                # Match numbered items: "1. ...", "2) ..."
                m = re.match(r'^\d+[.)]\s+(.+)', line)
                if not m:
                    # Match bullet items: "- ...", "* ...", "• ..."
                    m = re.match(r'^[-*•]\s+(.+)', line)
                if m:
                    desc = m.group(1).strip()
                    if desc and len(desc) >= MIN_IDEA_LENGTH:
                        tasks.append(Task(
                            description=desc,
                            priority=4,
                            source="claude_idea",
                        ))
            if tasks:
                logger.info(
                    "Extracted %d tasks from analysis-format text "
                    "(no IDEA: prefixed lines found, used numbered/bullet fallback).",
                    len(tasks),
                )
            else:
                logger.debug(
                    "Claude response contained no IDEA lines and no extractable "
                    "analysis-format items (%d non-empty lines)",
                    len(lines),
                )
        return tasks[:5]

    def _discover_coverage_gaps(self) -> List[Task]:
        """Discover files with low test coverage (optional, requires pytest-cov)."""
        try:
            result = run_with_group_kill(
                "python3 -m pytest --cov --cov-report=json --cov-report=term -q",
                shell=True,
                cwd=self.target_dir,
                timeout=self.config.validation.test_timeout,
            )
        except OSError:
            return []

        if result.timed_out:
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
