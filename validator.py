"""Run validation commands (tests, lint, build) and report results."""

from __future__ import annotations

import ast
import logging
import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

from config_schema import Config
from process_utils import run_with_group_kill

logger = logging.getLogger(__name__)


@dataclass
class ValidationStep:
    name: str
    command: str
    passed: bool
    output: str = ""
    return_code: int = 0


@dataclass
class ValidationResult:
    passed: bool
    steps: List[ValidationStep] = field(default_factory=list)

    @property
    def summary(self) -> str:
        parts = []
        for s in self.steps:
            status = "PASS" if s.passed else "FAIL"
            parts.append(f"{s.name}: {status}")
        return ", ".join(parts) if parts else "no validations run"


class Validator:
    def __init__(self, config: Config):
        self.config = config

    def _run_command(self, name: str, command: str, timeout: int, cwd: str) -> ValidationStep:
        """Run a single validation command."""
        if not command.strip():
            return ValidationStep(name=name, command="", passed=True, output="skipped")

        logger.info("Running %s: %s", name, command)
        try:
            result = run_with_group_kill(
                command,
                shell=True,
                cwd=cwd,
                timeout=timeout,
            )
            if result.timed_out:
                return ValidationStep(
                    name=name,
                    command=command,
                    passed=False,
                    output=f"Timed out after {timeout}s",
                    return_code=-1,
                )
            passed = result.returncode == 0
            output = result.stdout + result.stderr
            return ValidationStep(
                name=name,
                command=command,
                passed=passed,
                output=output.strip(),
                return_code=result.returncode,
            )
        except OSError as e:
            return ValidationStep(
                name=name,
                command=command,
                passed=False,
                output=str(e),
                return_code=-1,
            )
        except Exception as e:
            logger.warning("Unexpected error running %s: %s", name, e)
            return ValidationStep(
                name=name,
                command=command,
                passed=False,
                output=f"Unexpected error: {e}",
                return_code=-1,
            )

    def validate(self, working_dir: Optional[str] = None) -> ValidationResult:
        """Run test, lint, build commands sequentially.

        Short-circuits on first failure.
        """
        cwd = working_dir or self.config.target_dir
        vc = self.config.validation
        steps: List[ValidationStep] = []

        commands = [
            ("lint", vc.lint_command, vc.lint_timeout),
            ("tests", vc.test_command, vc.test_timeout),
            ("build", vc.build_command, vc.build_timeout),
        ]

        for name, command, timeout in commands:
            step = self._run_command(name, command, timeout, cwd)
            steps.append(step)
            if not step.passed and command.strip():
                logger.warning("%s failed (rc=%d)", name, step.return_code)
                return ValidationResult(passed=False, steps=steps)

        return ValidationResult(passed=True, steps=steps)

    def validate_syntax_only(
        self,
        changed_files: List[str],
        working_dir: Optional[str] = None,
    ) -> ValidationResult:
        """Fast pre-check: run ast.parse() on changed .py files.

        Catches syntax errors in <1 second without running the full test suite.
        """
        cwd = working_dir or self.config.target_dir
        errors: List[str] = []

        py_files = [f for f in changed_files if f.endswith(".py")]
        if not py_files:
            return ValidationResult(
                passed=True,
                steps=[ValidationStep(
                    name="syntax",
                    command="ast.parse()",
                    passed=True,
                    output="No .py files to check",
                )],
            )

        cwd_resolved = Path(cwd).resolve()
        for filepath in py_files:
            full_path = os.path.join(cwd, filepath)
            # Guard against path traversal (e.g. "../../../etc/passwd")
            try:
                Path(full_path).resolve().relative_to(cwd_resolved)
            except ValueError:
                logger.warning("Skipping file outside working dir: %s", filepath)
                continue
            if not os.path.isfile(full_path):
                continue
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    source = f.read()
                ast.parse(source, filename=filepath)
            except SyntaxError as e:
                errors.append(f"{filepath}:{e.lineno}: {e.msg}")
            except (OSError, UnicodeDecodeError) as e:
                logger.warning("Could not read %s for syntax check: %s", filepath, e)
            except Exception as e:
                logger.warning("Unexpected error parsing %s: %s", filepath, e)
                errors.append(f"{filepath}: parse error: {e}")

        if errors:
            output = "Syntax errors found:\n" + "\n".join(errors)
            logger.warning("Syntax check failed: %d error(s)", len(errors))
            return ValidationResult(
                passed=False,
                steps=[ValidationStep(
                    name="syntax",
                    command="ast.parse()",
                    passed=False,
                    output=output,
                )],
            )

        return ValidationResult(
            passed=True,
            steps=[ValidationStep(
                name="syntax",
                command="ast.parse()",
                passed=True,
                output=f"All {len(py_files)} file(s) passed syntax check",
            )],
        )

    # Regex to parse pytest FAILED lines, e.g.:
    # FAILED tests/test_foo.py::TestBar::test_baz - AssertionError: ...
    _FAILED_LINE_RE = re.compile(r'^FAILED\s+(\S+)', re.MULTILINE)

    def capture_baseline(self, working_dir: Optional[str] = None) -> Set[str]:
        """Run the test suite and return the set of already-failing test IDs.

        Returns an empty set if all tests pass or on parse error.
        """
        cwd = working_dir or self.config.target_dir
        vc = self.config.validation
        test_cmd = vc.test_command.strip()
        if not test_cmd:
            return set()

        # Append --tb=line for faster output (only for pytest-compatible commands)
        if "pytest" in test_cmd or "py.test" in test_cmd:
            baseline_cmd = f"{test_cmd} --tb=line"
        else:
            baseline_cmd = test_cmd
        step = self._run_command("baseline", baseline_cmd, vc.test_timeout, cwd)

        if step.passed:
            logger.info("Baseline: all tests pass")
            return set()

        failures = set(self._FAILED_LINE_RE.findall(step.output))
        if failures:
            logger.info("Baseline: %d pre-existing test failure(s)", len(failures))
        else:
            logger.warning(
                "Baseline: tests failed (rc=%d) but no FAILED lines parsed",
                step.return_code,
            )
        return failures

    def validate_with_baseline(
        self,
        working_dir: Optional[str] = None,
        baseline_failures: Optional[Set[str]] = None,
    ) -> ValidationResult:
        """Like validate(), but pre-existing test failures are ignored.

        If baseline_failures is None or empty, behaves identically to validate().
        """
        if not baseline_failures:
            return self.validate(working_dir)

        cwd = working_dir or self.config.target_dir
        vc = self.config.validation
        steps: List[ValidationStep] = []

        # Run lint first
        lint_step = self._run_command("lint", vc.lint_command, vc.lint_timeout, cwd)
        steps.append(lint_step)
        if not lint_step.passed and vc.lint_command.strip():
            logger.warning("lint failed (rc=%d)", lint_step.return_code)
            return ValidationResult(passed=False, steps=steps)

        # Run tests — strip -x when baseline is active so all failures are visible
        test_cmd = vc.test_command
        if baseline_failures and " -x " in f" {test_cmd} ":
            # Use regex to reliably remove standalone -x flag in any position
            test_cmd = re.sub(r'(?:^|\s)-x(?=\s|$)', ' ', test_cmd).strip()
            # Collapse any double spaces left behind
            test_cmd = re.sub(r'\s{2,}', ' ', test_cmd)
            logger.debug("Removed -x from test command for baseline comparison")
        test_step = self._run_command("tests", test_cmd, vc.test_timeout, cwd)
        if test_step.passed:
            steps.append(test_step)
        else:
            # Parse current failures and subtract baseline
            current_failures = set(self._FAILED_LINE_RE.findall(test_step.output))

            if not current_failures:
                # Tests failed but we couldn't parse any FAILED lines —
                # could be a collection error, import error, segfault, etc.
                # Treat as a genuine failure rather than silently passing.
                logger.warning(
                    "Tests failed (rc=%d) but no FAILED lines parsed, "
                    "treating as genuine failure",
                    test_step.return_code,
                )
                steps.append(test_step)
                return ValidationResult(passed=False, steps=steps)

            new_failures = current_failures - baseline_failures
            ignored = current_failures & baseline_failures

            if ignored:
                logger.warning(
                    "Ignoring %d pre-existing test failure(s): %s",
                    len(ignored), sorted(ignored),
                )

            if new_failures:
                # Real new failures — report as FAIL
                logger.warning(
                    "New test failures (not in baseline): %s",
                    sorted(new_failures),
                )
                steps.append(test_step)
                return ValidationResult(passed=False, steps=steps)
            else:
                # All failures are pre-existing — treat as PASS
                logger.info(
                    "All %d test failure(s) are pre-existing, treating tests as passed",
                    len(current_failures),
                )
                baseline_note = (
                    f"NOTE: All {len(current_failures)} failure(s) are "
                    f"pre-existing baseline failures\n\n"
                )
                steps.append(ValidationStep(
                    name="tests",
                    command=test_step.command,
                    passed=True,
                    output=baseline_note + test_step.output,
                    return_code=test_step.return_code,
                ))

        # Run build
        build_step = self._run_command("build", vc.build_command, vc.build_timeout, cwd)
        steps.append(build_step)
        if not build_step.passed and vc.build_command.strip():
            logger.warning("build failed (rc=%d)", build_step.return_code)
            return ValidationResult(passed=False, steps=steps)

        return ValidationResult(passed=True, steps=steps)

    def _map_changed_to_test_files(
        self, changed_files: List[str], working_dir: str,
    ) -> List[str]:
        """Map changed .py files to their likely test files.

        For each changed file like `foo.py`, looks for `test_foo.py` or
        `tests/test_foo.py`. Also includes any changed test files directly.
        Returns test file paths relative to working_dir.
        """
        test_files = []
        wd = Path(working_dir)

        for f in changed_files:
            if not f.endswith(".py"):
                continue

            p = Path(f)
            basename = p.stem  # e.g., "foo" from "foo.py"

            # If it's already a test file, include it directly
            if basename.startswith("test_"):
                if (wd / f).exists():
                    test_files.append(f)
                continue

            # Look for test_<basename>.py in common locations
            candidates = [
                p.parent / f"test_{basename}.py",
                Path("tests") / f"test_{basename}.py",
                Path("test") / f"test_{basename}.py",
            ]

            for candidate in candidates:
                full = wd / candidate
                if full.exists():
                    test_files.append(str(candidate))
                    break

        return list(set(test_files))

    def validate_incremental(
        self, working_dir: str, changed_files: List[str],
    ) -> ValidationResult:
        """Run targeted tests for changed files, then full suite on success.

        1. Maps changed files to test files.
        2. Runs only those tests first (fast feedback).
        3. If targeted tests pass, runs the full test suite as final check.
        4. Falls back to full validation if no test files can be mapped.
        """
        cwd = working_dir or self.config.target_dir
        vc = self.config.validation

        if not vc.test_command.strip():
            return self.validate(working_dir)

        test_files = self._map_changed_to_test_files(changed_files, cwd)
        if not test_files:
            # Can't determine affected tests, fall back to full suite
            return self.validate(working_dir)

        # Phase 1: Run targeted tests using the configured test command
        base_test_cmd = vc.test_command.strip()
        quoted_files = ' '.join(shlex.quote(f) for f in test_files)
        # Only append pytest-specific flags (-x -q) when the test command is pytest-based
        is_pytest = "pytest" in base_test_cmd
        targeted_cmd = f"{base_test_cmd} {quoted_files} -x -q" if is_pytest else f"{base_test_cmd} {quoted_files}"
        steps: List[ValidationStep] = []

        # Run lint first
        lint_step = self._run_command("lint", vc.lint_command, vc.lint_timeout, cwd)
        steps.append(lint_step)
        if not lint_step.passed and vc.lint_command.strip():
            return ValidationResult(passed=False, steps=steps)

        # Run targeted tests
        targeted_step = self._run_command(
            "tests (targeted)", targeted_cmd, vc.test_timeout, cwd,
        )
        steps.append(targeted_step)
        if not targeted_step.passed:
            return ValidationResult(passed=False, steps=steps)

        # Phase 2: Run full test suite to catch regressions
        full_step = self._run_command(
            "tests (full)", vc.test_command, vc.test_timeout, cwd,
        )
        steps.append(full_step)
        if not full_step.passed:
            return ValidationResult(passed=False, steps=steps)

        # Run build
        build_step = self._run_command("build", vc.build_command, vc.build_timeout, cwd)
        steps.append(build_step)
        if not build_step.passed and vc.build_command.strip():
            return ValidationResult(passed=False, steps=steps)

        return ValidationResult(passed=True, steps=steps)
