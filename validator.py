"""Run validation commands (tests, lint, build) and report results."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

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

        # Phase 1: Run targeted tests
        targeted_cmd = f"python3 -m pytest {' '.join(shlex.quote(f) for f in test_files)} -x -q"
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
