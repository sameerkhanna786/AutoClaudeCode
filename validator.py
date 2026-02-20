"""Run validation commands (tests, lint, build) and report results."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
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
