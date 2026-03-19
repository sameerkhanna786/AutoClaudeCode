"""LLM Judges: independent AI quality evaluation beyond 'tests pass'."""

from __future__ import annotations

import copy
import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

from claude_runner import ClaudeResult
from config_schema import Config
from provider_runner import create_runner

logger = logging.getLogger(__name__)


@dataclass
class JudgeVerdict:
    judge_name: str       # "security", "quality", "architecture"
    passed: bool
    score: float          # 0.0-1.0
    feedback: str         # explanation for retry prompt
    cost_usd: float
    duration_seconds: float


@dataclass
class JudgePanelResult:
    passed: bool
    verdicts: List[JudgeVerdict] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_duration_seconds: float = 0.0
    blocking_feedback: str = ""  # concatenated feedback from failing judges


class LLMJudge:
    """Base class for LLM-based code judges.

    Uses ClaudeRunner with a cheap model (sonnet by default) to evaluate
    code changes independently of test results.
    """

    judge_name: str = "base"

    def __init__(self, config: Config, judge_config):
        self.config = config
        self.judge_config = judge_config
        self._runner = self._build_runner()

    def _build_runner(self):
        """Create a runner configured for judge evaluation.

        Uses create_runner() to respect the configured provider (Claude,
        OpenAI, Gemini) instead of always using ClaudeRunner.
        """
        judge_config = copy.deepcopy(self.config)
        judge_config.claude.model = self.judge_config.model
        judge_config.claude.resolved_model = ""
        judge_config.claude.max_turns = self.judge_config.max_turns
        judge_config.claude.timeout_seconds = self.judge_config.timeout_seconds
        return create_runner(judge_config)

    def _build_prompt(self, changed_files: List[str], diff_text: str,
                      task_description: str) -> str:
        """Build the evaluation prompt. Subclasses override this."""
        raise NotImplementedError

    def _parse_verdict(self, result_text: str) -> tuple:
        """Parse VERDICT: PASS/FAIL, score, and feedback from result text.

        Returns (passed: bool, score: float, feedback: str).
        """
        passed = False
        score = 0.0
        feedback = ""

        for line in result_text.splitlines():
            line_stripped = line.strip()
            verdict_match = re.match(
                r"VERDICT:\s*(PASS|FAIL)", line_stripped, re.IGNORECASE,
            )
            if verdict_match:
                passed = verdict_match.group(1).upper() == "PASS"

            score_match = re.match(
                r"SCORE:\s*([0-9]*\.?[0-9]+)", line_stripped, re.IGNORECASE,
            )
            if score_match:
                try:
                    score = min(1.0, max(0.0, float(score_match.group(1))))
                except ValueError:
                    pass

            feedback_match = re.match(
                r"FEEDBACK:\s*(.+)", line_stripped, re.IGNORECASE,
            )
            if feedback_match:
                feedback = feedback_match.group(1).strip()

        # If no explicit feedback line, grab all non-verdict/score lines
        if not feedback:
            feedback_lines = []
            for line in result_text.splitlines():
                line_stripped = line.strip()
                if line_stripped and not line_stripped.upper().startswith(("VERDICT:", "SCORE:")):
                    feedback_lines.append(line_stripped)
            feedback = "\n".join(feedback_lines[-5:])  # Last 5 lines

        return passed, score, feedback

    def evaluate(self, changed_files: List[str], diff_text: str,
                 task_description: str) -> JudgeVerdict:
        """Run the judge evaluation and return a verdict."""
        start = time.time()
        prompt = self._build_prompt(changed_files, diff_text, task_description)

        result = self._runner.run(prompt)
        duration = time.time() - start

        if not result.success:
            logger.warning(
                "%s judge failed: %s", self.judge_name, result.error,
            )
            return JudgeVerdict(
                judge_name=self.judge_name,
                passed=True,  # Don't block on judge failure
                score=0.0,
                feedback=f"Judge failed: {result.error}",
                cost_usd=result.cost_usd,
                duration_seconds=duration,
            )

        passed, score, feedback = self._parse_verdict(result.result_text)

        return JudgeVerdict(
            judge_name=self.judge_name,
            passed=passed,
            score=score,
            feedback=feedback,
            cost_usd=result.cost_usd,
            duration_seconds=duration,
        )


class SecurityJudge(LLMJudge):
    """Checks for security issues: secrets, injection, auth bypass."""

    judge_name = "security"

    def _build_prompt(self, changed_files: List[str], diff_text: str,
                      task_description: str) -> str:
        files_list = ", ".join(changed_files[:20])
        return (
            "You are a SECURITY REVIEWER. Analyze the following code changes "
            "for security vulnerabilities.\n\n"
            f"TASK: {task_description}\n\n"
            f"CHANGED FILES: {files_list}\n\n"
            f"DIFF:\n```\n{diff_text[:8000]}\n```\n\n"
            "Check for:\n"
            "- Hardcoded secrets, API keys, or credentials\n"
            "- SQL injection, command injection, or XSS vulnerabilities\n"
            "- Authentication or authorization bypass\n"
            "- Insecure file operations or path traversal\n"
            "- Sensitive data exposure in logs or error messages\n\n"
            "Respond with:\n"
            "VERDICT: PASS (if no security issues) or VERDICT: FAIL (if issues found)\n"
            "SCORE: 0.0-1.0 (1.0 = perfectly secure)\n"
            "FEEDBACK: <explanation of any issues found>\n"
        )


class QualityJudge(LLMJudge):
    """Checks code quality: test slop, dead code, naming."""

    judge_name = "quality"

    def _build_prompt(self, changed_files: List[str], diff_text: str,
                      task_description: str) -> str:
        files_list = ", ".join(changed_files[:20])
        return (
            "You are a CODE QUALITY REVIEWER. Analyze the following code changes "
            "for quality issues.\n\n"
            f"TASK: {task_description}\n\n"
            f"CHANGED FILES: {files_list}\n\n"
            f"DIFF:\n```\n{diff_text[:8000]}\n```\n\n"
            "Check for:\n"
            "- Test slop (tests that don't actually test anything meaningful)\n"
            "- Dead code or unreachable branches\n"
            "- Poor naming conventions or unclear variable names\n"
            "- Missing error handling for important edge cases\n"
            "- Code duplication that should be refactored\n\n"
            "Respond with:\n"
            "VERDICT: PASS (if quality is acceptable) or VERDICT: FAIL (if issues found)\n"
            "SCORE: 0.0-1.0 (1.0 = excellent quality)\n"
            "FEEDBACK: <explanation of any issues found>\n"
        )


class ArchitectureJudge(LLMJudge):
    """Checks architecture: circular deps, layer violations."""

    judge_name = "architecture"

    def _build_prompt(self, changed_files: List[str], diff_text: str,
                      task_description: str) -> str:
        files_list = ", ".join(changed_files[:20])
        return (
            "You are an ARCHITECTURE REVIEWER. Analyze the following code changes "
            "for architectural issues.\n\n"
            f"TASK: {task_description}\n\n"
            f"CHANGED FILES: {files_list}\n\n"
            f"DIFF:\n```\n{diff_text[:8000]}\n```\n\n"
            "Check for:\n"
            "- Circular dependencies between modules\n"
            "- Layer violations (e.g., data layer accessing UI layer)\n"
            "- God classes or modules with too many responsibilities\n"
            "- Tight coupling that reduces testability\n"
            "- Breaking changes to public interfaces\n\n"
            "Respond with:\n"
            "VERDICT: PASS (if architecture is sound) or VERDICT: FAIL (if issues found)\n"
            "SCORE: 0.0-1.0 (1.0 = excellent architecture)\n"
            "FEEDBACK: <explanation of any issues found>\n"
        )


# Map judge names to classes
_JUDGE_CLASSES = {
    "security": SecurityJudge,
    "quality": QualityJudge,
    "architecture": ArchitectureJudge,
}


class JudgePanel:
    """Runs a panel of LLM judges and aggregates results."""

    def __init__(self, config: Config):
        self.config = config
        self._judges: List[LLMJudge] = []

        judges_config = config.judges
        for name, judge_cls in _JUDGE_CLASSES.items():
            judge_cfg = getattr(judges_config, name, None)
            if judge_cfg and judge_cfg.enabled:
                self._judges.append(judge_cls(config, judge_cfg))

    def evaluate(self, changed_files: List[str], diff_text: str,
                 task_description: str) -> JudgePanelResult:
        """Run all enabled judges sequentially, short-circuiting on cost cap."""
        verdicts: List[JudgeVerdict] = []
        total_cost = 0.0
        total_duration = 0.0
        max_cost = self.config.judges.max_total_cost_usd

        for judge in self._judges:
            # Cost cap check
            if total_cost >= max_cost:
                logger.warning(
                    "Judge panel cost cap reached ($%.2f >= $%.2f), "
                    "skipping remaining judges",
                    total_cost, max_cost,
                )
                break

            verdict = judge.evaluate(changed_files, diff_text, task_description)
            verdicts.append(verdict)
            total_cost += verdict.cost_usd
            total_duration += verdict.duration_seconds

            logger.info(
                "Judge %s: %s (score=%.2f, cost=$%.4f)",
                verdict.judge_name,
                "PASS" if verdict.passed else "FAIL",
                verdict.score,
                verdict.cost_usd,
            )

        # Aggregate results
        all_passed = all(v.passed for v in verdicts)
        blocking_feedback = "\n\n".join(
            f"[{v.judge_name}] {v.feedback}"
            for v in verdicts if not v.passed
        )

        return JudgePanelResult(
            passed=all_passed,
            verdicts=verdicts,
            total_cost_usd=total_cost,
            total_duration_seconds=total_duration,
            blocking_feedback=blocking_feedback,
        )
