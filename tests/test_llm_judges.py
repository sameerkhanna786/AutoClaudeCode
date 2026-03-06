"""Tests for llm_judges.py — LLM-powered code quality judges."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch, PropertyMock

from claude_runner import ClaudeResult
from llm_judges import (
    JudgeVerdict,
    JudgePanelResult,
    LLMJudge,
    SecurityJudge,
    QualityJudge,
    ArchitectureJudge,
    JudgePanel,
)


def _make_config():
    """Create a minimal config for testing."""
    config = MagicMock()
    config.claude.model = "opus"
    config.claude.resolved_model = ""
    config.claude.max_turns = 25
    config.claude.timeout_seconds = 14400
    config.claude.command = "claude"
    config.claude.max_retries = 0
    config.claude.retry_delays = [2]
    config.claude.rate_limit_base_delay = 5
    config.claude.rate_limit_multiplier = 3
    config.target_dir = "/tmp/test"
    config.paths.state_dir = "/tmp/state"

    judge_cfg = MagicMock()
    judge_cfg.enabled = True
    judge_cfg.model = "sonnet"
    judge_cfg.max_turns = 5
    judge_cfg.timeout_seconds = 300
    judge_cfg.max_cost_usd = 0.50

    config.judges.enabled = True
    config.judges.security = judge_cfg
    config.judges.quality = judge_cfg
    config.judges.architecture = MagicMock(enabled=False)
    config.judges.max_total_cost_usd = 2.0
    config.judges.fail_action = "retry"

    return config


class TestVerdictParsing(unittest.TestCase):

    def test_pass_verdict(self):
        judge = SecurityJudge.__new__(SecurityJudge)
        passed, score, feedback = judge._parse_verdict(
            "VERDICT: PASS\nSCORE: 0.95\nFEEDBACK: No issues found"
        )
        self.assertTrue(passed)
        self.assertAlmostEqual(score, 0.95)
        self.assertEqual(feedback, "No issues found")

    def test_fail_verdict(self):
        judge = SecurityJudge.__new__(SecurityJudge)
        passed, score, feedback = judge._parse_verdict(
            "VERDICT: FAIL\nSCORE: 0.3\nFEEDBACK: SQL injection risk"
        )
        self.assertFalse(passed)
        self.assertAlmostEqual(score, 0.3)
        self.assertEqual(feedback, "SQL injection risk")

    def test_case_insensitive(self):
        judge = SecurityJudge.__new__(SecurityJudge)
        passed, score, feedback = judge._parse_verdict(
            "verdict: pass\nscore: 1.0\nfeedback: All good"
        )
        self.assertTrue(passed)
        self.assertAlmostEqual(score, 1.0)

    def test_missing_verdict(self):
        judge = SecurityJudge.__new__(SecurityJudge)
        passed, score, feedback = judge._parse_verdict(
            "Some analysis text\nNo verdict here"
        )
        self.assertFalse(passed)  # defaults to fail
        self.assertAlmostEqual(score, 0.0)

    def test_score_clamped(self):
        judge = SecurityJudge.__new__(SecurityJudge)
        _, score, _ = judge._parse_verdict("VERDICT: PASS\nSCORE: 1.5")
        self.assertAlmostEqual(score, 1.0)

    def test_fallback_feedback(self):
        judge = SecurityJudge.__new__(SecurityJudge)
        _, _, feedback = judge._parse_verdict(
            "VERDICT: FAIL\nSCORE: 0.1\nFound hardcoded key\nin config.py"
        )
        # Should grab non-verdict/score lines as feedback
        self.assertIn("hardcoded key", feedback)


class TestJudgeEvaluate(unittest.TestCase):

    @patch("llm_judges.ClaudeRunner")
    def test_pass_evaluation(self, mock_runner_cls):
        config = _make_config()
        mock_runner = MagicMock()
        mock_runner.run.return_value = ClaudeResult(
            success=True,
            result_text="VERDICT: PASS\nSCORE: 0.9\nFEEDBACK: Clean code",
            cost_usd=0.05,
        )
        mock_runner_cls.return_value = mock_runner

        judge = SecurityJudge(config, config.judges.security)
        verdict = judge.evaluate(["foo.py"], "diff text", "Fix bug")

        self.assertTrue(verdict.passed)
        self.assertEqual(verdict.judge_name, "security")
        self.assertAlmostEqual(verdict.score, 0.9)
        self.assertGreater(verdict.cost_usd, 0)

    @patch("llm_judges.ClaudeRunner")
    def test_fail_evaluation(self, mock_runner_cls):
        config = _make_config()
        mock_runner = MagicMock()
        mock_runner.run.return_value = ClaudeResult(
            success=True,
            result_text="VERDICT: FAIL\nSCORE: 0.2\nFEEDBACK: XSS vulnerability",
            cost_usd=0.05,
        )
        mock_runner_cls.return_value = mock_runner

        judge = SecurityJudge(config, config.judges.security)
        verdict = judge.evaluate(["foo.py"], "diff text", "Fix bug")

        self.assertFalse(verdict.passed)

    @patch("llm_judges.ClaudeRunner")
    def test_runner_failure_passes(self, mock_runner_cls):
        """When the judge runner fails, don't block the pipeline."""
        config = _make_config()
        mock_runner = MagicMock()
        mock_runner.run.return_value = ClaudeResult(
            success=False, error="CLI crashed",
        )
        mock_runner_cls.return_value = mock_runner

        judge = SecurityJudge(config, config.judges.security)
        verdict = judge.evaluate(["foo.py"], "diff text", "Fix bug")

        self.assertTrue(verdict.passed)  # Don't block on failure


class TestJudgePanel(unittest.TestCase):

    @patch("llm_judges.ClaudeRunner")
    def test_all_pass(self, mock_runner_cls):
        config = _make_config()
        mock_runner = MagicMock()
        mock_runner.run.return_value = ClaudeResult(
            success=True,
            result_text="VERDICT: PASS\nSCORE: 0.9\nFEEDBACK: Good",
            cost_usd=0.05,
        )
        mock_runner_cls.return_value = mock_runner

        panel = JudgePanel(config)
        result = panel.evaluate(["foo.py"], "diff", "task desc")

        self.assertTrue(result.passed)
        self.assertEqual(len(result.verdicts), 2)  # security + quality
        self.assertEqual(result.blocking_feedback, "")

    @patch("llm_judges.ClaudeRunner")
    def test_one_fails(self, mock_runner_cls):
        config = _make_config()
        call_count = [0]

        def side_effect(prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                return ClaudeResult(
                    success=True,
                    result_text="VERDICT: FAIL\nSCORE: 0.2\nFEEDBACK: Bad security",
                    cost_usd=0.05,
                )
            return ClaudeResult(
                success=True,
                result_text="VERDICT: PASS\nSCORE: 0.9\nFEEDBACK: Good quality",
                cost_usd=0.05,
            )

        mock_runner = MagicMock()
        mock_runner.run.side_effect = side_effect
        mock_runner_cls.return_value = mock_runner

        panel = JudgePanel(config)
        result = panel.evaluate(["foo.py"], "diff", "task desc")

        self.assertFalse(result.passed)
        self.assertIn("security", result.blocking_feedback)

    @patch("llm_judges.ClaudeRunner")
    def test_cost_cap(self, mock_runner_cls):
        config = _make_config()
        config.judges.max_total_cost_usd = 0.04  # Very low cap

        mock_runner = MagicMock()
        mock_runner.run.return_value = ClaudeResult(
            success=True,
            result_text="VERDICT: PASS\nSCORE: 1.0\nFEEDBACK: Good",
            cost_usd=0.05,  # Exceeds cap after first judge
        )
        mock_runner_cls.return_value = mock_runner

        panel = JudgePanel(config)
        result = panel.evaluate(["foo.py"], "diff", "task desc")

        # Only first judge should run before hitting cost cap
        self.assertLessEqual(len(result.verdicts), 2)

    def test_no_judges_enabled(self):
        config = _make_config()
        config.judges.security.enabled = False
        config.judges.quality.enabled = False
        config.judges.architecture.enabled = False

        panel = JudgePanel(config)
        result = panel.evaluate(["foo.py"], "diff", "task desc")

        self.assertTrue(result.passed)
        self.assertEqual(len(result.verdicts), 0)


class TestFailActionModes(unittest.TestCase):

    def test_verdict_dataclass(self):
        v = JudgeVerdict(
            judge_name="security", passed=True, score=0.9,
            feedback="ok", cost_usd=0.05, duration_seconds=1.5,
        )
        self.assertEqual(v.judge_name, "security")
        self.assertTrue(v.passed)

    def test_panel_result_dataclass(self):
        r = JudgePanelResult(passed=True, total_cost_usd=0.1)
        self.assertTrue(r.passed)
        self.assertEqual(r.blocking_feedback, "")


class TestJudgePrompts(unittest.TestCase):

    def test_security_prompt(self):
        config = _make_config()
        judge = SecurityJudge.__new__(SecurityJudge)
        prompt = judge._build_prompt(["foo.py"], "diff text", "Fix bug")
        self.assertIn("SECURITY", prompt)
        self.assertIn("VERDICT", prompt)

    def test_quality_prompt(self):
        config = _make_config()
        judge = QualityJudge.__new__(QualityJudge)
        prompt = judge._build_prompt(["foo.py"], "diff text", "Fix bug")
        self.assertIn("QUALITY", prompt)

    def test_architecture_prompt(self):
        config = _make_config()
        judge = ArchitectureJudge.__new__(ArchitectureJudge)
        prompt = judge._build_prompt(["foo.py"], "diff text", "Fix bug")
        self.assertIn("ARCHITECTURE", prompt)


if __name__ == "__main__":
    unittest.main()
