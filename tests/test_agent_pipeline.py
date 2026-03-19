"""Tests for agent_pipeline module."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from agent_pipeline import (
    AgentPipeline,
    AgentResult,
    AgentRole,
    AgentWorkspace,
    PipelineResult,
    ReviewFinding,
    ReviewReport,
)
from claude_runner import ClaudeResult
from config_schema import AgentPipelineConfig, AgentRoleConfig, Config


class TestAgentWorkspace:
    def test_write_and_read(self, tmp_path):
        ws = AgentWorkspace(str(tmp_path / "workspace"))
        ws.clean()
        ws.write("plan.md", "my plan")
        assert ws.read("plan.md") == "my plan"

    def test_read_missing_returns_none(self, tmp_path):
        ws = AgentWorkspace(str(tmp_path / "workspace"))
        ws.clean()
        assert ws.read("nonexistent.md") is None

    def test_clean_removes_files(self, tmp_path):
        ws = AgentWorkspace(str(tmp_path / "workspace"))
        ws.clean()
        ws.write("plan.md", "content")
        assert ws.exists("plan.md")
        ws.clean()
        assert not ws.exists("plan.md")

    def test_exists(self, tmp_path):
        ws = AgentWorkspace(str(tmp_path / "workspace"))
        ws.clean()
        assert not ws.exists("plan.md")
        ws.write("plan.md", "x")
        assert ws.exists("plan.md")

    def test_write_creates_dirs(self, tmp_path):
        ws = AgentWorkspace(str(tmp_path / "deep" / "nested" / "workspace"))
        ws.write("test.md", "content")
        assert ws.read("test.md") == "content"

    def test_path_traversal_write_blocked(self, tmp_path):
        ws = AgentWorkspace(str(tmp_path / "workspace"))
        ws.clean()
        with pytest.raises(ValueError, match="Path traversal blocked"):
            ws.write("../../etc/evil.txt", "malicious")

    def test_path_traversal_read_blocked(self, tmp_path):
        ws = AgentWorkspace(str(tmp_path / "workspace"))
        ws.clean()
        with pytest.raises(ValueError, match="Path traversal blocked"):
            ws.read("../../../etc/passwd")

    def test_path_traversal_exists_blocked(self, tmp_path):
        ws = AgentWorkspace(str(tmp_path / "workspace"))
        ws.clean()
        with pytest.raises(ValueError, match="Path traversal blocked"):
            ws.exists("../../etc/passwd")

    def test_normal_filenames_allowed(self, tmp_path):
        ws = AgentWorkspace(str(tmp_path / "workspace"))
        ws.clean()
        # Normal filenames within workspace should work fine
        ws.write("plan.md", "content")
        assert ws.read("plan.md") == "content"
        assert ws.exists("plan.md")


class TestParseReviewVerdict:
    def setup_method(self):
        self.config = Config()
        self.pipeline = AgentPipeline(self.config)

    def test_approved(self):
        assert self.pipeline._parse_review_verdict("VERDICT: APPROVED\nLooks good.") is True

    def test_revise(self):
        assert self.pipeline._parse_review_verdict("VERDICT: REVISE\nNeeds fixes.") is False

    def test_no_verdict_defaults_revise(self):
        assert self.pipeline._parse_review_verdict("Some review text without verdict") is False

    def test_empty_string_defaults_revise(self):
        assert self.pipeline._parse_review_verdict("") is False

    def test_case_insensitive_approved(self):
        assert self.pipeline._parse_review_verdict("verdict: approved\nGood work.") is True

    def test_case_insensitive_revise(self):
        assert self.pipeline._parse_review_verdict("Verdict: Revise\nFix things.") is False

    def test_verdict_not_on_first_line(self):
        content = "Some intro text\nVERDICT: REVISE\nDetails here."
        assert self.pipeline._parse_review_verdict(content) is False

    def test_verdict_with_extra_whitespace(self):
        assert self.pipeline._parse_review_verdict("  VERDICT:   APPROVED  \nOK.") is True


@dataclass
class MockTask:
    description: str = "Fix the bug"
    priority: int = 1
    source: str = "test"
    source_file: str = ""
    task_key: str = "test:fix_the_bug"


def _make_success_result(text="done"):
    return ClaudeResult(success=True, result_text=text, cost_usd=0.10, duration_seconds=5.0)


def _make_failure_result(error="failed"):
    return ClaudeResult(success=False, error=error)


class TestAgentPipelineFlow:
    def setup_method(self):
        self.config = Config()
        self.config.agent_pipeline.enabled = True

    @patch("provider_runner.create_runner")
    def test_full_pipeline_approved_first_pass(self, mock_create_runner, tmp_path):
        self.config.target_dir = str(tmp_path)
        pipeline = AgentPipeline(self.config)
        rollback_fn = MagicMock()
        ws_dir = Path(str(tmp_path)) / self.config.paths.agent_workspace_dir

        runner_instance = mock_create_runner.return_value

        def side_effect_fn(prompt):
            """Write review.md when reviewer runs."""
            if "REVIEWER" in prompt:
                ws_dir.mkdir(parents=True, exist_ok=True)
                (ws_dir / "review.md").write_text("VERDICT: APPROVED\nAll good.")
                return _make_success_result("review output")
            return _make_success_result("output")

        runner_instance.run.side_effect = side_effect_fn

        result = pipeline.run([MockTask()], rollback_fn, "snapshot123")

        assert result.success is True
        assert result.final_review_approved is True
        assert result.revision_count == 0
        assert len(result.agent_results) == 4
        assert result.total_cost_usd == pytest.approx(0.40)

    @patch("provider_runner.create_runner")
    def test_planner_failure_stops_pipeline(self, mock_create_runner, tmp_path):
        self.config.target_dir = str(tmp_path)
        pipeline = AgentPipeline(self.config)
        rollback_fn = MagicMock()

        runner_instance = mock_create_runner.return_value
        runner_instance.run.return_value = _make_failure_result("planner error")

        result = pipeline.run([MockTask()], rollback_fn, "snap")

        assert result.success is False
        assert "Planner failed" in result.error
        assert len(result.agent_results) == 1

    @patch("provider_runner.create_runner")
    def test_coder_failure_stops_pipeline(self, mock_create_runner, tmp_path):
        self.config.target_dir = str(tmp_path)
        pipeline = AgentPipeline(self.config)
        rollback_fn = MagicMock()

        call_count = {"n": 0}
        runner_instance = mock_create_runner.return_value

        def side_effect_fn(prompt):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_success_result("plan")
            return _make_failure_result("coder error")

        runner_instance.run.side_effect = side_effect_fn

        result = pipeline.run([MockTask()], rollback_fn, "snap")

        assert result.success is False
        assert "Coder failed" in result.error

    @patch("provider_runner.create_runner")
    def test_tester_disabled_skipped(self, mock_create_runner, tmp_path):
        self.config.target_dir = str(tmp_path)
        self.config.agent_pipeline.tester.enabled = False
        pipeline = AgentPipeline(self.config)
        rollback_fn = MagicMock()
        ws_dir = Path(str(tmp_path)) / self.config.paths.agent_workspace_dir

        runner_instance = mock_create_runner.return_value

        def side_effect_fn(prompt):
            if "REVIEWER" in prompt:
                ws_dir.mkdir(parents=True, exist_ok=True)
                (ws_dir / "review.md").write_text("VERDICT: APPROVED\nOK")
                return _make_success_result("review")
            return _make_success_result("output")

        runner_instance.run.side_effect = side_effect_fn

        result = pipeline.run([MockTask()], rollback_fn, "snap")

        assert result.success is True
        # planner + coder + tester(skipped) + reviewer = 4
        assert len(result.agent_results) == 4
        tester_results = [r for r in result.agent_results if r.role == AgentRole.TESTER]
        assert len(tester_results) == 1
        assert tester_results[0].output_text == "(skipped)"

    @patch("provider_runner.create_runner")
    def test_reviewer_disabled_auto_approve(self, mock_create_runner, tmp_path):
        self.config.target_dir = str(tmp_path)
        self.config.agent_pipeline.reviewer.enabled = False
        pipeline = AgentPipeline(self.config)
        rollback_fn = MagicMock()

        runner_instance = mock_create_runner.return_value
        runner_instance.run.return_value = _make_success_result("output")

        result = pipeline.run([MockTask()], rollback_fn, "snap")

        assert result.success is True
        assert result.final_review_approved is True

    @patch("provider_runner.create_runner")
    def test_revision_loop(self, mock_create_runner, tmp_path):
        """Reviewer rejects first, approves second attempt."""
        self.config.target_dir = str(tmp_path)
        self.config.agent_pipeline.max_revisions = 2
        pipeline = AgentPipeline(self.config)
        rollback_fn = MagicMock()
        ws_dir = Path(str(tmp_path)) / self.config.paths.agent_workspace_dir

        reviewer_count = {"n": 0}
        runner_instance = mock_create_runner.return_value

        def side_effect_fn(prompt):
            if "REVIEWER" in prompt:
                reviewer_count["n"] += 1
                ws_dir.mkdir(parents=True, exist_ok=True)
                if reviewer_count["n"] == 1:
                    (ws_dir / "review.md").write_text("VERDICT: REVISE\nFix the naming.")
                else:
                    (ws_dir / "review.md").write_text("VERDICT: APPROVED\nLooks good now.")
                return _make_success_result("review")
            return _make_success_result("output")

        runner_instance.run.side_effect = side_effect_fn

        result = pipeline.run([MockTask()], rollback_fn, "snap")

        assert result.success is True
        assert result.final_review_approved is True
        assert result.revision_count == 1
        # Rollback: once after planner, once before revision retry
        assert rollback_fn.call_count == 2

    @patch("provider_runner.create_runner")
    def test_max_revisions_exhausted(self, mock_create_runner, tmp_path):
        """All revisions exhausted -> success=False, review_approved=False."""
        self.config.target_dir = str(tmp_path)
        self.config.agent_pipeline.max_revisions = 1
        pipeline = AgentPipeline(self.config)
        rollback_fn = MagicMock()
        ws_dir = Path(str(tmp_path)) / self.config.paths.agent_workspace_dir

        runner_instance = mock_create_runner.return_value

        def side_effect_fn(prompt):
            if "REVIEWER" in prompt:
                ws_dir.mkdir(parents=True, exist_ok=True)
                (ws_dir / "review.md").write_text("VERDICT: REVISE\nStill issues.")
                return _make_success_result("review")
            return _make_success_result("output")

        runner_instance.run.side_effect = side_effect_fn

        result = pipeline.run([MockTask()], rollback_fn, "snap")

        assert result.success is False
        assert result.final_review_approved is False
        assert result.revision_count == 1
        assert "rejected" in result.error.lower() or "exhausting" in result.error.lower()

    @patch("provider_runner.create_runner")
    def test_git_rollback_called_between_revisions(self, mock_create_runner, tmp_path):
        """Verify git rollback is called after planner and between revisions."""
        self.config.target_dir = str(tmp_path)
        self.config.agent_pipeline.max_revisions = 1
        pipeline = AgentPipeline(self.config)
        rollback_fn = MagicMock()
        ws_dir = Path(str(tmp_path)) / self.config.paths.agent_workspace_dir

        reviewer_count = {"n": 0}
        runner_instance = mock_create_runner.return_value

        def side_effect_fn(prompt):
            if "REVIEWER" in prompt:
                reviewer_count["n"] += 1
                ws_dir.mkdir(parents=True, exist_ok=True)
                if reviewer_count["n"] == 1:
                    (ws_dir / "review.md").write_text("VERDICT: REVISE\nFix it.")
                else:
                    (ws_dir / "review.md").write_text("VERDICT: APPROVED\nOK.")
                return _make_success_result("review")
            return _make_success_result("output")

        runner_instance.run.side_effect = side_effect_fn

        result = pipeline.run([MockTask()], rollback_fn, "snap")

        # Rollback: after planner + before revision retry
        assert rollback_fn.call_count == 2
        rollback_fn.assert_any_call("snap")

    def test_per_agent_model_overrides(self, tmp_path):
        """Each agent should get its own model config (no mock on ClaudeRunner)."""
        self.config.target_dir = str(tmp_path)
        pipeline = AgentPipeline(self.config)

        planner_runner = pipeline._build_runner_for_agent(AgentRole.PLANNER)
        assert planner_runner.config.claude.model == "opus"
        assert planner_runner.config.claude.max_turns == 10

        coder_runner = pipeline._build_runner_for_agent(AgentRole.CODER)
        assert coder_runner.config.claude.model == "opus"
        assert coder_runner.config.claude.max_turns == 25

        tester_runner = pipeline._build_runner_for_agent(AgentRole.TESTER)
        assert tester_runner.config.claude.model == "opus"
        assert tester_runner.config.claude.max_turns == 15

        reviewer_runner = pipeline._build_runner_for_agent(AgentRole.REVIEWER)
        assert reviewer_runner.config.claude.model == "opus"
        assert reviewer_runner.config.claude.max_turns == 10

    @patch("provider_runner.create_runner")
    def test_cost_accumulation(self, mock_create_runner, tmp_path):
        """Total cost should accumulate across all agents."""
        self.config.target_dir = str(tmp_path)
        self.config.agent_pipeline.reviewer.enabled = False
        pipeline = AgentPipeline(self.config)
        rollback_fn = MagicMock()

        costs = [0.05, 0.15, 0.08]
        durations = [2.0, 8.0, 4.0]
        call_idx = {"n": 0}
        runner_instance = mock_create_runner.return_value

        def side_effect_fn(prompt):
            idx = call_idx["n"]
            call_idx["n"] += 1
            return ClaudeResult(
                success=True, result_text="output",
                cost_usd=costs[idx], duration_seconds=durations[idx],
            )

        runner_instance.run.side_effect = side_effect_fn

        result = pipeline.run([MockTask()], rollback_fn, "snap")

        assert result.success is True
        assert result.total_cost_usd == pytest.approx(0.28)
        assert result.total_duration_seconds == pytest.approx(14.0)

    @patch("provider_runner.create_runner")
    def test_multiple_tasks(self, mock_create_runner, tmp_path):
        """Pipeline should handle batch tasks."""
        self.config.target_dir = str(tmp_path)
        self.config.agent_pipeline.reviewer.enabled = False
        pipeline = AgentPipeline(self.config)
        rollback_fn = MagicMock()

        runner_instance = mock_create_runner.return_value
        runner_instance.run.return_value = _make_success_result("output")

        tasks = [
            MockTask(description="Task A"),
            MockTask(description="Task B"),
        ]

        result = pipeline.run(tasks, rollback_fn, "snap")
        assert result.success is True

        # Verify the prompt included both tasks
        first_call_prompt = runner_instance.run.call_args_list[0][0][0]
        assert "Task A" in first_call_prompt
        assert "Task B" in first_call_prompt

    def test_resolved_model_propagated_to_agent_runner(self, tmp_path):
        """Agent runners should use their role-specific model, not the parent's resolved_model."""
        self.config.target_dir = str(tmp_path)
        self.config.claude.resolved_model = "claude-opus-4-6"
        self.config.agent_pipeline.planner.model = "haiku"
        pipeline = AgentPipeline(self.config)

        runner = pipeline._build_runner_for_agent(AgentRole.PLANNER)
        # resolved_model should be cleared so the agent's own model alias is used
        assert runner.config.claude.resolved_model == ""
        assert runner.config.claude.model == "haiku"
        cmd = runner._build_command("test prompt")
        assert "haiku" in cmd
        assert "claude-opus-4-6" not in cmd


class TestPipelineCostGuard:
    """Test that the pipeline aborts when accumulated cost exceeds the limit."""

    @patch("provider_runner.create_runner")
    def test_cost_guard_aborts_revisions(self, mock_create_runner, tmp_path):
        config = Config()
        config.target_dir = str(tmp_path)
        config.agent_pipeline.enabled = True
        config.agent_pipeline.max_revisions = 5
        config.agent_pipeline.max_pipeline_cost_usd = 0.50
        pipeline = AgentPipeline(config)
        rollback_fn = MagicMock()
        ws_dir = Path(str(tmp_path)) / config.paths.agent_workspace_dir

        call_idx = {"n": 0}
        runner_instance = mock_create_runner.return_value

        def side_effect_fn(prompt):
            call_idx["n"] += 1
            if "REVIEWER" in prompt:
                ws_dir.mkdir(parents=True, exist_ok=True)
                (ws_dir / "review.md").write_text("VERDICT: REVISE\nNeeds work.")
                return ClaudeResult(success=True, result_text="review",
                                    cost_usd=0.15, duration_seconds=2.0)
            return ClaudeResult(success=True, result_text="output",
                                cost_usd=0.15, duration_seconds=2.0)

        runner_instance.run.side_effect = side_effect_fn

        result = pipeline.run([MockTask()], rollback_fn, "snap")

        assert result.success is False
        assert "cost limit" in result.error.lower()

    @patch("provider_runner.create_runner")
    def test_cost_guard_uses_safety_default(self, mock_create_runner, tmp_path):
        """When max_pipeline_cost_usd is 0, uses safety.max_cost_usd_per_hour * 0.5."""
        config = Config()
        config.target_dir = str(tmp_path)
        config.agent_pipeline.enabled = True
        config.agent_pipeline.max_pipeline_cost_usd = 0.0  # use default
        config.safety.max_cost_usd_per_hour = 2.0  # effective limit = 1.0
        config.agent_pipeline.max_revisions = 10
        pipeline = AgentPipeline(config)
        rollback_fn = MagicMock()
        ws_dir = Path(str(tmp_path)) / config.paths.agent_workspace_dir

        runner_instance = mock_create_runner.return_value

        def side_effect_fn(prompt):
            if "REVIEWER" in prompt:
                ws_dir.mkdir(parents=True, exist_ok=True)
                (ws_dir / "review.md").write_text("VERDICT: REVISE\nKeep going.")
                return ClaudeResult(success=True, result_text="review",
                                    cost_usd=0.30, duration_seconds=2.0)
            return ClaudeResult(success=True, result_text="output",
                                cost_usd=0.30, duration_seconds=2.0)

        runner_instance.run.side_effect = side_effect_fn

        result = pipeline.run([MockTask()], rollback_fn, "snap")

        assert result.success is False
        assert "cost limit" in result.error.lower()


class TestBuildRunnerValidation:
    """Test that _build_runner_for_agent validates role config attributes."""

    def test_build_runner_missing_model_raises(self, tmp_path):
        config = Config()
        config.target_dir = str(tmp_path)
        # Replace planner config with an object missing 'model'
        config.agent_pipeline.planner = type("BadCfg", (), {
            "max_turns": 10, "timeout_seconds": 300, "enabled": True,
        })()
        pipeline = AgentPipeline(config)
        with pytest.raises(ValueError, match="model"):
            pipeline._build_runner_for_agent(AgentRole.PLANNER)

    def test_build_runner_missing_timeout_raises(self, tmp_path):
        config = Config()
        config.target_dir = str(tmp_path)
        # Replace coder config with an object missing 'timeout_seconds'
        config.agent_pipeline.coder = type("BadCfg", (), {
            "model": "opus", "max_turns": 25, "enabled": True,
        })()
        pipeline = AgentPipeline(config)
        with pytest.raises(ValueError, match="timeout_seconds"):
            pipeline._build_runner_for_agent(AgentRole.CODER)

    def test_build_runner_missing_multiple_attrs_raises(self, tmp_path):
        config = Config()
        config.target_dir = str(tmp_path)
        # Replace reviewer config with a bare object missing all required attrs
        config.agent_pipeline.reviewer = type("BadCfg", (), {"enabled": True})()
        pipeline = AgentPipeline(config)
        with pytest.raises(ValueError, match="model.*max_turns.*timeout_seconds"):
            pipeline._build_runner_for_agent(AgentRole.REVIEWER)


class TestParseStructuredReview:
    """Test structured review JSON parsing."""

    def setup_method(self):
        self.config = Config()
        self.pipeline = AgentPipeline(self.config)

    def test_parse_valid_structured_review(self):
        review_text = '''Some preamble text.

```json
{
  "approved": false,
  "summary": "Needs fixes",
  "findings": [
    {
      "filepath": "foo.py",
      "line_number": 10,
      "severity": "error",
      "category": "quality",
      "description": "Missing null check",
      "suggestion": "Add if x is not None",
      "confidence": 0.95
    },
    {
      "filepath": "bar.py",
      "line_number": 20,
      "severity": "warning",
      "category": "security",
      "description": "Unvalidated input",
      "suggestion": "Sanitize user input",
      "confidence": 0.80
    }
  ],
  "test_suggestions": [
    "Test with empty input",
    "Test with unicode characters"
  ]
}
```

VERDICT: REVISE
'''
        report = self.pipeline._parse_structured_review(review_text)
        assert report is not None
        assert report.approved is False
        assert report.summary == "Needs fixes"
        assert len(report.findings) == 2
        assert report.findings[0].filepath == "foo.py"
        assert report.findings[0].line_number == 10
        assert report.findings[0].severity == "error"
        assert report.findings[0].category == "quality"
        assert report.findings[0].confidence == 0.95
        assert report.findings[1].severity == "warning"
        assert len(report.test_suggestions) == 2
        assert "empty input" in report.test_suggestions[0]

    def test_parse_approved_structured_review(self):
        review_text = '''```json
{
  "approved": true,
  "summary": "All good",
  "findings": [],
  "test_suggestions": []
}
```

VERDICT: APPROVED
'''
        report = self.pipeline._parse_structured_review(review_text)
        assert report is not None
        assert report.approved is True
        assert len(report.findings) == 0

    def test_parse_no_json_block_returns_none(self):
        """Fall back to legacy parsing when no JSON block present."""
        review_text = "VERDICT: APPROVED\nLooks good."
        report = self.pipeline._parse_structured_review(review_text)
        assert report is None

    def test_parse_invalid_json_returns_none(self):
        review_text = '''```json
{invalid json here}
```
VERDICT: REVISE
'''
        report = self.pipeline._parse_structured_review(review_text)
        assert report is None

    def test_parse_json_without_findings_key_returns_none(self):
        review_text = '''```json
{"summary": "no findings key"}
```
VERDICT: REVISE
'''
        report = self.pipeline._parse_structured_review(review_text)
        assert report is None

    def test_parse_empty_string_returns_none(self):
        assert self.pipeline._parse_structured_review("") is None

    def test_approved_falls_back_to_verdict_line(self):
        """When approved is missing from JSON, fall back to VERDICT line."""
        review_text = '''```json
{
  "findings": [],
  "summary": "Looks fine"
}
```
VERDICT: APPROVED
'''
        report = self.pipeline._parse_structured_review(review_text)
        assert report is not None
        assert report.approved is True

    def test_malformed_line_number_and_confidence(self):
        """Non-numeric line_number/confidence from LLM should not crash."""
        review_text = '''```json
{
  "approved": false,
  "summary": "Issues",
  "findings": [
    {
      "filepath": "foo.py",
      "line_number": "N/A",
      "severity": "error",
      "category": "bug",
      "description": "Something wrong",
      "suggestion": "Fix it",
      "confidence": "high"
    }
  ]
}
```'''
        report = self.pipeline._parse_structured_review(review_text)
        assert report is not None
        assert len(report.findings) == 1
        assert report.findings[0].line_number == 0
        assert report.findings[0].confidence == 0.8


class TestFilterFindings:
    """Test confidence-based filtering and deduplication."""

    def test_filter_by_confidence_error(self):
        findings = [
            ReviewFinding(filepath="a.py", line_number=1, severity="error",
                          confidence=0.90, description="high conf error"),
            ReviewFinding(filepath="b.py", line_number=2, severity="error",
                          confidence=0.50, description="low conf error"),
        ]
        filtered = AgentPipeline._filter_findings(findings)
        assert len(filtered) == 1
        assert filtered[0].description == "high conf error"

    def test_filter_by_confidence_warning(self):
        findings = [
            ReviewFinding(filepath="a.py", line_number=1, severity="warning",
                          confidence=0.75, description="ok warning"),
            ReviewFinding(filepath="b.py", line_number=2, severity="warning",
                          confidence=0.50, description="low warning"),
        ]
        filtered = AgentPipeline._filter_findings(findings)
        assert len(filtered) == 1
        assert filtered[0].description == "ok warning"

    def test_filter_by_confidence_information(self):
        findings = [
            ReviewFinding(filepath="a.py", line_number=1, severity="information",
                          confidence=0.30, description="info ok"),
            ReviewFinding(filepath="b.py", line_number=2, severity="information",
                          confidence=0.10, description="info too low"),
        ]
        filtered = AgentPipeline._filter_findings(findings)
        assert len(filtered) == 1
        assert filtered[0].description == "info ok"

    def test_deduplication_by_file_and_line(self):
        findings = [
            ReviewFinding(filepath="a.py", line_number=10, severity="warning",
                          confidence=0.90, description="first"),
            ReviewFinding(filepath="a.py", line_number=10, severity="error",
                          confidence=0.95, description="duplicate"),
        ]
        filtered = AgentPipeline._filter_findings(findings)
        assert len(filtered) == 1
        assert filtered[0].description == "first"

    def test_no_dedup_when_line_is_zero(self):
        findings = [
            ReviewFinding(filepath="a.py", line_number=0, severity="warning",
                          confidence=0.90, description="general1"),
            ReviewFinding(filepath="a.py", line_number=0, severity="warning",
                          confidence=0.90, description="general2"),
        ]
        filtered = AgentPipeline._filter_findings(findings)
        assert len(filtered) == 2

    def test_custom_confidence_threshold(self):
        findings = [
            ReviewFinding(filepath="a.py", line_number=1, severity="warning",
                          confidence=0.85, description="above"),
            ReviewFinding(filepath="b.py", line_number=2, severity="warning",
                          confidence=0.79, description="below"),
        ]
        filtered = AgentPipeline._filter_findings(findings, confidence_threshold=0.80)
        assert len(filtered) == 1
        assert filtered[0].description == "above"

    def test_error_threshold_is_at_least_085(self):
        """Error threshold should be max(0.85, custom_threshold)."""
        findings = [
            ReviewFinding(filepath="a.py", line_number=1, severity="error",
                          confidence=0.82, description="below 0.85"),
        ]
        # Even with custom threshold=0.50, errors need >= 0.85
        filtered = AgentPipeline._filter_findings(findings, confidence_threshold=0.50)
        assert len(filtered) == 0


class TestFormatStructuredFeedback:
    """Test formatting of structured review for coder prompt."""

    def test_format_groups_by_file(self):
        report = ReviewReport(
            approved=False,
            findings=[
                ReviewFinding(filepath="a.py", line_number=10, severity="error",
                              description="Bug here", suggestion="Fix it"),
                ReviewFinding(filepath="a.py", line_number=20, severity="warning",
                              description="Style issue"),
                ReviewFinding(filepath="b.py", line_number=5, severity="warning",
                              description="Another issue"),
            ],
        )
        text = AgentPipeline._format_structured_feedback(report)
        assert "### a.py" in text
        assert "### b.py" in text
        assert "[ERROR]" in text
        assert "[WARNING]" in text
        assert "Fix it" in text

    def test_format_skips_information_severity(self):
        report = ReviewReport(
            approved=False,
            findings=[
                ReviewFinding(filepath="a.py", severity="information",
                              description="FYI note"),
            ],
        )
        text = AgentPipeline._format_structured_feedback(report)
        assert "FYI note" not in text

    def test_format_empty_findings_returns_summary(self):
        report = ReviewReport(approved=False, summary="All good", findings=[])
        text = AgentPipeline._format_structured_feedback(report)
        assert text == "All good"

    def test_format_no_findings_no_summary(self):
        report = ReviewReport(approved=False, findings=[])
        text = AgentPipeline._format_structured_feedback(report)
        assert text == "No actionable findings."


class TestFormatTestSuggestions:
    """Test formatting of test suggestions for tester prompt."""

    def test_format_test_suggestions(self):
        report = ReviewReport(
            approved=False,
            findings=[],
            test_suggestions=["Test empty input", "Test large files"],
        )
        text = AgentPipeline._format_test_suggestions(report)
        assert "Test empty input" in text
        assert "Test large files" in text
        assert "1." in text
        assert "2." in text

    def test_format_testing_category_findings(self):
        report = ReviewReport(
            approved=False,
            findings=[
                ReviewFinding(filepath="a.py", line_number=10,
                              category="testing", description="No edge case test"),
            ],
        )
        text = AgentPipeline._format_test_suggestions(report)
        assert "No edge case test" in text
        assert "a.py:10" in text

    def test_format_empty_suggestions_and_no_testing_findings(self):
        report = ReviewReport(approved=False, findings=[])
        text = AgentPipeline._format_test_suggestions(report)
        assert text == ""


class TestStructuredReviewInPipeline:
    """Integration tests for structured review flowing through the pipeline."""

    def setup_method(self):
        self.config = Config()
        self.config.agent_pipeline.enabled = True

    @patch("provider_runner.create_runner")
    def test_structured_review_approved(self, mock_create_runner, tmp_path):
        """Structured review with approved=true should succeed."""
        self.config.target_dir = str(tmp_path)
        pipeline = AgentPipeline(self.config)
        rollback_fn = MagicMock()
        ws_dir = Path(str(tmp_path)) / self.config.paths.agent_workspace_dir

        runner_instance = mock_create_runner.return_value

        def side_effect_fn(prompt):
            if "REVIEWER" in prompt:
                ws_dir.mkdir(parents=True, exist_ok=True)
                (ws_dir / "review.md").write_text(
                    '```json\n'
                    '{"approved": true, "summary": "LGTM", "findings": [], '
                    '"test_suggestions": []}\n'
                    '```\n\nVERDICT: APPROVED'
                )
                return _make_success_result("review")
            return _make_success_result("output")

        runner_instance.run.side_effect = side_effect_fn
        result = pipeline.run([MockTask()], rollback_fn, "snap")
        assert result.success is True
        assert result.final_review_approved is True

    @patch("provider_runner.create_runner")
    def test_structured_review_feeds_back_to_coder(self, mock_create_runner, tmp_path):
        """On revision, structured findings should appear in coder prompt."""
        self.config.target_dir = str(tmp_path)
        self.config.agent_pipeline.max_revisions = 1
        pipeline = AgentPipeline(self.config)
        rollback_fn = MagicMock()
        ws_dir = Path(str(tmp_path)) / self.config.paths.agent_workspace_dir

        reviewer_count = {"n": 0}
        coder_prompts = []
        runner_instance = mock_create_runner.return_value

        def side_effect_fn(prompt):
            if "CODER" in prompt:
                coder_prompts.append(prompt)
                return _make_success_result("code")
            if "REVIEWER" in prompt:
                reviewer_count["n"] += 1
                ws_dir.mkdir(parents=True, exist_ok=True)
                if reviewer_count["n"] == 1:
                    (ws_dir / "review.md").write_text(
                        '```json\n'
                        '{"approved": false, "summary": "Needs work", '
                        '"findings": [{"filepath": "x.py", "line_number": 5, '
                        '"severity": "error", "category": "quality", '
                        '"description": "Missing check", '
                        '"suggestion": "Add validation", "confidence": 0.95}], '
                        '"test_suggestions": ["Test null input"]}\n'
                        '```\n\nVERDICT: REVISE'
                    )
                else:
                    (ws_dir / "review.md").write_text(
                        '```json\n'
                        '{"approved": true, "summary": "Fixed", "findings": []}\n'
                        '```\n\nVERDICT: APPROVED'
                    )
                return _make_success_result("review")
            return _make_success_result("output")

        runner_instance.run.side_effect = side_effect_fn
        result = pipeline.run([MockTask()], rollback_fn, "snap")

        assert result.success is True
        assert len(coder_prompts) == 2
        # The second coder prompt should contain structured feedback
        assert "Missing check" in coder_prompts[1]
        assert "Add validation" in coder_prompts[1]
        assert "[ERROR]" in coder_prompts[1]

    @patch("provider_runner.create_runner")
    def test_legacy_verdict_still_works(self, mock_create_runner, tmp_path):
        """Old-style VERDICT: APPROVED without JSON should still work."""
        self.config.target_dir = str(tmp_path)
        pipeline = AgentPipeline(self.config)
        rollback_fn = MagicMock()
        ws_dir = Path(str(tmp_path)) / self.config.paths.agent_workspace_dir

        runner_instance = mock_create_runner.return_value

        def side_effect_fn(prompt):
            if "REVIEWER" in prompt:
                ws_dir.mkdir(parents=True, exist_ok=True)
                (ws_dir / "review.md").write_text("VERDICT: APPROVED\nAll good.")
                return _make_success_result("review")
            return _make_success_result("output")

        runner_instance.run.side_effect = side_effect_fn
        result = pipeline.run([MockTask()], rollback_fn, "snap")
        assert result.success is True
        assert result.final_review_approved is True

    @patch("provider_runner.create_runner")
    def test_test_suggestions_forwarded_to_tester(self, mock_create_runner, tmp_path):
        """Test suggestions from review should appear in tester prompt on revision."""
        self.config.target_dir = str(tmp_path)
        self.config.agent_pipeline.max_revisions = 1
        pipeline = AgentPipeline(self.config)
        rollback_fn = MagicMock()
        ws_dir = Path(str(tmp_path)) / self.config.paths.agent_workspace_dir

        reviewer_count = {"n": 0}
        tester_prompts = []
        runner_instance = mock_create_runner.return_value

        def side_effect_fn(prompt):
            if "TESTER" in prompt:
                tester_prompts.append(prompt)
                return _make_success_result("tests pass")
            if "REVIEWER" in prompt:
                reviewer_count["n"] += 1
                ws_dir.mkdir(parents=True, exist_ok=True)
                if reviewer_count["n"] == 1:
                    (ws_dir / "review.md").write_text(
                        '```json\n'
                        '{"approved": false, "summary": "Add tests", '
                        '"findings": [{"filepath": "a.py", "line_number": 1, '
                        '"severity": "warning", "category": "testing", '
                        '"description": "Untested edge case", '
                        '"confidence": 0.90}], '
                        '"test_suggestions": ["Test with empty list"]}\n'
                        '```\n\nVERDICT: REVISE'
                    )
                else:
                    (ws_dir / "review.md").write_text(
                        '```json\n'
                        '{"approved": true, "findings": []}\n'
                        '```\n\nVERDICT: APPROVED'
                    )
                return _make_success_result("review")
            return _make_success_result("output")

        runner_instance.run.side_effect = side_effect_fn
        result = pipeline.run([MockTask()], rollback_fn, "snap")

        assert result.success is True
        assert len(tester_prompts) == 2
        # Second tester prompt should include test suggestions
        assert "Test with empty list" in tester_prompts[1]
        assert "Untested edge case" in tester_prompts[1]


class TestSkipPlanningFor:
    def setup_method(self):
        self.config = Config()
        self.config.agent_pipeline.enabled = True

    @patch("provider_runner.create_runner")
    def test_skip_planner_for_lint_task(self, mock_create_runner, tmp_path):
        """Planner is skipped when all tasks have source in skip_planning_for."""
        self.config.target_dir = str(tmp_path)
        self.config.agent_pipeline.skip_planning_for = ["lint", "todo"]
        pipeline = AgentPipeline(self.config)
        rollback_fn = MagicMock()
        ws_dir = Path(str(tmp_path)) / self.config.paths.agent_workspace_dir

        runner_instance = mock_create_runner.return_value
        prompts_seen = []

        def side_effect_fn(prompt):
            prompts_seen.append(prompt)
            if "REVIEWER" in prompt:
                ws_dir.mkdir(parents=True, exist_ok=True)
                (ws_dir / "review.md").write_text("VERDICT: APPROVED\nOK")
                return _make_success_result("review output")
            return _make_success_result("output")

        runner_instance.run.side_effect = side_effect_fn

        lint_task = MockTask(description="Fix unused import", source="lint")
        result = pipeline.run([lint_task], rollback_fn, "snap")

        assert result.success is True
        # No planner prompt should have been sent
        planner_prompts = [p for p in prompts_seen if "PLANNER" in p]
        assert len(planner_prompts) == 0
        # Coder + Tester + Reviewer = 3 agent calls
        assert len(prompts_seen) == 3
        # Rollback should NOT be called for planner (no planner ran)
        assert rollback_fn.call_count == 0

    @patch("provider_runner.create_runner")
    def test_planner_runs_for_non_skip_task(self, mock_create_runner, tmp_path):
        """Planner runs normally when task source is not in skip_planning_for."""
        self.config.target_dir = str(tmp_path)
        self.config.agent_pipeline.skip_planning_for = ["lint", "todo"]
        pipeline = AgentPipeline(self.config)
        rollback_fn = MagicMock()
        ws_dir = Path(str(tmp_path)) / self.config.paths.agent_workspace_dir

        runner_instance = mock_create_runner.return_value
        prompts_seen = []

        def side_effect_fn(prompt):
            prompts_seen.append(prompt)
            if "REVIEWER" in prompt:
                ws_dir.mkdir(parents=True, exist_ok=True)
                (ws_dir / "review.md").write_text("VERDICT: APPROVED\nOK")
                return _make_success_result("review output")
            return _make_success_result("output")

        runner_instance.run.side_effect = side_effect_fn

        test_task = MockTask(description="Fix test failure", source="test_failure")
        result = pipeline.run([test_task], rollback_fn, "snap")

        assert result.success is True
        # Planner prompt should have been sent
        planner_prompts = [p for p in prompts_seen if "PLANNER" in p]
        assert len(planner_prompts) == 1
        # Planner + Coder + Tester + Reviewer = 4 agent calls
        assert len(prompts_seen) == 4

    @patch("provider_runner.create_runner")
    def test_mixed_sources_runs_planner(self, mock_create_runner, tmp_path):
        """Planner runs when tasks have mixed sources (not all skippable)."""
        self.config.target_dir = str(tmp_path)
        self.config.agent_pipeline.skip_planning_for = ["lint", "todo"]
        pipeline = AgentPipeline(self.config)
        rollback_fn = MagicMock()
        ws_dir = Path(str(tmp_path)) / self.config.paths.agent_workspace_dir

        runner_instance = mock_create_runner.return_value
        prompts_seen = []

        def side_effect_fn(prompt):
            prompts_seen.append(prompt)
            if "REVIEWER" in prompt:
                ws_dir.mkdir(parents=True, exist_ok=True)
                (ws_dir / "review.md").write_text("VERDICT: APPROVED\nOK")
                return _make_success_result("review output")
            return _make_success_result("output")

        runner_instance.run.side_effect = side_effect_fn

        tasks = [
            MockTask(description="Fix lint", source="lint"),
            MockTask(description="Fix test", source="test_failure"),
        ]
        result = pipeline.run(tasks, rollback_fn, "snap")

        assert result.success is True
        planner_prompts = [p for p in prompts_seen if "PLANNER" in p]
        assert len(planner_prompts) == 1

    @patch("provider_runner.create_runner")
    def test_empty_skip_list_runs_planner(self, mock_create_runner, tmp_path):
        """Planner runs when skip_planning_for is empty."""
        self.config.target_dir = str(tmp_path)
        self.config.agent_pipeline.skip_planning_for = []
        pipeline = AgentPipeline(self.config)
        rollback_fn = MagicMock()
        ws_dir = Path(str(tmp_path)) / self.config.paths.agent_workspace_dir

        runner_instance = mock_create_runner.return_value
        prompts_seen = []

        def side_effect_fn(prompt):
            prompts_seen.append(prompt)
            if "REVIEWER" in prompt:
                ws_dir.mkdir(parents=True, exist_ok=True)
                (ws_dir / "review.md").write_text("VERDICT: APPROVED\nOK")
                return _make_success_result("review output")
            return _make_success_result("output")

        runner_instance.run.side_effect = side_effect_fn

        lint_task = MockTask(description="Fix lint", source="lint")
        result = pipeline.run([lint_task], rollback_fn, "snap")

        assert result.success is True
        planner_prompts = [p for p in prompts_seen if "PLANNER" in p]
        assert len(planner_prompts) == 1
