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


class TestParseReviewVerdict:
    def setup_method(self):
        self.config = Config()
        self.pipeline = AgentPipeline(self.config)

    def test_approved(self):
        assert self.pipeline._parse_review_verdict("VERDICT: APPROVED\nLooks good.") is True

    def test_revise(self):
        assert self.pipeline._parse_review_verdict("VERDICT: REVISE\nNeeds fixes.") is False

    def test_no_verdict_defaults_approved(self):
        assert self.pipeline._parse_review_verdict("Some review text without verdict") is True

    def test_empty_string_defaults_approved(self):
        assert self.pipeline._parse_review_verdict("") is True

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

    @patch("agent_pipeline.ClaudeRunner")
    def test_full_pipeline_approved_first_pass(self, MockRunner, tmp_path):
        self.config.target_dir = str(tmp_path)
        pipeline = AgentPipeline(self.config)
        rollback_fn = MagicMock()
        ws_dir = Path(str(tmp_path)) / self.config.paths.agent_workspace_dir

        runner_instance = MockRunner.return_value

        def side_effect_fn(prompt, working_dir):
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

    @patch("agent_pipeline.ClaudeRunner")
    def test_planner_failure_stops_pipeline(self, MockRunner, tmp_path):
        self.config.target_dir = str(tmp_path)
        pipeline = AgentPipeline(self.config)
        rollback_fn = MagicMock()

        runner_instance = MockRunner.return_value
        runner_instance.run.return_value = _make_failure_result("planner error")

        result = pipeline.run([MockTask()], rollback_fn, "snap")

        assert result.success is False
        assert "Planner failed" in result.error
        assert len(result.agent_results) == 1

    @patch("agent_pipeline.ClaudeRunner")
    def test_coder_failure_stops_pipeline(self, MockRunner, tmp_path):
        self.config.target_dir = str(tmp_path)
        pipeline = AgentPipeline(self.config)
        rollback_fn = MagicMock()

        call_count = {"n": 0}
        runner_instance = MockRunner.return_value

        def side_effect_fn(prompt, working_dir):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_success_result("plan")
            return _make_failure_result("coder error")

        runner_instance.run.side_effect = side_effect_fn

        result = pipeline.run([MockTask()], rollback_fn, "snap")

        assert result.success is False
        assert "Coder failed" in result.error

    @patch("agent_pipeline.ClaudeRunner")
    def test_tester_disabled_skipped(self, MockRunner, tmp_path):
        self.config.target_dir = str(tmp_path)
        self.config.agent_pipeline.tester.enabled = False
        pipeline = AgentPipeline(self.config)
        rollback_fn = MagicMock()
        ws_dir = Path(str(tmp_path)) / self.config.paths.agent_workspace_dir

        runner_instance = MockRunner.return_value

        def side_effect_fn(prompt, working_dir):
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

    @patch("agent_pipeline.ClaudeRunner")
    def test_reviewer_disabled_auto_approve(self, MockRunner, tmp_path):
        self.config.target_dir = str(tmp_path)
        self.config.agent_pipeline.reviewer.enabled = False
        pipeline = AgentPipeline(self.config)
        rollback_fn = MagicMock()

        runner_instance = MockRunner.return_value
        runner_instance.run.return_value = _make_success_result("output")

        result = pipeline.run([MockTask()], rollback_fn, "snap")

        assert result.success is True
        assert result.final_review_approved is True

    @patch("agent_pipeline.ClaudeRunner")
    def test_revision_loop(self, MockRunner, tmp_path):
        """Reviewer rejects first, approves second attempt."""
        self.config.target_dir = str(tmp_path)
        self.config.agent_pipeline.max_revisions = 2
        pipeline = AgentPipeline(self.config)
        rollback_fn = MagicMock()
        ws_dir = Path(str(tmp_path)) / self.config.paths.agent_workspace_dir

        reviewer_count = {"n": 0}
        runner_instance = MockRunner.return_value

        def side_effect_fn(prompt, working_dir):
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

    @patch("agent_pipeline.ClaudeRunner")
    def test_max_revisions_exhausted(self, MockRunner, tmp_path):
        """All revisions exhausted -> success=False, review_approved=False."""
        self.config.target_dir = str(tmp_path)
        self.config.agent_pipeline.max_revisions = 1
        pipeline = AgentPipeline(self.config)
        rollback_fn = MagicMock()
        ws_dir = Path(str(tmp_path)) / self.config.paths.agent_workspace_dir

        runner_instance = MockRunner.return_value

        def side_effect_fn(prompt, working_dir):
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

    @patch("agent_pipeline.ClaudeRunner")
    def test_git_rollback_called_between_revisions(self, MockRunner, tmp_path):
        """Verify git rollback is called after planner and between revisions."""
        self.config.target_dir = str(tmp_path)
        self.config.agent_pipeline.max_revisions = 1
        pipeline = AgentPipeline(self.config)
        rollback_fn = MagicMock()
        ws_dir = Path(str(tmp_path)) / self.config.paths.agent_workspace_dir

        reviewer_count = {"n": 0}
        runner_instance = MockRunner.return_value

        def side_effect_fn(prompt, working_dir):
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

    @patch("agent_pipeline.ClaudeRunner")
    def test_cost_accumulation(self, MockRunner, tmp_path):
        """Total cost should accumulate across all agents."""
        self.config.target_dir = str(tmp_path)
        self.config.agent_pipeline.reviewer.enabled = False
        pipeline = AgentPipeline(self.config)
        rollback_fn = MagicMock()

        costs = [0.05, 0.15, 0.08]
        durations = [2.0, 8.0, 4.0]
        call_idx = {"n": 0}
        runner_instance = MockRunner.return_value

        def side_effect_fn(prompt, working_dir):
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

    @patch("agent_pipeline.ClaudeRunner")
    def test_multiple_tasks(self, MockRunner, tmp_path):
        """Pipeline should handle batch tasks."""
        self.config.target_dir = str(tmp_path)
        self.config.agent_pipeline.reviewer.enabled = False
        pipeline = AgentPipeline(self.config)
        rollback_fn = MagicMock()

        runner_instance = MockRunner.return_value
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
        """When resolved_model is set, agent runners should inherit it."""
        self.config.target_dir = str(tmp_path)
        self.config.claude.resolved_model = "claude-opus-4-6"
        pipeline = AgentPipeline(self.config)

        runner = pipeline._build_runner_for_agent(AgentRole.PLANNER)
        assert runner.config.claude.resolved_model == "claude-opus-4-6"
        cmd = runner._build_command("test prompt")
        assert "claude-opus-4-6" in cmd
