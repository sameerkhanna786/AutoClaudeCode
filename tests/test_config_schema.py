"""Tests for config_schema module."""

import tempfile
from pathlib import Path

import pytest

from config_schema import Config, load_config


class TestConfigDefaults:
    def test_default_config_has_expected_values(self):
        config = Config()
        assert config.target_dir == "."
        assert config.claude.model == "opus"
        assert config.claude.max_turns == 25
        assert config.orchestrator.loop_interval_seconds == 30
        assert config.safety.max_consecutive_failures == 5
        assert config.validation.test_command == "python3 -m pytest tests/ -x -q"

    def test_default_protected_files(self):
        config = Config()
        assert "main.py" in config.safety.protected_files
        assert "config.yaml" in config.safety.protected_files

    def test_default_discovery_patterns(self):
        config = Config()
        assert "TODO" in config.discovery.todo_patterns
        assert "FIXME" in config.discovery.todo_patterns


class TestLoadConfig:
    def test_load_config_none_returns_defaults(self):
        config = load_config(None)
        assert config.target_dir == "."
        assert config.claude.model == "opus"

    def test_load_config_missing_file_returns_defaults(self):
        config = load_config("/nonexistent/path.yaml")
        assert config.target_dir == "."

    def test_load_config_empty_file_returns_defaults(self, tmp_path):
        f = tmp_path / "empty.yaml"
        f.write_text("")
        config = load_config(str(f))
        assert config.target_dir == "."

    def test_load_config_partial_override(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text(
            "target_dir: /my/project\n"
            "claude:\n"
            "  model: opus\n"
            "  max_turns: 10\n"
        )
        config = load_config(str(f))
        assert config.target_dir == "/my/project"
        assert config.claude.model == "opus"
        assert config.claude.max_turns == 10
        # Unset values keep defaults
        assert config.claude.timeout_seconds == 300
        assert config.orchestrator.loop_interval_seconds == 30

    def test_load_config_full_file(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text(
            "target_dir: /proj\n"
            "safety:\n"
            "  max_consecutive_failures: 3\n"
            "  protected_files:\n"
            "    - main.py\n"
            "    - config.yaml\n"
            "    - secret.py\n"
        )
        config = load_config(str(f))
        assert config.target_dir == "/proj"
        assert config.safety.max_consecutive_failures == 3
        assert "secret.py" in config.safety.protected_files

    def test_load_config_ignores_unknown_keys(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text("unknown_key: value\nclaude:\n  model: haiku\n")
        config = load_config(str(f))
        assert config.claude.model == "haiku"


class TestNewConfigFields:
    def test_discovery_model_default(self):
        config = Config()
        assert config.discovery.discovery_model == "opus"

    def test_resolved_model_default(self):
        config = Config()
        assert config.claude.resolved_model == ""

    def test_discovery_timeout_default(self):
        config = Config()
        assert config.discovery.discovery_timeout == 180

    def test_max_feedback_retries_default(self):
        config = Config()
        assert config.orchestrator.max_feedback_retries == 3

    def test_feedback_failed_dir_default(self):
        config = Config()
        assert config.paths.feedback_failed_dir == "feedback/failed"

    def test_discovery_model_override(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text("discovery:\n  discovery_model: haiku\n  discovery_timeout: 300\n")
        config = load_config(str(f))
        assert config.discovery.discovery_model == "haiku"
        assert config.discovery.discovery_timeout == 300

    def test_max_feedback_retries_override(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text("orchestrator:\n  max_feedback_retries: 5\n")
        config = load_config(str(f))
        assert config.orchestrator.max_feedback_retries == 5

    def test_feedback_failed_dir_override(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text("paths:\n  feedback_failed_dir: custom/failed\n")
        config = load_config(str(f))
        assert config.paths.feedback_failed_dir == "custom/failed"


class TestBatchConfigFields:
    def test_max_tasks_per_cycle_default(self):
        config = Config()
        assert config.orchestrator.max_tasks_per_cycle == 10

    def test_batch_mode_default(self):
        config = Config()
        assert config.orchestrator.batch_mode is True

    def test_max_tasks_per_cycle_override(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text("orchestrator:\n  max_tasks_per_cycle: 5\n")
        config = load_config(str(f))
        assert config.orchestrator.max_tasks_per_cycle == 5

    def test_batch_mode_override(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text("orchestrator:\n  batch_mode: false\n")
        config = load_config(str(f))
        assert config.orchestrator.batch_mode is False


class TestAdaptiveBatchConfig:
    def test_adaptive_batch_defaults(self):
        config = Config()
        assert config.orchestrator.initial_batch_size == 3
        assert config.orchestrator.min_batch_size == 1
        assert config.orchestrator.max_batch_size == 10
        assert config.orchestrator.batch_grow_step == 1
        assert config.orchestrator.batch_shrink_step == 2
        assert config.orchestrator.adaptive_batch_window == 10

    def test_adaptive_batch_override(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text(
            "orchestrator:\n"
            "  min_batch_size: 2\n"
            "  max_batch_size: 20\n"
            "  initial_batch_size: 5\n"
            "  batch_grow_step: 2\n"
            "  batch_shrink_step: 3\n"
            "  adaptive_batch_window: 15\n"
        )
        config = load_config(str(f))
        assert config.orchestrator.min_batch_size == 2
        assert config.orchestrator.max_batch_size == 20
        assert config.orchestrator.initial_batch_size == 5
        assert config.orchestrator.batch_grow_step == 2
        assert config.orchestrator.batch_shrink_step == 3
        assert config.orchestrator.adaptive_batch_window == 15

    def test_legacy_max_tasks_per_cycle_migrated(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text("orchestrator:\n  max_tasks_per_cycle: 7\n")
        config = load_config(str(f))
        assert config.orchestrator.max_batch_size == 7

    def test_legacy_migration_not_applied_when_max_batch_size_set(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text(
            "orchestrator:\n"
            "  max_tasks_per_cycle: 7\n"
            "  max_batch_size: 15\n"
        )
        config = load_config(str(f))
        assert config.orchestrator.max_batch_size == 15


class TestAgentPipelineConfig:
    def test_default_agent_pipeline_disabled(self):
        config = Config()
        assert config.agent_pipeline.enabled is False

    def test_default_max_revisions(self):
        config = Config()
        assert config.agent_pipeline.max_revisions == 2

    def test_default_planner_config(self):
        config = Config()
        assert config.agent_pipeline.planner.model == "opus"
        assert config.agent_pipeline.planner.max_turns == 10
        assert config.agent_pipeline.planner.timeout_seconds == 180

    def test_default_coder_config(self):
        config = Config()
        assert config.agent_pipeline.coder.model == "opus"
        assert config.agent_pipeline.coder.max_turns == 25
        assert config.agent_pipeline.coder.timeout_seconds == 300

    def test_default_tester_config(self):
        config = Config()
        assert config.agent_pipeline.tester.model == "opus"
        assert config.agent_pipeline.tester.max_turns == 15
        assert config.agent_pipeline.tester.timeout_seconds == 240

    def test_default_reviewer_config(self):
        config = Config()
        assert config.agent_pipeline.reviewer.model == "opus"
        assert config.agent_pipeline.reviewer.max_turns == 10
        assert config.agent_pipeline.reviewer.timeout_seconds == 180

    def test_default_agent_workspace_dir(self):
        config = Config()
        assert config.paths.agent_workspace_dir == "state/agent_workspace"

    def test_yaml_merge_enabled(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text(
            "agent_pipeline:\n"
            "  enabled: true\n"
            "  max_revisions: 3\n"
        )
        config = load_config(str(f))
        assert config.agent_pipeline.enabled is True
        assert config.agent_pipeline.max_revisions == 3
        # Agent defaults preserved
        assert config.agent_pipeline.planner.model == "opus"

    def test_yaml_nested_agent_override(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text(
            "agent_pipeline:\n"
            "  enabled: true\n"
            "  coder:\n"
            "    model: haiku\n"
            "    max_turns: 50\n"
        )
        config = load_config(str(f))
        assert config.agent_pipeline.enabled is True
        assert config.agent_pipeline.coder.model == "haiku"
        assert config.agent_pipeline.coder.max_turns == 50
        # Coder timeout_seconds keeps default
        assert config.agent_pipeline.coder.timeout_seconds == 300
        # Other agents unchanged
        assert config.agent_pipeline.planner.model == "opus"
        assert config.agent_pipeline.reviewer.model == "opus"

    def test_yaml_partial_override_preserves_defaults(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text(
            "agent_pipeline:\n"
            "  planner:\n"
            "    max_turns: 20\n"
        )
        config = load_config(str(f))
        # planner max_turns overridden
        assert config.agent_pipeline.planner.max_turns == 20
        # planner model keeps default
        assert config.agent_pipeline.planner.model == "opus"
        # enabled keeps default
        assert config.agent_pipeline.enabled is False
        # other agents fully default
        assert config.agent_pipeline.coder.model == "opus"
        assert config.agent_pipeline.tester.max_turns == 15
