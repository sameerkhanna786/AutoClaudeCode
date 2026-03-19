"""Tests for config_schema module."""

import subprocess
import tempfile
from pathlib import Path

import pytest

from config_schema import (
    Config, load_config,
    ClaudeConfig, DiscoveryConfig, AgentPipelineConfig, ParallelConfig,
)


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repository and return its path."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo), capture_output=True, check=True)
    return repo


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

    def test_load_config_partial_override(self, tmp_path, git_repo):
        f = tmp_path / "config.yaml"
        f.write_text(
            f"target_dir: {git_repo}\n"
            "claude:\n"
            "  model: opus\n"
            "  max_turns: 10\n"
        )
        config = load_config(str(f))
        assert config.target_dir == str(git_repo)
        assert config.claude.model == "opus"
        assert config.claude.max_turns == 10
        # Unset values keep defaults
        assert config.claude.timeout_seconds == ClaudeConfig().timeout_seconds
        assert config.orchestrator.loop_interval_seconds == 30

    def test_load_config_full_file(self, tmp_path, git_repo):
        f = tmp_path / "config.yaml"
        f.write_text(
            f"target_dir: {git_repo}\n"
            "safety:\n"
            "  max_consecutive_failures: 3\n"
            "  protected_files:\n"
            "    - main.py\n"
            "    - config.yaml\n"
            "    - secret.py\n"
        )
        config = load_config(str(f))
        assert config.target_dir == str(git_repo)
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
        assert config.discovery.discovery_timeout == DiscoveryConfig().discovery_timeout

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
        assert config.agent_pipeline.planner.timeout_seconds == AgentPipelineConfig().planner.timeout_seconds

    def test_default_coder_config(self):
        config = Config()
        assert config.agent_pipeline.coder.model == "opus"
        assert config.agent_pipeline.coder.max_turns == 25
        assert config.agent_pipeline.coder.timeout_seconds == AgentPipelineConfig().coder.timeout_seconds

    def test_default_tester_config(self):
        config = Config()
        assert config.agent_pipeline.tester.model == "opus"
        assert config.agent_pipeline.tester.max_turns == 15
        assert config.agent_pipeline.tester.timeout_seconds == AgentPipelineConfig().tester.timeout_seconds

    def test_default_reviewer_config(self):
        config = Config()
        assert config.agent_pipeline.reviewer.model == "opus"
        assert config.agent_pipeline.reviewer.max_turns == 10
        assert config.agent_pipeline.reviewer.timeout_seconds == AgentPipelineConfig().reviewer.timeout_seconds

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
        assert config.agent_pipeline.coder.timeout_seconds == AgentPipelineConfig().coder.timeout_seconds
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


class TestParallelConfig:
    def test_default_parallel_disabled(self):
        config = Config()
        assert config.parallel.enabled is False

    def test_default_max_workers(self):
        config = Config()
        assert config.parallel.max_workers == 3

    def test_default_merge_strategy(self):
        config = Config()
        assert config.parallel.merge_strategy == "rebase"

    def test_default_worktree_base_dir(self):
        config = Config()
        assert config.parallel.worktree_base_dir == ".worktrees"

    def test_default_cleanup_on_exit(self):
        config = Config()
        assert config.parallel.cleanup_on_exit is True

    def test_default_cleanup_timeout(self):
        config = Config()
        assert config.parallel.cleanup_timeout == 60

    def test_parallel_yaml_override(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text(
            "parallel:\n"
            "  enabled: true\n"
            "  max_workers: 5\n"
            "  merge_strategy: merge\n"
            "  worktree_base_dir: custom_worktrees\n"
            "  max_merge_retries: 4\n"
            "  cleanup_on_exit: false\n"
            "  cleanup_timeout: 120\n"
        )
        config = load_config(str(f))
        assert config.parallel.enabled is True
        assert config.parallel.max_workers == 5
        assert config.parallel.merge_strategy == "merge"
        assert config.parallel.worktree_base_dir == "custom_worktrees"
        assert config.parallel.max_merge_retries == 4
        assert config.parallel.cleanup_on_exit is False
        assert config.parallel.cleanup_timeout == 120

    def test_parallel_partial_override(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text(
            "parallel:\n"
            "  enabled: true\n"
        )
        config = load_config(str(f))
        assert config.parallel.enabled is True
        # Other fields keep defaults
        assert config.parallel.max_workers == 3
        assert config.parallel.merge_strategy == "rebase"


class TestValidateConfig:
    def test_valid_default_config_passes(self):
        from config_schema import validate_config
        config = Config()
        # Should not raise
        validate_config(config)

    def test_zero_claude_timeout_raises(self):
        from config_schema import validate_config
        config = Config()
        config.claude.timeout_seconds = 0
        with pytest.raises(ValueError, match="claude.timeout_seconds"):
            validate_config(config)

    def test_negative_test_timeout_raises(self):
        from config_schema import validate_config
        config = Config()
        config.validation.test_timeout = -1
        with pytest.raises(ValueError, match="validation.test_timeout"):
            validate_config(config)

    def test_zero_loop_interval_raises(self):
        from config_schema import validate_config
        config = Config()
        config.orchestrator.loop_interval_seconds = 0
        with pytest.raises(ValueError, match="orchestrator.loop_interval_seconds"):
            validate_config(config)

    def test_zero_discovery_timeout_raises(self):
        from config_schema import validate_config
        config = Config()
        config.discovery.discovery_timeout = 0
        with pytest.raises(ValueError, match="discovery.discovery_timeout"):
            validate_config(config)

    def test_zero_max_consecutive_failures_raises(self):
        from config_schema import validate_config
        config = Config()
        config.safety.max_consecutive_failures = 0
        with pytest.raises(ValueError, match="safety.max_consecutive_failures"):
            validate_config(config)

    def test_zero_max_cycles_per_hour_raises(self):
        from config_schema import validate_config
        config = Config()
        config.safety.max_cycles_per_hour = 0
        with pytest.raises(ValueError, match="safety.max_cycles_per_hour"):
            validate_config(config)

    def test_zero_max_cost_raises(self):
        from config_schema import validate_config
        config = Config()
        config.safety.max_cost_usd_per_hour = 0
        with pytest.raises(ValueError, match="safety.max_cost_usd_per_hour"):
            validate_config(config)

    def test_zero_min_disk_space_raises(self):
        from config_schema import validate_config
        config = Config()
        config.safety.min_disk_space_mb = 0
        with pytest.raises(ValueError, match="safety.min_disk_space_mb"):
            validate_config(config)

    def test_zero_max_workers_raises(self):
        from config_schema import validate_config
        config = Config()
        config.parallel.enabled = True
        config.parallel.max_workers = 0
        with pytest.raises(ValueError, match="parallel.max_workers"):
            validate_config(config)

    def test_claude_timeout_le_test_timeout_raises(self):
        from config_schema import validate_config
        config = Config()
        config.claude.timeout_seconds = 100
        config.validation.test_timeout = 100
        with pytest.raises(ValueError, match="claude.timeout_seconds.*must be greater than"):
            validate_config(config)

    def test_pipeline_zero_timeout_raises(self):
        from config_schema import validate_config
        config = Config()
        config.agent_pipeline.enabled = True
        config.agent_pipeline.planner.timeout_seconds = 0
        with pytest.raises(ValueError, match="agent_pipeline.planner.timeout_seconds"):
            validate_config(config)

    def test_pipeline_zero_max_turns_raises(self):
        from config_schema import validate_config
        config = Config()
        config.agent_pipeline.enabled = True
        config.agent_pipeline.coder.max_turns = 0
        with pytest.raises(ValueError, match="agent_pipeline.coder.max_turns"):
            validate_config(config)

    def test_pipeline_empty_model_raises(self):
        from config_schema import validate_config
        config = Config()
        config.agent_pipeline.enabled = True
        config.agent_pipeline.tester.model = ""
        with pytest.raises(ValueError, match="agent_pipeline.tester.model"):
            validate_config(config)

    def test_pipeline_disabled_skips_role_validation(self):
        from config_schema import validate_config
        config = Config()
        config.agent_pipeline.enabled = False
        config.agent_pipeline.planner.timeout_seconds = 0
        config.agent_pipeline.coder.max_turns = 0
        # Should not raise — pipeline is disabled
        validate_config(config)


class TestValidateTargetDir:
    """Tests for target_dir validation in validate_config()."""

    def test_nonexistent_target_dir_raises(self):
        from config_schema import validate_config
        config = Config()
        config.target_dir = "/nonexistent/path/that/does/not/exist"
        with pytest.raises(ValueError, match="target_dir does not exist"):
            validate_config(config)

    def test_target_dir_is_file_raises(self, tmp_path):
        from config_schema import validate_config
        f = tmp_path / "afile.txt"
        f.write_text("hello")
        config = Config()
        config.target_dir = str(f)
        with pytest.raises(ValueError, match="target_dir does not exist"):
            validate_config(config)

    def test_target_dir_not_git_repo_raises(self, tmp_path):
        from config_schema import validate_config
        non_git_dir = tmp_path / "not_a_repo"
        non_git_dir.mkdir()
        config = Config()
        config.target_dir = str(non_git_dir)
        with pytest.raises(ValueError, match="target_dir is not a git repository"):
            validate_config(config)

    def test_valid_git_repo_passes(self, tmp_path):
        import subprocess
        from config_schema import validate_config
        repo_dir = tmp_path / "valid_repo"
        repo_dir.mkdir()
        subprocess.run(["git", "init"], cwd=str(repo_dir),
                        capture_output=True, check=True)
        config = Config()
        config.target_dir = str(repo_dir)
        # Should not raise
        validate_config(config)


class TestValidateBatchSizeOrdering:
    """Tests for batch size ordering validation in validate_config()."""

    def test_min_greater_than_initial_raises(self, tmp_path):
        import subprocess
        from config_schema import validate_config
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        subprocess.run(["git", "init"], cwd=str(repo_dir),
                        capture_output=True, check=True)
        config = Config()
        config.target_dir = str(repo_dir)
        config.orchestrator.min_batch_size = 5
        config.orchestrator.initial_batch_size = 3
        config.orchestrator.max_batch_size = 10
        with pytest.raises(ValueError, match="min_batch_size.*must be <=.*initial_batch_size"):
            validate_config(config)

    def test_initial_greater_than_max_raises(self, tmp_path):
        import subprocess
        from config_schema import validate_config
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        subprocess.run(["git", "init"], cwd=str(repo_dir),
                        capture_output=True, check=True)
        config = Config()
        config.target_dir = str(repo_dir)
        config.orchestrator.min_batch_size = 1
        config.orchestrator.initial_batch_size = 15
        config.orchestrator.max_batch_size = 10
        with pytest.raises(ValueError, match="initial_batch_size.*must be <=.*max_batch_size"):
            validate_config(config)

    def test_valid_ordering_passes(self, tmp_path):
        import subprocess
        from config_schema import validate_config
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        subprocess.run(["git", "init"], cwd=str(repo_dir),
                        capture_output=True, check=True)
        config = Config()
        config.target_dir = str(repo_dir)
        config.orchestrator.min_batch_size = 2
        config.orchestrator.initial_batch_size = 5
        config.orchestrator.max_batch_size = 10
        # Should not raise
        validate_config(config)

    def test_equal_values_passes(self, tmp_path):
        import subprocess
        from config_schema import validate_config
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        subprocess.run(["git", "init"], cwd=str(repo_dir),
                        capture_output=True, check=True)
        config = Config()
        config.target_dir = str(repo_dir)
        config.orchestrator.min_batch_size = 5
        config.orchestrator.initial_batch_size = 5
        config.orchestrator.max_batch_size = 5
        # Should not raise — all equal is valid
        validate_config(config)


class TestValidateModelNames:
    """Tests for model name validation in validate_config()."""

    def test_empty_model_name_raises(self):
        from config_schema import validate_config
        config = Config()
        config.claude.model = ""
        with pytest.raises(ValueError, match="claude.model must be a non-empty string"):
            validate_config(config)

    def test_whitespace_only_model_name_raises(self):
        from config_schema import validate_config
        config = Config()
        config.claude.model = "  "
        with pytest.raises(ValueError, match="claude.model must be a non-empty string"):
            validate_config(config)

    def test_model_with_embedded_spaces_raises(self):
        from config_schema import validate_config
        config = Config()
        config.claude.model = "claude opus"
        with pytest.raises(ValueError, match="contains whitespace"):
            validate_config(config)

    def test_known_alias_passes(self):
        from config_schema import validate_config
        config = Config()
        config.claude.model = "opus"
        validate_config(config)

    def test_full_model_id_passes(self):
        from config_schema import validate_config
        config = Config()
        config.claude.model = "claude-opus-4-6"
        validate_config(config)

    def test_unknown_alias_warns(self, caplog):
        import logging
        from config_schema import validate_config
        config = Config()
        config.claude.model = "unknown-model"
        with caplog.at_level(logging.WARNING):
            validate_config(config)
        assert any("not a recognized alias" in r.message for r in caplog.records)

    def test_discovery_model_with_whitespace_raises(self):
        from config_schema import validate_config
        config = Config()
        config.discovery.discovery_model = "opus model"
        with pytest.raises(ValueError, match="discovery.discovery_model contains whitespace"):
            validate_config(config)


class TestValidateMaxRetries:
    """Tests for max_retries and max_turns validation."""

    def test_negative_max_retries_raises(self):
        from config_schema import validate_config
        config = Config()
        config.claude.max_retries = -1
        with pytest.raises(ValueError, match="claude.max_retries"):
            validate_config(config)

    def test_zero_max_retries_passes(self):
        from config_schema import validate_config
        config = Config()
        config.claude.max_retries = 0
        validate_config(config)

    def test_zero_max_turns_raises(self):
        from config_schema import validate_config
        config = Config()
        config.claude.max_turns = 0
        with pytest.raises(ValueError, match="claude.max_turns"):
            validate_config(config)


class TestValidateFilePaths:
    """Tests for file path validation."""

    def test_empty_history_file_raises(self):
        from config_schema import validate_config
        config = Config()
        config.paths.history_file = ""
        with pytest.raises(ValueError, match="paths.history_file"):
            validate_config(config)

    def test_empty_lock_file_raises(self):
        from config_schema import validate_config
        config = Config()
        config.paths.lock_file = ""
        with pytest.raises(ValueError, match="paths.lock_file"):
            validate_config(config)

    def test_empty_state_dir_raises(self):
        from config_schema import validate_config
        config = Config()
        config.paths.state_dir = ""
        with pytest.raises(ValueError, match="paths.state_dir"):
            validate_config(config)

    def test_empty_feedback_dir_raises(self):
        from config_schema import validate_config
        config = Config()
        config.paths.feedback_dir = ""
        with pytest.raises(ValueError, match="paths.feedback_dir"):
            validate_config(config)

    def test_valid_paths_pass(self):
        from config_schema import validate_config
        config = Config()
        # Default paths should be valid
        validate_config(config)


class TestValidateMemoryConfig:
    """Tests for min_memory_mb validation."""

    def test_negative_min_memory_raises(self):
        from config_schema import validate_config
        config = Config()
        config.safety.min_memory_mb = -1
        with pytest.raises(ValueError, match="safety.min_memory_mb"):
            validate_config(config)

    def test_zero_min_memory_passes(self):
        from config_schema import validate_config
        config = Config()
        config.safety.min_memory_mb = 0
        validate_config(config)

    def test_default_min_memory(self):
        config = Config()
        assert config.safety.min_memory_mb == 256


class TestValidateEnumFields:
    """Tests for enum-like config field validation."""

    def test_invalid_provider_warns(self, caplog):
        import logging
        from config_schema import validate_config
        config = Config()
        config.claude.provider = "invalid_provider"
        with caplog.at_level(logging.WARNING):
            validate_config(config)
        assert any(
            "claude.provider" in r.message and "unrecognized value" in r.message
            for r in caplog.records
        )

    def test_valid_providers_no_warning(self, caplog):
        import logging
        from config_schema import validate_config
        for provider in ("claude", "openai", "gemini"):
            caplog.clear()
            config = Config()
            config.claude.provider = provider
            with caplog.at_level(logging.WARNING):
                validate_config(config)
            assert not any("claude.provider" in r.message for r in caplog.records)

    def test_invalid_merge_strategy_warns(self, caplog):
        import logging
        from config_schema import validate_config
        config = Config()
        config.parallel.merge_strategy = "squash"
        with caplog.at_level(logging.WARNING):
            validate_config(config)
        assert any(
            "parallel.merge_strategy" in r.message and "unrecognized value" in r.message
            for r in caplog.records
        )

    def test_invalid_fail_action_warns(self, caplog):
        import logging
        from config_schema import validate_config
        config = Config()
        config.judges.fail_action = "panic"
        with caplog.at_level(logging.WARNING):
            validate_config(config)
        assert any(
            "judges.fail_action" in r.message and "unrecognized value" in r.message
            for r in caplog.records
        )

    def test_invalid_logging_format_warns(self, caplog):
        import logging
        from config_schema import validate_config
        config = Config()
        config.logging.format = "xml"
        with caplog.at_level(logging.WARNING):
            validate_config(config)
        assert any(
            "logging.format" in r.message and "unrecognized value" in r.message
            for r in caplog.records
        )

    def test_invalid_review_detail_warns(self, caplog):
        import logging
        from config_schema import validate_config
        config = Config()
        config.agent_pipeline.review_detail = "extreme"
        with caplog.at_level(logging.WARNING):
            validate_config(config)
        assert any(
            "agent_pipeline.review_detail" in r.message and "unrecognized value" in r.message
            for r in caplog.records
        )

    def test_valid_enum_values_no_warnings(self, caplog):
        import logging
        from config_schema import validate_config
        config = Config()
        # All defaults are valid
        with caplog.at_level(logging.WARNING):
            validate_config(config)
        enum_fields = [
            "claude.provider", "parallel.merge_strategy",
            "judges.fail_action", "logging.format",
            "agent_pipeline.review_detail",
        ]
        for field in enum_fields:
            assert not any(
                field in r.message and "unrecognized value" in r.message
                for r in caplog.records
            ), f"Unexpected warning for {field}"

    def test_invalid_enum_does_not_raise(self):
        from config_schema import validate_config
        config = Config()
        config.claude.provider = "invalid"
        config.parallel.merge_strategy = "invalid"
        config.judges.fail_action = "invalid"
        config.logging.format = "invalid"
        config.agent_pipeline.review_detail = "invalid"
        # Should NOT raise — only warns
        validate_config(config)


class TestConfigLoadingRobustness:
    """Tests for error handling during config loading."""

    def test_malformed_yaml_raises_valueerror(self):
        """Malformed YAML should raise ValueError, not yaml.YAMLError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = str(Path(tmpdir) / "bad.yaml")
            Path(config_path).write_text("{{{{bad yaml: [unterminated")
            with pytest.raises(ValueError, match="Invalid YAML"):
                load_config(config_path)

    def test_valid_yaml_loads_normally(self):
        """Valid YAML should load without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = str(Path(tmpdir) / "good.yaml")
            Path(config_path).write_text("claude:\n  model: sonnet\n")
            config = load_config(config_path)
            assert config.claude.model == "sonnet"

    def test_webhook_unknown_keys_ignored(self):
        """Unknown keys in webhook config should be silently ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = str(Path(tmpdir) / "webhooks.yaml")
            Path(config_path).write_text(
                "notifications:\n"
                "  enabled: true\n"
                "  webhooks:\n"
                "    - url: https://example.com/hook\n"
                "      type: slack\n"
                "      name: test\n"
                "      headers: {Authorization: Bearer xyz}\n"
                "      custom_field: should_be_ignored\n"
            )
            config = load_config(config_path)
            assert len(config.notifications.webhooks) == 1
            assert config.notifications.webhooks[0].url == "https://example.com/hook"
            assert config.notifications.webhooks[0].type == "slack"

    def test_git_not_found_raises_valueerror(self):
        """Missing git binary should raise ValueError during validation."""
        from unittest.mock import patch
        from config_schema import validate_config
        config = Config()
        with patch("config_schema.subprocess.run", side_effect=FileNotFoundError("git")):
            with pytest.raises(ValueError, match="git is not installed"):
                validate_config(config)

    def test_git_timeout_raises_valueerror(self):
        """Git subprocess timeout should raise ValueError."""
        from unittest.mock import patch
        from config_schema import validate_config
        config = Config()
        with patch("config_schema.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="git", timeout=10)):
            with pytest.raises(ValueError, match="timed out"):
                validate_config(config)


class TestGetExpectedTypeDictHandling:
    """Test that _get_expected_type handles Dict[K,V] types."""

    def test_dict_type_recognized(self):
        from config_schema import _get_expected_type, PricingConfig
        result = _get_expected_type(PricingConfig, "cost_per_million_input_tokens")
        assert result is dict

    def test_non_dict_value_rejected_by_merge(self):
        """_merge_dataclass should reject non-dict values for dict fields."""
        from config_schema import _merge_dataclass, PricingConfig
        pricing = PricingConfig()
        _merge_dataclass(pricing, {"cost_per_million_input_tokens": "not_a_dict"})
        # Should remain unchanged (invalid value skipped)
        assert isinstance(pricing.cost_per_million_input_tokens, dict)


class TestBoolIntTypeConfusion:
    """Test that _merge_dataclass rejects bool values for int fields.

    Python's bool is a subclass of int, so isinstance(True, int) returns True.
    The merge logic must disambiguate to prevent e.g. max_turns=True (treated as 1).
    """

    def test_bool_rejected_for_int_field(self):
        """Bool value should be rejected when an int field is expected."""
        from config_schema import _merge_dataclass, ClaudeConfig
        cc = ClaudeConfig()
        original_max_turns = cc.max_turns
        _merge_dataclass(cc, {"max_turns": True})
        # Should remain unchanged — bool should not be accepted as int
        assert cc.max_turns == original_max_turns
        assert type(cc.max_turns) is int

    def test_bool_rejected_for_float_field(self):
        """Bool value should be rejected when a float field is expected."""
        from config_schema import _merge_dataclass, ClaudeConfig
        cc = ClaudeConfig()
        original = cc.timeout_seconds
        _merge_dataclass(cc, {"timeout_seconds": False})
        assert cc.timeout_seconds == original

    def test_int_still_accepted_for_int_field(self):
        """Regular int values should still be accepted for int fields."""
        from config_schema import _merge_dataclass, ClaudeConfig
        cc = ClaudeConfig()
        _merge_dataclass(cc, {"max_turns": 42})
        assert cc.max_turns == 42


class TestTimeoutValidationNoTestCommand:
    """Test that timeout cross-validation is skipped when no test command is set."""

    def test_no_test_command_allows_equal_timeouts(self):
        """When test_command is empty, timeout_seconds == test_timeout should be allowed."""
        from config_schema import validate_config
        config = Config()
        config.validation.test_command = ""
        config.claude.timeout_seconds = 7200
        config.validation.test_timeout = 7200
        # Should NOT raise
        validate_config(config)

    def test_with_test_command_rejects_equal_timeouts(self):
        """When test_command is set, timeout_seconds must exceed test_timeout."""
        from config_schema import validate_config
        config = Config()
        config.validation.test_command = "python3 -m pytest"
        config.claude.timeout_seconds = 7200
        config.validation.test_timeout = 7200
        with pytest.raises(ValueError, match="must be greater than"):
            validate_config(config)
