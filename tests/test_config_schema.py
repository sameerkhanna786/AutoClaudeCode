"""Tests for config_schema module."""

import tempfile
from pathlib import Path

import pytest

from config_schema import Config, load_config


class TestConfigDefaults:
    def test_default_config_has_expected_values(self):
        config = Config()
        assert config.target_dir == "."
        assert config.claude.model == "sonnet"
        assert config.claude.max_turns == 25
        assert config.orchestrator.loop_interval_seconds == 30
        assert config.safety.max_consecutive_failures == 5
        assert config.validation.test_command == "python -m pytest tests/ -x -q"

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
        assert config.claude.model == "sonnet"

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
