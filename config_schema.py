"""Load and validate configuration from YAML with sensible defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class ClaudeConfig:
    model: str = "opus"
    max_turns: int = 25
    timeout_seconds: int = 300
    command: str = "claude"


@dataclass
class OrchestratorConfig:
    loop_interval_seconds: int = 30
    max_changed_files: int = 20
    self_improve: bool = False
    push_after_commit: bool = False
    plan_changes: bool = False


@dataclass
class ValidationConfig:
    test_command: str = "python3 -m pytest tests/ -x -q"
    lint_command: str = ""
    build_command: str = ""
    test_timeout: int = 120
    lint_timeout: int = 60
    build_timeout: int = 120


@dataclass
class DiscoveryConfig:
    enable_test_failures: bool = True
    enable_lint_errors: bool = True
    enable_todos: bool = True
    enable_coverage: bool = False
    enable_quality_review: bool = False
    todo_patterns: List[str] = field(default_factory=lambda: ["TODO", "FIXME", "HACK"])
    exclude_dirs: List[str] = field(default_factory=lambda: [
        "__pycache__", ".git", "node_modules", ".venv", "venv",
    ])


@dataclass
class SafetyConfig:
    max_consecutive_failures: int = 5
    max_cycles_per_hour: int = 30
    max_cost_usd_per_hour: float = 10.0
    min_disk_space_mb: int = 500
    protected_files: List[str] = field(default_factory=lambda: ["main.py", "config.yaml"])


@dataclass
class PathsConfig:
    feedback_dir: str = "feedback"
    feedback_done_dir: str = "feedback/done"
    state_dir: str = "state"
    history_file: str = "state/history.json"
    lock_file: str = "state/lock.pid"
    backup_dir: str = "state/backups"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "state/auto_claude.log"
    max_bytes: int = 5_000_000
    backup_count: int = 3


@dataclass
class Config:
    target_dir: str = "."
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _merge_dataclass(dc_instance, overrides: dict):
    """Merge a dict of overrides into a dataclass instance."""
    if not overrides:
        return dc_instance
    for key, value in overrides.items():
        if hasattr(dc_instance, key):
            setattr(dc_instance, key, value)
    return dc_instance


def load_config(path: Optional[str] = None) -> Config:
    """Load configuration from a YAML file, merging with defaults.

    If path is None, returns a Config with all defaults.
    """
    config = Config()

    if path is None:
        return config

    config_path = Path(path)
    if not config_path.exists():
        return config

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    if not raw or not isinstance(raw, dict):
        return config

    if "target_dir" in raw:
        config.target_dir = raw["target_dir"]

    section_map = {
        "claude": config.claude,
        "orchestrator": config.orchestrator,
        "validation": config.validation,
        "discovery": config.discovery,
        "safety": config.safety,
        "paths": config.paths,
        "logging": config.logging,
    }

    for section_name, dc_instance in section_map.items():
        if section_name in raw and isinstance(raw[section_name], dict):
            _merge_dataclass(dc_instance, raw[section_name])

    return config
