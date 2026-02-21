"""Load and validate configuration from YAML with sensible defaults."""

from __future__ import annotations

import dataclasses
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, get_type_hints

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ClaudeConfig:
    model: str = "opus"
    resolved_model: str = ""   # Populated at startup by Orchestrator
    max_turns: int = 25
    timeout_seconds: int = 14400
    command: str = "claude"
    max_retries: int = 3
    retry_delays: List[int] = field(default_factory=lambda: [2, 8, 32])
    rate_limit_base_delay: int = 5
    rate_limit_multiplier: int = 3


@dataclass
class OrchestratorConfig:
    loop_interval_seconds: int = 30
    max_changed_files: int = 20
    self_improve: bool = False
    push_after_commit: bool = False
    plan_changes: bool = False
    planning_max_turns: int = 10
    max_feedback_retries: int = 3
    max_tasks_per_cycle: int = 10
    batch_mode: bool = True
    cycle_timeout_seconds: int = 43200
    # Adaptive batch sizing
    min_batch_size: int = 1
    max_batch_size: int = 10
    initial_batch_size: int = 3
    batch_grow_step: int = 1
    batch_shrink_step: int = 2
    adaptive_batch_window: int = 10
    batch_cost_ceiling: float = 8.0
    # Task priority decay: hours before a skipped task is promoted by 1 level
    priority_decay_hours: float = 24.0
    # Validation retry — re-invoke Claude with failure output to fix in-place
    max_validation_retries: int = 5
    retry_include_full_output: bool = True
    gc_interval: int = 10


@dataclass
class ValidationConfig:
    test_command: str = "python3 -m pytest tests/ -x -q"
    lint_command: str = ""
    build_command: str = ""
    test_timeout: int = 7200
    lint_timeout: int = 7200
    build_timeout: int = 7200


@dataclass
class DiscoveryConfig:
    enable_test_failures: bool = True
    enable_lint_errors: bool = True
    enable_todos: bool = True
    enable_coverage: bool = False
    enable_quality_review: bool = False
    enable_claude_ideas: bool = False
    todo_patterns: List[str] = field(default_factory=lambda: ["TODO", "FIXME", "HACK"])
    exclude_dirs: List[str] = field(default_factory=lambda: [
        "__pycache__", ".git", "node_modules", ".venv", "venv",
    ])
    max_todo_tasks: int = 20
    discovery_model: str = "opus"
    discovery_timeout: int = 7200
    discovery_max_turns: int = 15
    discovery_prompt: str = ""  # Custom prompt for claude_ideas; empty = use default


@dataclass
class SafetyConfig:
    max_consecutive_failures: int = 5
    max_cycles_per_hour: int = 30
    max_cost_usd_per_hour: float = 10.0
    min_disk_space_mb: int = 500
    min_memory_mb: int = 256
    max_history_records: int = 1000
    protected_files: List[str] = field(default_factory=lambda: ["main.py", "config.yaml"])


@dataclass
class AgentRoleConfig:
    """Per-agent settings."""
    enabled: bool = True
    model: str = "opus"
    max_turns: int = 25
    timeout_seconds: int = 7200


@dataclass
class AgentPipelineConfig:
    """Multi-agent pipeline settings."""
    enabled: bool = False
    max_revisions: int = 2
    max_pipeline_cost_usd: float = 0.0  # 0 = use safety.max_cost_usd_per_hour * 0.5
    planner: AgentRoleConfig = field(default_factory=lambda: AgentRoleConfig(
        model="opus", max_turns=10, timeout_seconds=7200))
    coder: AgentRoleConfig = field(default_factory=lambda: AgentRoleConfig(
        model="opus", max_turns=25, timeout_seconds=14400))
    tester: AgentRoleConfig = field(default_factory=lambda: AgentRoleConfig(
        model="opus", max_turns=15, timeout_seconds=7200))
    reviewer: AgentRoleConfig = field(default_factory=lambda: AgentRoleConfig(
        model="opus", max_turns=10, timeout_seconds=7200))


@dataclass
class ParallelConfig:
    enabled: bool = False
    max_workers: int = 3
    worktree_base_dir: str = ".worktrees"
    merge_strategy: str = "rebase"  # "rebase" or "merge"
    max_merge_retries: int = 2
    cleanup_on_exit: bool = True
    cleanup_timeout: int = 60


@dataclass
class PathsConfig:
    feedback_dir: str = "feedback"
    feedback_done_dir: str = "feedback/done"
    state_dir: str = "state"
    history_file: str = "state/history.json"
    lock_file: str = "state/lock.pid"
    backup_dir: str = "state/backups"
    feedback_failed_dir: str = "feedback/failed"
    agent_workspace_dir: str = "state/agent_workspace"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "state/auto_claude.log"
    max_bytes: int = 5_000_000
    backup_count: int = 3
    format: str = "text"  # "text" or "json"


@dataclass
class WebhookConfig:
    """Configuration for a single webhook endpoint."""
    url: str = ""
    type: str = "generic"  # "slack", "discord", "generic"
    name: str = ""


@dataclass
class NotificationEventsConfig:
    """Which events trigger notifications."""
    on_cycle_success: bool = True
    on_cycle_failure: bool = True
    on_consecutive_failure_threshold: bool = True
    on_cost_limit_exceeded: bool = True
    on_safety_error: bool = True


@dataclass
class NotificationsConfig:
    """Top-level notification configuration."""
    enabled: bool = False
    webhooks: List[WebhookConfig] = field(default_factory=list)
    events: NotificationEventsConfig = field(default_factory=NotificationEventsConfig)


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
    agent_pipeline: AgentPipelineConfig = field(default_factory=AgentPipelineConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)


def _get_expected_type(dc_class, field_name: str):
    """Return the expected primitive type for a dataclass field, or None if unknown."""
    try:
        hints = get_type_hints(dc_class)
    except Exception:
        return None
    hint = hints.get(field_name)
    if hint is None:
        return None
    # Unwrap Optional[X] -> X
    origin = getattr(hint, "__origin__", None)
    if origin is type(None):
        return None
    # Handle Union (Optional is Union[X, None])
    import typing
    if origin is getattr(typing, "Union", None):
        args = [a for a in hint.__args__ if a is not type(None)]
        if len(args) == 1:
            hint = args[0]
            origin = getattr(hint, "__origin__", None)
        else:
            return None
    # For List[X], accept list
    if origin is list:
        return list
    # For simple types
    if isinstance(hint, type):
        return hint
    return None


def _merge_dataclass(dc_instance, overrides: dict):
    """Merge a dict of overrides into a dataclass instance.

    Validates each value's type against the dataclass field annotation.
    Invalid types are logged and skipped.
    """
    if not overrides:
        return dc_instance
    dc_class = type(dc_instance)
    for key, value in overrides.items():
        if not hasattr(dc_instance, key):
            logger.warning("Unknown config key '%s.%s' — ignoring (typo?)", dc_class.__name__, key)
            continue
        expected = _get_expected_type(dc_class, key)
        if expected is not None and value is not None:
            # Allow int where float is expected (YAML often produces int for "10.0")
            if expected is float and isinstance(value, int):
                value = float(value)
            elif not isinstance(value, expected):
                logger.warning(
                    "Config field '%s.%s' expects %s but got %s (%r) — skipping",
                    dc_class.__name__, key, expected.__name__,
                    type(value).__name__, value,
                )
                continue
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
        "parallel": config.parallel,
    }

    for section_name, dc_instance in section_map.items():
        if section_name in raw and isinstance(raw[section_name], dict):
            _merge_dataclass(dc_instance, raw[section_name])

    # Legacy migration: max_tasks_per_cycle -> max_batch_size
    if "orchestrator" in raw and isinstance(raw["orchestrator"], dict):
        if "max_tasks_per_cycle" in raw["orchestrator"] and "max_batch_size" not in raw["orchestrator"]:
            config.orchestrator.max_batch_size = config.orchestrator.max_tasks_per_cycle

    # Nested agent pipeline config
    if "agent_pipeline" in raw and isinstance(raw["agent_pipeline"], dict):
        ap_raw = raw["agent_pipeline"]
        _merge_dataclass(config.agent_pipeline, {
            k: v for k, v in ap_raw.items()
            if k not in ("planner", "coder", "tester", "reviewer")
        })
        for agent_name in ("planner", "coder", "tester", "reviewer"):
            if agent_name in ap_raw and isinstance(ap_raw[agent_name], dict):
                _merge_dataclass(getattr(config.agent_pipeline, agent_name), ap_raw[agent_name])

    # Nested notifications config
    if "notifications" in raw and isinstance(raw["notifications"], dict):
        notif_raw = raw["notifications"]
        _merge_dataclass(config.notifications, {
            k: v for k, v in notif_raw.items()
            if k not in ("webhooks", "events")
        })
        if "events" in notif_raw and isinstance(notif_raw["events"], dict):
            _merge_dataclass(config.notifications.events, notif_raw["events"])
        if "webhooks" in notif_raw and isinstance(notif_raw["webhooks"], list):
            config.notifications.webhooks = [
                WebhookConfig(**wh) for wh in notif_raw["webhooks"]
                if isinstance(wh, dict) and wh.get("url")
            ]

    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    """Validate cross-field configuration constraints.

    Raises ValueError if configuration is invalid.
    """
    # Validate model names are non-empty strings without whitespace
    _KNOWN_MODEL_ALIASES = {
        "opus", "sonnet", "haiku",
        "claude-opus-4-6", "claude-sonnet-4-20250514",
        "claude-haiku-3-5-20241022",
    }
    if not config.claude.model or not config.claude.model.strip():
        raise ValueError(
            "claude.model must be a non-empty string"
        )
    model = config.claude.model.strip()
    if " " in model or "\t" in model:
        raise ValueError(
            f"claude.model contains whitespace: {model!r}"
        )
    if model not in _KNOWN_MODEL_ALIASES and not model.startswith("claude-"):
        logger.warning(
            "claude.model '%s' is not a recognized alias or model ID. "
            "Known aliases: %s. If this is intentional, ignore this warning.",
            model, ", ".join(sorted(_KNOWN_MODEL_ALIASES)),
        )

    # Validate discovery model
    if config.discovery.discovery_model:
        dm = config.discovery.discovery_model.strip()
        if " " in dm or "\t" in dm:
            raise ValueError(
                f"discovery.discovery_model contains whitespace: {dm!r}"
            )

    # Validate timeout values are positive integers
    if config.claude.timeout_seconds <= 0:
        raise ValueError(
            f"claude.timeout_seconds must be a positive integer, got {config.claude.timeout_seconds}"
        )
    if config.validation.test_timeout <= 0:
        raise ValueError(
            f"validation.test_timeout must be a positive integer, got {config.validation.test_timeout}"
        )
    if config.validation.lint_timeout <= 0:
        raise ValueError(
            f"validation.lint_timeout must be a positive integer, got {config.validation.lint_timeout}"
        )
    if config.validation.build_timeout <= 0:
        raise ValueError(
            f"validation.build_timeout must be a positive integer, got {config.validation.build_timeout}"
        )
    if config.orchestrator.cycle_timeout_seconds <= 0:
        raise ValueError(
            f"orchestrator.cycle_timeout_seconds must be a positive integer, "
            f"got {config.orchestrator.cycle_timeout_seconds}"
        )

    # Validate max_retries is non-negative
    if config.claude.max_retries < 0:
        raise ValueError(
            f"claude.max_retries must be non-negative, got {config.claude.max_retries}"
        )
    if config.claude.max_turns <= 0:
        raise ValueError(
            f"claude.max_turns must be positive, got {config.claude.max_turns}"
        )

    # Cross-field: Claude timeout must exceed test timeout
    if config.claude.timeout_seconds <= config.validation.test_timeout:
        raise ValueError(
            f"claude.timeout_seconds ({config.claude.timeout_seconds}) must be greater than "
            f"validation.test_timeout ({config.validation.test_timeout}) to ensure tests "
            f"can complete before Claude times out"
        )

    # Validate orchestrator fields
    if config.orchestrator.loop_interval_seconds <= 0:
        raise ValueError(
            f"orchestrator.loop_interval_seconds must be positive, "
            f"got {config.orchestrator.loop_interval_seconds}"
        )

    # Validate discovery timeout
    if config.discovery.discovery_timeout <= 0:
        raise ValueError(
            f"discovery.discovery_timeout must be positive, "
            f"got {config.discovery.discovery_timeout}"
        )

    # Validate file paths: ensure critical path fields are non-empty
    _PATH_FIELDS = [
        ("paths.history_file", config.paths.history_file),
        ("paths.lock_file", config.paths.lock_file),
        ("paths.state_dir", config.paths.state_dir),
        ("paths.feedback_dir", config.paths.feedback_dir),
    ]
    for field_name, field_value in _PATH_FIELDS:
        if not field_value or not field_value.strip():
            raise ValueError(
                f"{field_name} must be a non-empty path"
            )

    # Validate safety fields
    if config.safety.max_consecutive_failures <= 0:
        raise ValueError(
            f"safety.max_consecutive_failures must be positive, "
            f"got {config.safety.max_consecutive_failures}"
        )
    if config.safety.max_cycles_per_hour <= 0:
        raise ValueError(
            f"safety.max_cycles_per_hour must be positive, "
            f"got {config.safety.max_cycles_per_hour}"
        )
    if config.safety.max_cost_usd_per_hour <= 0:
        raise ValueError(
            f"safety.max_cost_usd_per_hour must be positive, "
            f"got {config.safety.max_cost_usd_per_hour}"
        )
    if config.safety.min_disk_space_mb <= 0:
        raise ValueError(
            f"safety.min_disk_space_mb must be positive, "
            f"got {config.safety.min_disk_space_mb}"
        )
    if config.safety.min_memory_mb < 0:
        raise ValueError(
            f"safety.min_memory_mb must be non-negative, "
            f"got {config.safety.min_memory_mb}"
        )

    # Validate parallel config
    if config.parallel.enabled and config.parallel.max_workers <= 0:
        raise ValueError(
            f"parallel.max_workers must be positive when parallel is enabled, "
            f"got {config.parallel.max_workers}"
        )

    # Validate agent pipeline timeouts when enabled
    if config.agent_pipeline.enabled:
        for agent_name in ("planner", "coder", "tester", "reviewer"):
            agent_cfg = getattr(config.agent_pipeline, agent_name)
            if agent_cfg.timeout_seconds <= 0:
                raise ValueError(
                    f"agent_pipeline.{agent_name}.timeout_seconds must be positive "
                    f"when agent_pipeline is enabled, got {agent_cfg.timeout_seconds}"
                )

    # Validate target_dir exists and is a git repository
    target = config.target_dir
    if not os.path.isdir(target):
        raise ValueError(
            f"target_dir does not exist or is not a directory: {target}"
        )
    git_check = subprocess.run(
        ["git", "rev-parse", "--git-dir"],
        cwd=target, capture_output=True, text=True,
    )
    if git_check.returncode != 0:
        raise ValueError(
            f"target_dir is not a git repository: {target}"
        )

    # Validate batch size ordering: min <= initial <= max
    orch = config.orchestrator
    if orch.min_batch_size > orch.initial_batch_size:
        raise ValueError(
            f"orchestrator.min_batch_size ({orch.min_batch_size}) must be <= "
            f"orchestrator.initial_batch_size ({orch.initial_batch_size})"
        )
    if orch.initial_batch_size > orch.max_batch_size:
        raise ValueError(
            f"orchestrator.initial_batch_size ({orch.initial_batch_size}) must be <= "
            f"orchestrator.max_batch_size ({orch.max_batch_size})"
        )

    # Validate notifications config
    _KNOWN_WEBHOOK_TYPES = {"slack", "discord", "generic"}
    if config.notifications.enabled:
        for i, wh in enumerate(config.notifications.webhooks):
            if not wh.url:
                logger.warning("notifications.webhooks[%d] has empty URL — will be skipped", i)
            if wh.type not in _KNOWN_WEBHOOK_TYPES:
                logger.warning(
                    "notifications.webhooks[%d].type '%s' is not recognized. "
                    "Known types: %s",
                    i, wh.type, ", ".join(sorted(_KNOWN_WEBHOOK_TYPES)),
                )
