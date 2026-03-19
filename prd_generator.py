"""PRD generation and import: create and parse Product Requirement Documents."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from task_discovery import Task

logger = logging.getLogger(__name__)


def generate_prd(tasks: List[Task], config=None, state_manager=None) -> Dict[str, Any]:
    """Convert discovered tasks into a structured PRD with performance data.

    Generates a machine-readable PRD from the current task list, optionally
    enriched with historical performance data from the state manager.
    """
    prd: Dict[str, Any] = {
        "version": "1.0",
        "generated_at": time.time(),
        "generator": "auto-claude-code",
        "tasks": [],
        "metadata": {},
    }

    if config:
        prd["metadata"]["target_dir"] = config.target_dir
        prd["metadata"]["model"] = config.claude.model

    # Add historical performance data
    if state_manager:
        try:
            performance = state_manager.get_strategy_performance(lookback_seconds=86400)
            if performance:
                prd["metadata"]["strategy_performance"] = performance
        except Exception:
            logger.debug("Failed to load strategy performance for PRD", exc_info=True)

    for task in tasks:
        task_entry: Dict[str, Any] = {
            "id": task.task_id,
            "description": task.description,
            "priority": task.priority,
            "source": task.source,
            "depends_on": task.depends_on,
        }
        if task.source_file:
            task_entry["source_file"] = task.source_file
        if task.line_number is not None:
            task_entry["line_number"] = task.line_number
        if task.context:
            task_entry["context"] = task.context[:500]  # Truncate for PRD

        prd["tasks"].append(task_entry)

    return prd


def import_prd(prd_path: str) -> List[Task]:
    """Import a PRD file (YAML or JSON) as a list of Tasks.

    Supports both auto-generated PRDs and Tomacco-format PRDs.
    """
    path = Path(prd_path)
    if not path.exists():
        logger.warning("PRD file not found: %s", prd_path)
        return []

    MAX_PRD_FILE_SIZE = 1024 * 1024  # 1 MB
    try:
        file_size = path.stat().st_size
    except OSError:
        logger.warning("Cannot stat PRD file: %s", prd_path)
        return []
    if file_size > MAX_PRD_FILE_SIZE:
        logger.warning(
            "PRD file too large (%d bytes, limit %d): %s",
            file_size, MAX_PRD_FILE_SIZE, prd_path,
        )
        return []

    content = path.read_text(encoding="utf-8")

    # Try JSON first
    data = None
    if path.suffix in (".json",):
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse PRD as JSON: %s", prd_path)
            return []
    else:
        # Try YAML
        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError:
            logger.warning("Failed to parse PRD as YAML: %s", prd_path)
            return []

    if not isinstance(data, dict):
        logger.warning("PRD root is not a dict: %s", prd_path)
        return []

    tasks: List[Task] = []

    # Auto-generated format: {"tasks": [...]}
    raw_tasks = data.get("tasks", [])

    # Tomacco format: {"phases": [{"tasks": [...]}]}
    if not raw_tasks and "phases" in data:
        for phase in data.get("phases", []):
            if isinstance(phase, dict):
                raw_tasks.extend(phase.get("tasks", []))

    for i, raw in enumerate(raw_tasks):
        if not isinstance(raw, dict):
            continue

        desc = raw.get("description", raw.get("title", raw.get("name", "")))
        if not desc:
            continue

        try:
            priority = int(raw.get("priority", 3))
        except (ValueError, TypeError):
            priority = 3

        task = Task(
            description=str(desc),
            priority=priority,
            source=raw.get("source", "feedback"),
            source_file=raw.get("source_file"),
            line_number=raw.get("line_number"),
            context=raw.get("context", ""),
            task_id=raw.get("id", raw.get("task_id", f"prd-task-{i}")),
            depends_on=raw.get("depends_on", []),
        )
        tasks.append(task)

    logger.info("Imported %d tasks from PRD: %s", len(tasks), prd_path)
    return tasks


def export_prd(prd: Dict[str, Any], output_path: str,
               fmt: str = "yaml") -> None:
    """Write PRD to a file in YAML or JSON format."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        content = json.dumps(prd, indent=2)
    else:
        content = yaml.dump(prd, default_flow_style=False, sort_keys=False)

    path.write_text(content, encoding="utf-8")
    logger.info("Exported PRD to %s (%s format, %d tasks)",
                output_path, fmt, len(prd.get("tasks", [])))
