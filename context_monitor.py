"""Context monitor: detect context window exhaustion and auto-split tasks."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from claude_runner import ClaudeResult
from config_schema import Config
from task_discovery import Task

logger = logging.getLogger(__name__)


@dataclass
class ContextSignals:
    """Signals extracted from a Claude result about context window usage."""
    input_tokens: int = 0
    output_tokens: int = 0
    context_window_pct: float = 0.0
    hit_max_turns: bool = False
    result_text_empty: bool = False

    @property
    def is_exhausted(self) -> bool:
        """Returns True if the context window was likely exhausted."""
        return self.context_window_pct > 80.0 or self.hit_max_turns


class ContextMonitor:
    """Monitors context window usage and generates split tasks when exhausted."""

    def __init__(self, config: Config):
        self.config = config
        self._split_threshold = config.orchestrator.max_context_pct
        self._max_depth = config.orchestrator.max_split_depth

    def extract_signals(self, result: ClaudeResult) -> ContextSignals:
        """Extract context usage signals from a Claude result."""
        hit_max_turns = False
        if result.error and "max_turns" in result.error:
            hit_max_turns = True
        if result.raw_json and result.raw_json.get("subtype") == "error_max_turns":
            hit_max_turns = True

        return ContextSignals(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            context_window_pct=result.context_window_pct,
            hit_max_turns=hit_max_turns,
            result_text_empty=not result.result_text.strip(),
        )

    def should_split(self, signals: ContextSignals) -> bool:
        """Determine if the task should be split based on context signals."""
        if signals.context_window_pct > self._split_threshold:
            return True
        if signals.hit_max_turns:
            return True
        return False

    def generate_split_tasks(self, task: Task, result: ClaudeResult,
                             split_depth: int = 0) -> List[Task]:
        """Parse result_text for remaining TODOs, create follow-up tasks.

        Analyzes Claude's output for indicators of incomplete work and
        generates focused follow-up tasks with dependency on the original.
        """
        if split_depth >= self._max_depth:
            logger.warning(
                "Max split depth %d reached for task %s, not splitting further",
                self._max_depth, task.task_id,
            )
            return []

        result_text = result.result_text
        if not result_text:
            # No output to parse — create a generic continuation task
            return [Task(
                description=f"Continue: {task.description}",
                priority=task.priority,
                source=task.source,
                task_id=f"{task.task_id}__split_{split_depth + 1}",
                depends_on=[task.task_id],
            )]

        split_tasks: List[Task] = []

        # Look for TODO/FIXME/remaining work patterns in output
        todo_patterns = [
            re.compile(r"(?:TODO|FIXME|REMAINING|STILL NEED TO):\s*(.+)", re.IGNORECASE),
            re.compile(r"(?:I (?:still )?need to|I haven't|I didn't|Not yet)(?:\s+\w+){1,8}", re.IGNORECASE),
            re.compile(r"(?:Next steps?|Remaining work):\s*(.+)", re.IGNORECASE),
        ]

        seen_descriptions = set()
        for pattern in todo_patterns:
            for match in pattern.finditer(result_text):
                desc = match.group(0).strip()
                if len(desc) < 10 or desc in seen_descriptions:
                    continue
                seen_descriptions.add(desc)

                # Truncate long descriptions
                if len(desc) > 200:
                    desc = desc[:200] + "..."

                split_tasks.append(Task(
                    description=f"{desc} (from: {task.description[:80]})",
                    priority=task.priority,
                    source=task.source,
                    task_id=f"{task.task_id}__split_{split_depth + 1}_{len(split_tasks)}",
                    depends_on=[task.task_id],
                ))

        # If no specific TODOs found but context was exhausted, create a continuation
        if not split_tasks:
            split_tasks.append(Task(
                description=f"Continue incomplete work: {task.description}",
                priority=task.priority,
                source=task.source,
                task_id=f"{task.task_id}__split_{split_depth + 1}",
                depends_on=[task.task_id],
            ))

        logger.info(
            "Generated %d split tasks from exhausted context (depth %d)",
            len(split_tasks), split_depth + 1,
        )
        return split_tasks


def write_split_tasks_as_feedback(tasks: List[Task], feedback_dir: str) -> int:
    """Write split tasks as feedback files with depends_on frontmatter.

    Returns the number of files written.
    """
    feedback_path = Path(feedback_dir)
    feedback_path.mkdir(parents=True, exist_ok=True)
    count = 0

    for task in tasks:
        deps_str = ", ".join(f'"{d}"' for d in task.depends_on)
        frontmatter = (
            f"---\n"
            f"task_id: {task.task_id}\n"
            f"depends_on: [{deps_str}]\n"
            f"---\n"
        )
        content = frontmatter + task.description

        filename = f"split-{task.task_id.replace('/', '-')}.md"
        filepath = feedback_path / filename
        try:
            filepath.write_text(content)
            count += 1
        except OSError as e:
            logger.warning("Failed to write split task %s: %s", filepath, e)

    return count
