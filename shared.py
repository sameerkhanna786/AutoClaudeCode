"""Shared utility functions used across orchestrator, coordinator, and worker."""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import List, Optional

from task_discovery import Task

logger = logging.getLogger(__name__)


def format_task_list(tasks: List[Task]) -> str:
    """Format tasks as a numbered list with source tags and context."""
    lines = []
    for i, task in enumerate(tasks, 1):
        lines.append(f"{i}. {task.description} [{task.source}]")
        if task.context:
            lines.append(f"   CONTEXT:")
            for ctx_line in task.context.split("\n"):
                lines.append(f"   {ctx_line}")
    return "\n".join(lines)


def syntax_check_files(changed_files: List[str], base_dir: str) -> Optional[str]:
    """Syntax-check modified .py files under base_dir.

    Returns an error string if a syntax error is found, None otherwise.
    """
    for f in changed_files:
        if f.endswith(".py"):
            full_path = Path(base_dir) / f
            if full_path.exists():
                try:
                    source = full_path.read_text()
                    ast.parse(source, filename=f)
                except SyntaxError as e:
                    return f"Syntax error in {f} at line {e.lineno}: {e.msg}"
    return None


def gather_tasks(config, feedback_manager, state_manager, discovery) -> List[Task]:
    """Gather all eligible tasks from feedback and auto-discovery.

    Shared between Orchestrator and ParallelCoordinator.
    When config.discovery.adaptive_priority is True, task priorities are
    boosted based on historical success rates per task type.
    """
    tasks: List[Task] = []

    # Priority 1: developer feedback
    max_retries = config.orchestrator.max_feedback_retries
    for task in feedback_manager.get_pending_feedback():
        failure_count = state_manager.get_task_failure_count(
            task.description, "feedback", task_key=task.task_key,
        )
        if failure_count >= max_retries:
            logger.warning(
                "Feedback task failed %d times, moving to failed/",
                failure_count,
            )
            if task.source_file:
                feedback_manager.mark_failed(task.source_file)
            continue
        if not state_manager.was_recently_attempted(
            task.description, task_key=task.task_key,
        ):
            tasks.append(task)

    # Auto-discovered tasks
    discovered = discovery.discover_all()
    for task in discovered:
        if not state_manager.was_recently_attempted(
            task.description, task_key=task.task_key,
        ):
            tasks.append(task)

    # Apply adaptive priority: boost tasks of types with high success rates
    if config.discovery.adaptive_priority and tasks:
        success_rates = state_manager.get_success_rate_by_type(lookback_seconds=86400)
        if success_rates:
            for task in tasks:
                rate = success_rates.get(task.source)
                if rate is not None and rate > 0:
                    # Higher success rate -> lower priority number (higher priority)
                    # Multiply priority by (1 - rate) to boost high-success types
                    # e.g., 80% success rate -> priority * 0.2
                    factor = max(0.1, 1.0 - rate)
                    task.priority = max(1, int(task.priority * factor))

    return tasks


# ------------------------------------------------------------------
# Commit-message helpers (shared between Orchestrator and Worker)
# ------------------------------------------------------------------

# Verb prefix applied to the commit subject based on the task source.
_SOURCE_VERBS = {
    "test_failure": "Fix",
    "lint": "Fix",
    "todo": None,       # handled by _derive_todo_subject()
    "feedback": None,   # human-written, use as-is
    "claude_idea": None,  # already descriptive, use as-is
    "coverage": "Add test coverage for",
    "quality": "Refactor",
}

# Prefixes commonly produced by task_discovery that should be stripped
_STRIP_PREFIXES = [
    "Fix test failure: ",
    "Fix test failure in ",
    "FAILED ",
    "Fix lint error in ",
    "Fix lint error: ",
    "Address TODO in ",
    "Address TODO: ",
    "IDEA: ",
]


def clean_description(desc: str) -> str:
    """Clean an auto-generated task description for use in commit messages."""
    text = desc.strip()

    lower = text.lower()
    for pfx in _STRIP_PREFIXES:
        if lower.startswith(pfx.lower()):
            text = text[len(pfx):]
            break

    text = text.replace("`", "")
    text = re.sub(r'(\.\w+):\d+(?:-\d+)?', r'\1', text)
    text = text.strip()

    if text:
        text = text[0].upper() + text[1:]

    return text


def _derive_todo_subject(description: str) -> str:
    """Derive a commit subject from a TODO task description."""
    text = description.strip()

    m = re.match(
        r'(?:Address\s+)?TODO\s+in\s+'
        r'([a-zA-Z0-9_/.\-]+\.\w+)'
        r'(?::\d+(?:-\d+)?)?'
        r':\s*(.+)',
        text, re.IGNORECASE,
    )
    if m:
        filepath = m.group(1)
        action = m.group(2).strip()
        action = re.sub(r'^(?:FIXME|TODO|XXX)\s*:?\s*', '', action, flags=re.IGNORECASE)
        if action:
            action = action[0].upper() + action[1:]
            return f"{action} in {filepath}"
        return f"Address TODO in {filepath}"

    cleaned = clean_description(text)
    return cleaned if cleaned else "Address TODO"


def extract_file_names(tasks: List[Task]) -> List[str]:
    """Extract file names mentioned in task descriptions."""
    files = []
    for t in tasks:
        m = re.search(
            r'([a-zA-Z0-9_/.\-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|rb|sh|yaml|yml|json|md|txt))',
            t.description,
        )
        if m:
            fname = m.group(1).split("/")[-1]
            if fname not in files:
                files.append(fname)
    return files


def build_commit_message(task: Task) -> str:
    """Build a conventional, human-style commit message for a single task."""
    cleaned = clean_description(task.description)
    verb = _SOURCE_VERBS.get(task.source)

    if task.source == "todo":
        subject = _derive_todo_subject(task.description)
    elif verb is None:
        subject = cleaned
    else:
        if cleaned.lower().startswith(verb.lower()):
            subject = cleaned
        else:
            subject = f"{verb} {cleaned[0].lower() + cleaned[1:]}" if cleaned else verb

    if subject:
        subject = subject[0].upper() + subject[1:]

    if len(subject) > 72:
        full_subject = subject
        truncated = subject[:69]
        last_space = truncated.rfind(" ")
        if last_space > 40:
            truncated = truncated[:last_space]
        subject = truncated + "..."
        return subject + "\n\n" + full_subject
    return subject


def build_batch_commit_message(tasks: List[Task]) -> str:
    """Build a natural commit message summarizing a batch of tasks."""
    sources = set(t.source for t in tasks)

    if len(sources) == 1:
        source = next(iter(sources))
        subject = _summarize_same_source(source, tasks)
    else:
        subject = _summarize_mixed_sources(sources, tasks)

    if len(subject) > 72:
        truncated = subject[:69]
        last_space = truncated.rfind(" ")
        if last_space > 40:
            truncated = truncated[:last_space]
        subject = truncated + "..."

    body_lines = [f"- {clean_description(t.description)}" for t in tasks]
    return subject + "\n\n" + "\n".join(body_lines)


def _summarize_same_source(source: str, tasks: List[Task]) -> str:
    """Generate a summary subject when all tasks share the same source."""
    count = len(tasks)
    files = extract_file_names(tasks)

    if source == "test_failure":
        if files and len(files) <= 2:
            return f"Fix test failures in {' and '.join(files)}"
        return f"Fix test failures in {count} files"

    if source == "lint":
        if files and len(files) <= 2:
            return f"Fix lint errors in {' and '.join(files)}"
        return f"Fix lint errors in {count} files"

    if source == "todo":
        return f"Address TODOs across {count} modules"

    if source == "coverage":
        return f"Add test coverage for {count} modules"

    if source == "quality":
        return f"Refactor {count} modules"

    if source == "claude_idea":
        cleaned = clean_description(tasks[0].description)
        if count > 1:
            return f"{cleaned} and {count - 1} more improvements"
        return cleaned

    if source == "feedback":
        cleaned = clean_description(tasks[0].description)
        if count > 1:
            return f"{cleaned} and {count - 1} more tasks"
        return cleaned

    return f"Apply {count} changes"


def _summarize_mixed_sources(sources: set, tasks: List[Task]) -> str:
    """Generate a summary subject for tasks with mixed source types."""
    parts = []
    source_groups = {}
    for t in tasks:
        source_groups.setdefault(t.source, []).append(t)

    for src in ["test_failure", "lint", "todo", "coverage", "quality",
                 "claude_idea", "feedback"]:
        group = source_groups.get(src)
        if not group:
            continue
        if src == "test_failure":
            parts.append("fix test failures")
        elif src == "lint":
            parts.append("fix lint errors")
        elif src == "todo":
            parts.append("address TODOs")
        elif src == "coverage":
            parts.append("add test coverage")
        elif src == "quality":
            parts.append("refactor")
        elif src in ("claude_idea", "feedback"):
            parts.append(clean_description(group[0].description).lower())

    file_count = len(set().union(*(set(extract_file_names(source_groups[s]))
                                   for s in source_groups)))
    subject_parts = " and ".join(parts[:2])
    if len(parts) > 2:
        subject_parts += f" and {len(parts) - 2} more"

    subject = subject_parts[0].upper() + subject_parts[1:] if subject_parts else "Apply changes"

    if file_count:
        subject += f" in {file_count} files"

    return subject
