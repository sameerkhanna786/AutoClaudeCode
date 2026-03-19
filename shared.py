"""Shared utility functions used across orchestrator, coordinator, and worker."""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from task_discovery import Task

logger = logging.getLogger(__name__)

# Regex to extract file references from task descriptions and context.
# Matches backtick-quoted file paths like `foo.py:123` and unquoted paths
# preceded by "in" or "for".
_FILE_REF_RE = re.compile(
    r'`([a-zA-Z0-9_/.\-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|rb|sh|yaml|yml|json|md|txt))'
    r'(?::(\d+))?(?:-\d+)?`'
)
_FILE_REF_FALLBACK_RE = re.compile(
    r'(?:in\s+|for\s+)([a-zA-Z0-9_/.\-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|rb|sh|yaml|yml|json|md|txt))'
    r'(?::(\d+))?'
)


def _extract_relevant_files(tasks: List[Task], max_files: int = 5) -> List[str]:
    """Extract unique file references from task descriptions and context.

    Returns up to *max_files* unique file paths, ordered by first appearance.
    """
    seen: set = set()
    files: List[str] = []

    for task in tasks:
        texts = [task.description]
        if task.context:
            texts.append(task.context)
        if task.source_file:
            path = task.source_file
            if path not in seen:
                seen.add(path)
                files.append(path)

        for text in texts:
            for match in _FILE_REF_RE.finditer(text):
                path = match.group(1)
                if path not in seen:
                    seen.add(path)
                    files.append(path)
            for match in _FILE_REF_FALLBACK_RE.finditer(text):
                path = match.group(1)
                if path not in seen:
                    seen.add(path)
                    files.append(path)

        if len(files) >= max_files:
            break

    return files[:max_files]


# ------------------------------------------------------------------
# Task dependency ordering (DAG topological sort)
# ------------------------------------------------------------------

def topological_sort_tasks(tasks: List[Task]) -> List[Task]:
    """Topological sort using Kahn's algorithm with priority tie-breaking.

    Uses a pre-built reverse adjacency map and a min-heap for O(E + V log V)
    complexity instead of the previous O(V²) inner loop per node extraction.

    Raises ValueError if a dependency cycle is detected.
    """
    import heapq

    task_map = {t.task_id: t for t in tasks}
    in_degree = {t.task_id: 0 for t in tasks}
    # Build reverse adjacency: dep_id -> list of dependent task_ids
    dependents: Dict[str, List[str]] = {t.task_id: [] for t in tasks}
    for t in tasks:
        for dep in t.depends_on:
            if dep in task_map:
                in_degree[t.task_id] += 1
                dependents[dep].append(t.task_id)
    # Use a heap for O(log n) priority extraction instead of repeated sorting
    counter = 0  # tie-breaker for stable ordering
    heap: List[tuple] = []
    for t in tasks:
        if in_degree[t.task_id] == 0:
            heapq.heappush(heap, (t.priority, counter, t))
            counter += 1
    result: List[Task] = []
    while heap:
        _, _, task = heapq.heappop(heap)
        result.append(task)
        for dep_id in dependents[task.task_id]:
            in_degree[dep_id] -= 1
            if in_degree[dep_id] == 0:
                heapq.heappush(heap, (task_map[dep_id].priority, counter, task_map[dep_id]))
                counter += 1
    if len(result) != len(tasks):
        raise ValueError("Cycle detected in task dependencies")
    return result


# ------------------------------------------------------------------
# Task-type-specific instructions appended to prompts
# ------------------------------------------------------------------
TASK_TYPE_INSTRUCTIONS = {
    "test_failure": """\
- Read the failing test(s) carefully. Understand what the test expects.
- The traceback in the CONTEXT section shows exactly where the failure occurs.
- Determine whether the bug is in the implementation or the test.
- Fix whichever side is wrong. You may fix the code, fix the tests, or both.
- Run the specific failing test to verify your fix.""",

    "lint": """\
- The lint error details are in the CONTEXT section showing the exact line.
- Fix only the specific lint violations listed. Do not refactor unrelated code.
- For style issues (line length, whitespace), make minimal formatting changes.
- For semantic issues (unused imports, undefined names), fix the root cause.""",

    "todo": """\
- The TODO/FIXME comment and surrounding code are in the CONTEXT section.
- Address the intent of the comment. Remove the TODO/FIXME marker when done.
- If the TODO requires significant design decisions, implement the simplest correct approach.
- Add or update tests if your change modifies behavior.""",

    "coverage": """\
- Write tests for the uncovered code paths in the specified file.
- Focus on testing meaningful behavior, not just line coverage.
- Follow the existing test patterns in the project's test directory.
- Use descriptive test names that explain what scenario is being tested.""",

    "quality": """\
- Focus on improving code clarity and maintainability.
- Break up overly long functions or files into logical units.
- Do NOT change behavior. All existing tests must continue to pass.""",

    "claude_idea": """\
- Implement the improvement described above.
- Be thorough but conservative. Follow existing code patterns.
- Add tests for any new functionality you introduce.""",

    "feedback": """\
- This task was written by a human developer. Follow it precisely.
- If the instructions are ambiguous, make the most reasonable interpretation.
- After implementing, verify that your changes match the developer's intent.""",
}


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


def gather_tasks(config, feedback_manager, state_manager, discovery,
                  dashboard_active: bool = False,
                  task_approval_queue=None) -> List[Task]:
    """Gather all eligible tasks from feedback and auto-discovery.

    Shared between Orchestrator and ParallelCoordinator.
    When config.discovery.adaptive_priority is True, task priorities are
    boosted based on historical success rates per task type.

    When dashboard_active is True and task_approval_queue is provided and
    config.orchestrator.task_approval is True:
    - Feedback tasks are returned immediately (bypass approval gate)
    - Auto-discovered tasks are enqueued for approval and excluded from return
    - Previously approved tasks are included in the return list
    """
    tasks: List[Task] = []
    approval_active = (
        dashboard_active
        and task_approval_queue is not None
        and getattr(config.orchestrator, 'task_approval', True)
    )

    # Priority 1: developer feedback (always bypass approval gate)
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
    discovered_eligible = []
    for task in discovered:
        if not state_manager.was_recently_attempted(
            task.description, task_key=task.task_key,
        ):
            discovered_eligible.append(task)

    if approval_active:
        # Enqueue auto-discovered tasks for approval instead of returning them
        # Exception: test_failure tasks are always safe to attempt (the test
        # already exists and is already failing), so auto-approve them.
        cooldown = config.discovery.idea_cooldown_seconds
        enqueued_count = 0
        for task in discovered_eligible:
            if task.source == "test_failure":
                tasks.append(task)
            else:
                task_approval_queue.enqueue(task, cooldown_seconds=cooldown)
                enqueued_count += 1

        # Include previously approved tasks
        approved = task_approval_queue.get_approved()
        tasks.extend(approved)

        if enqueued_count and not approved:
            pending_total = task_approval_queue.pending_count() if hasattr(task_approval_queue, 'pending_count') else enqueued_count
            logger.info(
                "Task approval gate active: %d task(s) pending approval in dashboard, "
                "0 approved. Approve tasks at the dashboard to proceed.",
                pending_total,
            )
    else:
        # No approval gate — include all discovered tasks directly
        tasks.extend(discovered_eligible)

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

    # Apply topological sort if any tasks have dependencies
    if any(t.depends_on for t in tasks):
        try:
            tasks = topological_sort_tasks(tasks)
        except ValueError as e:
            logger.warning("Task dependency cycle: %s", e)
            tasks.sort(key=lambda t: t.priority)

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
    seen: set = set()
    files: List[str] = []
    for t in tasks:
        m = re.search(
            r'([a-zA-Z0-9_/.\-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|rb|sh|yaml|yml|json|md|txt))',
            t.description,
        )
        if m:
            fname = m.group(1).split("/")[-1]
            if fname not in seen:
                seen.add(fname)
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


# ------------------------------------------------------------------
# Shared prompt builders (used by both Orchestrator and Worker)
# ------------------------------------------------------------------

def _working_dir_preamble(working_dir: Optional[str] = None) -> str:
    """Build the working directory preamble for prompts.

    When working_dir is set (worker mode), tells Claude to use absolute paths
    within that directory and to not modify files outside it.
    When None (orchestrator mode), uses simpler "current directory" phrasing.
    """
    if working_dir:
        return (
            f"You are working on the project at {working_dir}.\n"
            "All file reads, writes, and edits MUST use absolute paths within that directory.\n"
            "WARNING: Do NOT modify any files outside that directory. Do NOT use relative paths.\n"
        )
    return "You are working on the project in the current directory.\n"


def build_task_prompt(
    tasks: List[Task],
    protected_files: List[str],
    working_dir: Optional[str] = None,
) -> str:
    """Build a direct execution prompt for one or more tasks.

    Args:
        tasks: The task(s) to include.
        protected_files: Files that must not be modified.
        working_dir: If set, absolute path used in worktree preamble.
    """
    protected = ", ".join(protected_files)
    preamble = _working_dir_preamble(working_dir)

    # Build RELEVANT FILES section from file references in tasks
    relevant = _extract_relevant_files(tasks)
    relevant_section = ""
    if relevant:
        lines = "\n".join(f"- {f}" for f in relevant)
        relevant_section = f"\nRELEVANT FILES:\n{lines}\n"

    if len(tasks) > 1:
        task_list = format_task_list(tasks)
        return (
            f"{preamble}\n"
            "You have been given a batch of tasks to address in a single comprehensive change.\n\n"
            f"TASKS:\n{task_list}\n"
            f"{relevant_section}\n"
            "INSTRUCTIONS:\n"
            "- Make the minimal changes needed to complete ALL tasks above.\n"
            "- Do NOT run git commands (add, commit, push). The orchestrator handles git.\n"
            f"- Do NOT modify these protected files: {protected}\n"
            "- Focus on correctness. Run tests if available.\n"
            "- If a task is unclear or impossible, make your best effort and explain what you did.\n"
            "- Use the CONTEXT provided with each task to understand the code and errors involved.\n"
            "- Make your changes immediately. Do NOT spend turns exploring the codebase — go directly to editing files.\n"
        )

    task = tasks[0]
    context_section = ""
    if task.context:
        context_section = f"\nCONTEXT:\n{task.context}\n"
    specific_instructions = TASK_TYPE_INSTRUCTIONS.get(task.source, "")
    return (
        f"{preamble}\n"
        f"TASK: {task.description}\n"
        f"{context_section}"
        f"{relevant_section}\n"
        "INSTRUCTIONS:\n"
        "- Make the minimal changes needed to complete this task.\n"
        "- Do NOT run git commands (add, commit, push). The orchestrator handles git.\n"
        f"- Do NOT modify these protected files: {protected}\n"
        "- Focus on correctness. Run tests if available.\n"
        "- If the task is unclear or impossible, make your best effort and explain what you did.\n"
        f"{specific_instructions}\n"
        "- Make your changes immediately. Do NOT spend turns exploring the codebase — go directly to editing files.\n"
    )


def build_plan_prompt(
    tasks: List[Task],
    protected_files: List[str],
    working_dir: Optional[str] = None,
) -> str:
    """Build a planning-only prompt (no file changes).

    Args:
        tasks: The task(s) to plan for.
        protected_files: Files that must not be modified.
        working_dir: If set, absolute path used in worktree preamble.
    """
    protected = ", ".join(protected_files)
    preamble = _working_dir_preamble(working_dir)

    if len(tasks) > 1:
        task_list = format_task_list(tasks)
        task_count = len(tasks) + 1
        task_count_plus1 = len(tasks) + 2
        return (
            f"{preamble}\n"
            "You have been given a batch of tasks to address in a single comprehensive change.\n\n"
            f"TASKS:\n{task_list}\n\n"
            f"ADDITIONAL CHECKS (always perform these):\n"
            f"{task_count}. Check whether any of the above changes require NEW tests to be added. "
            "If new functionality is introduced or existing behavior is changed, plan to add or update tests.\n"
            f"{task_count_plus1}. Check whether README.md needs updating to reflect any of the above changes. "
            "If user-facing behavior, configuration options, or architecture changed, plan to update README.md.\n\n"
            "INSTRUCTIONS:\n"
            "- Analyze the codebase and create a detailed, comprehensive plan that addresses ALL tasks above.\n"
            "- Do NOT make any changes yet. Only output a plan.\n"
            "- List every file you would modify and what changes you would make in each.\n"
            f"- Do NOT modify these protected files: {protected}\n"
            "- Be specific about the changes (function names, line numbers, etc.).\n"
            "- Group related changes together where possible for clarity.\n"
            "- Address the tasks in priority order but look for opportunities to combine related changes.\n"
            "- Use the CONTEXT provided with each task to understand the code and errors involved.\n"
            "- Output your complete plan within 5 turns. Do NOT spend turns reading files — focus on producing the plan.\n"
            "\nOUTPUT FORMAT:\n"
            "Output your plan as a numbered list where each item specifies:\n"
            "  FILE: <path/to/file>\n"
            "  CHANGE_TYPE: add | modify | delete\n"
            "  DESCRIPTION: <what you will change and why>\n"
        )

    task = tasks[0]
    context_section = ""
    if task.context:
        context_section = f"\nCONTEXT:\n{task.context}\n"
    specific_instructions = TASK_TYPE_INSTRUCTIONS.get(task.source, "")
    return (
        f"{preamble}\n"
        f"TASK: {task.description}\n"
        f"{context_section}\n"
        "INSTRUCTIONS:\n"
        "- Analyze the codebase and create a detailed plan to complete this task.\n"
        "- Do NOT make any changes yet. Only output a plan.\n"
        "- List the files you would modify and what changes you would make.\n"
        f"- Do NOT modify these protected files: {protected}\n"
        "- Be specific about the changes (function names, line numbers, etc.).\n"
        f"{specific_instructions}\n"
        "- Output your complete plan within 5 turns. Do NOT spend turns reading files — focus on producing the plan.\n"
        "\nOUTPUT FORMAT:\n"
        "Output your plan as a numbered list where each item specifies:\n"
        "  FILE: <path/to/file>\n"
        "  CHANGE_TYPE: add | modify | delete\n"
        "  DESCRIPTION: <what you will change and why>\n"
    )


def build_execute_prompt(
    tasks: List[Task],
    plan_text: str,
    protected_files: List[str],
    working_dir: Optional[str] = None,
) -> str:
    """Build an execution prompt that includes a pre-made plan.

    Args:
        tasks: The task(s) to execute.
        plan_text: The plan generated by the planning phase.
        protected_files: Files that must not be modified.
        working_dir: If set, absolute path used in worktree preamble.
    """
    protected = ", ".join(protected_files)
    preamble = _working_dir_preamble(working_dir)

    if len(tasks) > 1:
        task_list = format_task_list(tasks)
        return (
            f"{preamble}\n"
            "You have been given a batch of tasks to address in a single comprehensive change.\n\n"
            f"TASKS:\n{task_list}\n\n"
            f"PLAN TO EXECUTE:\n{plan_text}\n\n"
            "INSTRUCTIONS:\n"
            "- Execute the plan above by making ALL described changes.\n"
            "- Do NOT run git commands (add, commit, push). The orchestrator handles git.\n"
            f"- Do NOT modify these protected files: {protected}\n"
            "- Focus on correctness. Run tests after making changes.\n"
            "- Stick to the plan. Do not deviate unless the plan has an obvious error.\n"
            "- Make ALL changes in this single session. This is a comprehensive revamp, not incremental.\n"
            "- Use the CONTEXT provided with each task to understand the code and errors involved.\n"
            "- Make your changes immediately. Do NOT spend turns exploring the codebase — go directly to editing files.\n"
        )

    task = tasks[0]
    context_section = ""
    if task.context:
        context_section = f"\nCONTEXT:\n{task.context}\n"
    specific_instructions = TASK_TYPE_INSTRUCTIONS.get(task.source, "")
    return (
        f"{preamble}\n"
        f"TASK: {task.description}\n"
        f"{context_section}\n"
        f"PLAN TO EXECUTE:\n{plan_text}\n\n"
        "INSTRUCTIONS:\n"
        "- Execute the plan above by making the described changes.\n"
        "- Do NOT run git commands (add, commit, push). The orchestrator handles git.\n"
        f"- Do NOT modify these protected files: {protected}\n"
        "- Focus on correctness. Run tests if available.\n"
        "- Stick to the plan. Do not deviate unless the plan has an obvious error.\n"
        f"{specific_instructions}\n"
        "- Make your changes immediately. Do NOT spend turns exploring the codebase — go directly to editing files.\n"
    )


_COMMON_FAILURE_PATTERNS: Dict[str, str] = {
    "test_failure": (
        "- Wrong assertion value or outdated expected result\n"
        "- Missing import in the module under test\n"
        "- Function signature changed but callers not updated\n"
        "- Fixture or mock not matching new behavior\n"
        "- Off-by-one errors in boundary conditions"
    ),
    "lint": (
        "- Unused import not removed after refactoring\n"
        "- Line too long (exceeds configured max length)\n"
        "- Trailing whitespace or missing newline at end of file\n"
        "- Variable assigned but never used\n"
        "- Wrong import order (stdlib vs third-party vs local)"
    ),
    "todo": (
        "- Implementation incomplete — only part of the TODO was addressed\n"
        "- New code introduced a test regression\n"
        "- Missing error handling for edge cases mentioned in the TODO"
    ),
    "quality": (
        "- Refactoring changed behavior instead of just restructuring\n"
        "- Extracted function has wrong parameter list or return value\n"
        "- Existing tests rely on the old structure"
    ),
    "coverage": (
        "- New test imports the wrong module or function name\n"
        "- Test assertions don't match actual function behavior\n"
        "- Missing test fixtures or setup for the code under test"
    ),
    "claude_idea": (
        "- Change scope was too broad — unrelated code was affected\n"
        "- New code conflicts with existing patterns or conventions\n"
        "- Missing edge case handling that existing tests exercise"
    ),
}


def _common_failure_patterns(tasks: List[Task]) -> str:
    """Return a COMMON FAILURE PATTERNS section based on task types."""
    sources = {t.source for t in tasks if t.source}
    tips = []
    for source in sorted(sources):
        patterns = _COMMON_FAILURE_PATTERNS.get(source)
        if patterns:
            tips.append(f"For {source} tasks:\n{patterns}")
    if not tips:
        return ""
    return "COMMON FAILURE PATTERNS:\n" + "\n".join(tips) + "\n"


def _format_task_history(task_history: List[Dict]) -> str:
    """Format previous attempt history for inclusion in retry prompts."""
    if not task_history:
        return ""
    failed = [h for h in task_history if not h.get("success", False)]
    if not failed:
        return ""
    lines = ["PREVIOUS FAILED ATTEMPTS (do NOT repeat the same mistakes):"]
    for i, attempt in enumerate(failed, 1):
        error = attempt.get("error", "").strip()
        summary = attempt.get("validation_summary", "").strip()
        reason = error or summary or "unknown error"
        if len(reason) > 300:
            reason = reason[:297] + "..."
        lines.append(f"  Attempt {i}: {reason}")
    lines.append("")
    return "\n".join(lines) + "\n"


def build_retry_prompt(
    tasks: List[Task],
    failure_output: str,
    protected_files: List[str],
    working_dir: Optional[str] = None,
    attempt: int = 0,
    max_attempts: int = 0,
    task_history: Optional[List[Dict]] = None,
) -> str:
    """Build a retry prompt with validation failure output.

    Args:
        tasks: The task(s) being retried.
        failure_output: The validation failure output.
        protected_files: Files that must not be modified.
        working_dir: If set, absolute path used in worktree preamble.
        attempt: Current attempt number (for orchestrator-style formatting).
        max_attempts: Total max attempts (for orchestrator-style formatting).
        task_history: Previous attempt history from StateManager.get_task_success_history().
    """
    protected = ", ".join(protected_files)
    preamble = _working_dir_preamble(working_dir)

    # Truncate failure output to avoid exceeding prompt limits
    max_output = 8000
    if len(failure_output) > max_output:
        failure_output = failure_output[:max_output] + "\n... (truncated)"

    if len(tasks) > 1:
        task_list = format_task_list(tasks)
        task_section = f"TASKS:\n{task_list}"
    else:
        task_section = f"TASK: {tasks[0].description}"

    attempt_info = ""
    if attempt and max_attempts:
        attempt_info = f" (attempt {attempt} of {max_attempts})"

    return (
        f"{preamble}\n"
        f"ORIGINAL {task_section}\n\n"
        f"YOUR PREVIOUS CHANGES FAILED VALIDATION{attempt_info}.\n\n"
        f"VALIDATION FAILURES:\n"
        f"```\n{failure_output}\n```\n\n"
        "INSTRUCTIONS:\n"
        "- Read the failure output above carefully.\n"
        "- Determine whether the bug is in the code you changed or in the tests.\n"
        "  Sometimes the test expectation is wrong (e.g., wrong return value asserted).\n"
        "  Other times the implementation has the actual bug.\n"
        "- Fix whichever side is wrong. You may fix the code, fix the tests, or both.\n"
        "- Do NOT run git commands (add, commit, push). The orchestrator handles git.\n"
        f"- Do NOT modify these protected files: {protected}\n"
        "- Your previous changes are still in the working tree. Build on them, do not start over.\n"
        "- Focus on making ALL validations pass.\n\n"
        f"{_format_task_history(task_history or [])}"
        f"{_common_failure_patterns(tasks)}"
    )
