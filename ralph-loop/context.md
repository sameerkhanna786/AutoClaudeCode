# Shared Context

This file contains research gathered to support all tasks in this ralph loop.
It is loaded before each task to provide shared understanding.

## Project Overview

**Auto Claude Code** is an autonomous development system that runs Claude Code in a continuous loop to discover and fix issues in a target project. It commits validated changes directly to `main`, accepts developer feedback via files, and can improve its own code.

- **Language**: Python 3 (stdlib + pyyaml only)
- **Test framework**: pytest (`python3 -m pytest tests/ -x -q`)
- **Protected files**: `main.py`, `config.yaml` (NEVER modify these)
- **Entry point**: `main.py` (starts orchestrator or parallel coordinator)

## Architecture

```
main.py              # Entry point + watchdog (PROTECTED)
config.yaml          # Configuration (PROTECTED)
config_schema.py     # Load/validate config, defaults
orchestrator.py      # Main loop tying everything together (~550 lines)
coordinator.py       # Parallel coordinator: distributes tasks to workers, merges results (~660 lines)
worker.py            # Parallel worker: runs Claude in a git worktree (~527 lines)
agent_pipeline.py    # Multi-agent pipeline (planner, coder, tester, reviewer) (~687 lines)
task_discovery.py    # Auto-discover tasks (test failures, lint, TODOs, coverage, quality) (~758 lines)
claude_runner.py     # Invoke `claude` CLI, parse JSON response (~537 lines)
model_resolver.py    # Resolve model aliases to actual model IDs at startup
validator.py         # Run test/lint/build commands, determine pass/fail (~333 lines)
git_manager.py       # Snapshot, rollback, commit
feedback.py          # Watch feedback/ dir for developer task files
state.py             # Persist history to state/history.json
cycle_state.py       # Live cycle state for dashboard visibility
state_lock.py        # Thread-safe StateManager wrapper for parallel mode
safety.py            # Lock file, failure counters, disk/rate/cost checks (~540 lines)
shared.py            # Shared prompt builders, commit message helpers (~680 lines)
process_utils.py     # Subprocess utilities with process group kill
conflict_resolver.py # AI-powered merge conflict resolution
coordinator.py       # Parallel coordinator
worker.py            # Parallel worker in git worktrees
dashboard.py         # Web dashboard
task_queue.py        # Task approval queue
notifications.py     # Notification system
telemetry.py         # Metrics computation
cost_predictor.py    # Cost prediction
llm_judges.py        # LLM-based code review judges
provider_runner.py   # Abstract runner supporting multiple providers
session_manager.py   # Session recovery for orphaned worktrees
structured_logging.py # JSON logging support
context_monitor.py   # Context window usage tracking
config_tuner.py      # Auto-tuning configuration
```

## Core Loop Flow

1. Pre-flight safety checks (lock, disk, memory, rate limit, cost, failure count)
2. Check `feedback/` for developer-submitted priority tasks
3. If no feedback: auto-discover tasks (test failures, lint, TODOs, coverage, quality, Claude ideas)
4. De-duplicate against recent history
5. Pick highest-priority task (or adaptive batch)
6. Record git snapshot
7. Invoke Claude Code with task prompt (or multi-agent pipeline)
8. Check changed files (count limit, protected files)
9. Validate: run tests, lint, build (short-circuit on failure)
10. If invalid: retry with failure output (up to max_validation_retries)
11. If valid: commit; If all retries exhausted: rollback
12. Record cycle in state/history.json
13. Sleep and repeat

## Key Patterns and Conventions

- **One external dep**: `pyyaml`. Everything else is stdlib.
- **File-based everything**: state is JSON, feedback is text files, config is YAML.
- **Claude prompt says "do NOT commit"** -- the orchestrator handles all git operations.
- **Self-improvement mode**: when `self_improve: true`, syntax-checks `.py` files before commit.
- **Shared prompt builders**: `shared.py` contains `build_task_prompt()`, `build_plan_prompt()`, `build_execute_prompt()`, `build_retry_prompt()` used by both orchestrator and worker.
- **Task types**: `test_failure` (priority 2), `lint` (priority 2), `todo` (priority 3), `coverage` (priority 4), `claude_idea` (priority 4), `quality` (priority 5), `feedback` (priority 1).
- **Adaptive priority**: tasks from historically high-success-rate sources get boosted priority.
- **Circuit breaker**: in `claude_runner.py`, blocks API calls after repeated failures with exponential backoff.
- **Process group kill**: all subprocesses use `start_new_session=True` and are killed via process group to prevent orphans.

## Key Interfaces

### Task dataclass (`task_discovery.py`)
```python
@dataclass
class Task:
    description: str
    priority: int       # 1=highest, 5=lowest
    source: str         # "test_failure", "lint", "todo", "coverage", "quality", "feedback", "claude_idea"
    source_file: Optional[str] = None
    line_number: Optional[int] = None
    context: str = ""   # rich context: tracebacks, file snippets, error details
    task_id: str = ""
    depends_on: List[str] = field(default_factory=list)
```

### ClaudeResult dataclass (`claude_runner.py`)
```python
@dataclass
class ClaudeResult:
    success: bool
    result_text: str = ""
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    raw_json: Optional[Dict] = None
    error: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    context_window_pct: float = 0.0
```

### ValidationResult dataclass (`validator.py`)
```python
@dataclass
class ValidationResult:
    passed: bool
    steps: List[ValidationStep] = field(default_factory=list)
    # steps contain name, command, passed, output, return_code
```

### WorkerResult dataclass (`worker.py`)
```python
@dataclass
class WorkerResult:
    success: bool
    branch_name: str = ""
    commit_hash: str = ""
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    error: str = ""
    tasks: List[Task] = field(default_factory=list)
```

## Test Structure

Tests are in `tests/` directory:
- `test_orchestrator.py`, `test_coordinator.py`, `test_worker.py` - core loop tests
- `test_agent_pipeline.py` - multi-agent pipeline tests
- `test_claude_runner.py` - CLI runner tests
- `test_task_discovery.py` - task discovery tests
- `test_validator.py` - validation tests
- `test_safety.py` - safety guard tests
- `test_state.py`, `test_state_lock.py` - state management tests
- `test_git_manager.py` - git operations tests
- `test_feedback.py` - feedback system tests
- `test_shared.py` - shared utilities tests
- `test_app.py` - web app tests
- `conftest.py` - shared fixtures

## Important Notes for All Tasks

1. **Never modify `main.py` or `config.yaml`** - they are protected files
2. **Run `python3 -m pytest tests/ -x -q` after changes** to verify nothing breaks
3. **Keep pyyaml as the only external dependency** - use stdlib for everything else
4. **Follow existing patterns** - dataclasses for data, classes for behavior, `logging` for output
5. **Existing tests must keep passing** - if you change behavior, update the tests too
6. **Use `process_utils.run_with_group_kill()`** for any new subprocess invocations
