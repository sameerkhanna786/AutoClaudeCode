# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Auto Claude Code** is an autonomous development system that runs Claude Code in a continuous loop to discover and fix issues in a target project. It commits validated changes directly to `main`, accepts optional developer feedback via files, and can improve its own code.

## Commands

- **Run tests**: `python3 -m pytest tests/ -v`
- **Run single cycle**: `python3 main.py --once`
- **Run continuous**: `python3 main.py`
- **Custom config**: `python3 main.py --config path/to/config.yaml`
- **Install deps**: `python3 -m pip install -r requirements.txt`

## Architecture

```
main.py              # Entry point + watchdog (PROTECTED)
config.yaml          # Configuration (PROTECTED)
config_schema.py     # Load/validate config, defaults
orchestrator.py      # Main loop tying everything together
coordinator.py       # Parallel coordinator: distributes tasks to workers, merges results
worker.py            # Parallel worker: runs Claude in a git worktree
agent_pipeline.py    # Multi-agent pipeline (planner, coder, tester, reviewer)
task_discovery.py    # Auto-discover tasks (test failures, lint, TODOs, coverage, quality)
claude_runner.py     # Invoke `claude` CLI, parse JSON response
model_resolver.py    # Resolve model aliases to actual model IDs at startup
validator.py         # Run test/lint/build commands, determine pass/fail
git_manager.py       # Snapshot, rollback, commit
feedback.py          # Watch feedback/ dir for developer task files
state.py             # Persist history to state/history.json
cycle_state.py       # Live cycle state for dashboard visibility (current_cycle.json)
state_lock.py        # Thread-safe StateManager wrapper for parallel mode
safety.py            # Lock file, failure counters, disk/rate/cost checks
```

## Core Loop

1. Pre-flight safety checks (lock, disk, rate limit, failure count)
2. Check `feedback/` for developer-submitted priority tasks
3. If no feedback: auto-discover tasks (test failures, lint, TODOs, coverage, quality)
4. De-duplicate against recent history
5. Pick highest-priority task (or adaptive batch of tasks)
6. Record git snapshot
7. Invoke Claude Code with task prompt (or run multi-agent pipeline if enabled)
8. Check changed files (count limit, protected files)
9. Validate: run tests, lint, build (short-circuit on failure)
10. If invalid: retry with failure output (up to `max_validation_retries` times)
11. If valid: commit; If all retries exhausted: rollback
12. Record cycle in `state/history.json`
13. Sleep and repeat

## Conventions

- **One external dependency**: `pyyaml`. Everything else is stdlib.
- **Protected files**: `main.py` and `config.yaml` must never be modified by Claude.
- **Claude prompt says "do NOT commit"** — the orchestrator handles all git operations.
- **File-based everything**: state is JSON, feedback is text files, config is YAML.
- **Self-improvement mode**: when `self_improve: true`, orchestrator syntax-checks modified `.py` files and backs up modules before each cycle.
- **Multi-agent mode**: when `agent_pipeline.enabled: true`, each cycle runs a Planner → Coder → Tester → Reviewer pipeline instead of a single Claude invocation.
- **Parallel mode**: when `parallel.enabled: true`, a `ParallelCoordinator` distributes tasks to multiple workers running in separate git worktrees, then merges validated results back to main.
- **Adaptive batch sizing**: batch size grows on success and shrinks on failure, within configured min/max bounds.
- **Discovery prompt customization**: set `discovery.discovery_prompt` in config to steer Claude idea discovery toward specific focus areas. Recent task history is automatically injected to avoid repetitive suggestions.
