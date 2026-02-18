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
task_discovery.py    # Auto-discover tasks (test failures, lint, TODOs, coverage, quality)
claude_runner.py     # Invoke `claude` CLI, parse JSON response
validator.py         # Run test/lint/build commands, determine pass/fail
git_manager.py       # Snapshot, rollback, commit
feedback.py          # Watch feedback/ dir for developer task files
state.py             # Persist history to state/history.json
safety.py            # Lock file, failure counters, disk/rate/cost checks
```

## Core Loop

1. Pre-flight safety checks (lock, disk, rate limit, failure count)
2. Check `feedback/` for developer-submitted priority tasks
3. If no feedback: auto-discover tasks (test failures, lint, TODOs, coverage, quality)
4. De-duplicate against recent history
5. Pick highest-priority task
6. Record git snapshot
7. Invoke Claude Code with task prompt
8. Check changed files (count limit, protected files)
9. Validate: run tests, lint, build (short-circuit on failure)
10. If valid: commit; If invalid: rollback
11. Record cycle in `state/history.json`
12. Sleep and repeat

## Conventions

- **One external dependency**: `pyyaml`. Everything else is stdlib.
- **Protected files**: `main.py` and `config.yaml` must never be modified by Claude.
- **Claude prompt says "do NOT commit"** — the orchestrator handles all git operations.
- **File-based everything**: state is JSON, feedback is text files, config is YAML.
- **Self-improvement mode**: when `self_improve: true`, orchestrator syntax-checks modified `.py` files and backs up modules before each cycle.
