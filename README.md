# Auto Claude Code

An autonomous development system that runs Claude Code in a continuous loop to discover and fix issues, automatically committing validated changes.

## Features

- **Autonomous task discovery** — finds test failures, lint errors, TODOs, and generates Claude-powered improvement ideas
- **Plan-then-execute mode** — Claude plans changes before implementing for higher quality results
- **Multi-agent pipeline** — specialized Planner, Coder, Tester, and Reviewer agents collaborate with revision loops for higher quality
- **Automatic validation** — runs tests/lint/build after every change, rolls back on failure
- **Developer feedback system** — drop task files in `feedback/` to steer priorities
- **Self-improvement mode** — can modify its own source code with syntax checking and backups
- **Safety guards** — rate limits, cost limits, disk checks, protected files, lock file
- **File-based state** — JSON history, YAML config, text feedback; no database required

## Quick Start

### Prerequisites

- Python 3.8+
- [Claude CLI](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated (`claude`)
- Git

### Install

```bash
pip install -r requirements.txt  # only pyyaml
```

### Run

```bash
# Single cycle
python3 main.py --once

# Continuous loop
python3 main.py

# Custom config
python3 main.py --config path/to/config.yaml

# Target a different project directory
python3 main.py --once --target-dir /path/to/project
```

## How It Works

Each cycle follows a 12-step loop:

1. **Safety checks** — verify lock file, disk space, rate limits, failure count
2. **Check feedback/** — look for developer-submitted priority tasks
3. **Auto-discover tasks** — scan for test failures, lint errors, TODOs, coverage gaps, quality issues
4. **Deduplicate** — skip tasks already addressed in recent history
5. **Pick task** — select the highest-priority task
6. **Git snapshot** — record current HEAD for potential rollback
7. **Invoke Claude** — send the task prompt to the Claude CLI
8. **Check changed files** — enforce file count limit and protected file rules
9. **Validate** — run test, lint, and build commands (short-circuit on first failure)
10. **Commit or rollback** — commit valid changes; rollback invalid ones
11. **Record history** — append cycle result to `state/history.json`
12. **Sleep & repeat**

## Configuration

All settings live in `config.yaml`. Key sections:

| Section | What it controls |
|---|---|
| `claude` | Model (`opus`), max turns, timeout, CLI command |
| `orchestrator` | Loop interval, self-improve toggle, plan-before-execute, push-after-commit, cycle timeout |
| `validation` | Test/lint/build commands and their timeouts |
| `discovery` | Toggle each discovery strategy, TODO patterns, excluded directories |
| `safety` | Max consecutive failures, cycles/hour, cost/hour, min disk space, protected files |
| `paths` | Feedback, state, history, lock file, and backup directories |
| `logging` | Log level, file path, rotation size and backup count |
| `agent_pipeline` | Multi-agent mode: enable/disable, per-agent model and timeout, max revision loops |

See [`config.yaml`](config.yaml) for the full annotated configuration.

## Developer Feedback

Submit priority tasks by dropping files into the `feedback/` directory:

1. Create a `.md` or `.txt` file in `feedback/`
2. Prefix with numbers for priority ordering (e.g., `01-fix-bug.md`, `02-add-feature.txt`)
3. Write the task description as the file content — this is sent directly to Claude
4. After processing, completed tasks are moved to `feedback/done/`

## Architecture

```
auto_claude_code/
├── main.py              # Entry point + two-layer watchdog (PROTECTED)
├── config.yaml          # Configuration (PROTECTED)
├── config_schema.py     # Load/validate config, apply defaults
├── orchestrator.py      # Main loop tying everything together
├── agent_pipeline.py    # Multi-agent pipeline: planner, coder, tester, reviewer
├── task_discovery.py    # Auto-discover tasks (tests, lint, TODOs, coverage, quality)
├── claude_runner.py     # Invoke claude CLI, parse JSON response
├── model_resolver.py    # Resolve model aliases to actual model IDs at startup
├── validator.py         # Run test/lint/build commands, determine pass/fail
├── git_manager.py       # Snapshot, rollback, commit
├── feedback.py          # Watch feedback/ dir for developer task files
├── state.py             # Persist history to state/history.json
├── safety.py            # Lock file, failure counters, disk/rate/cost checks
├── requirements.txt     # Dependencies (pyyaml)
├── CLAUDE.md            # Claude Code project instructions
├── feedback/            # Drop task files here
│   └── done/            # Completed tasks moved here
├── state/
│   ├── history.json     # Cycle history
│   ├── auto_claude.log  # Log file
│   ├── lock.pid         # Lock file for single-instance enforcement
│   ├── agent_workspace/ # Inter-agent communication files (multi-agent mode)
│   └── backups/         # Module backups (self-improvement mode)
└── tests/               # Pytest test suite
    ├── conftest.py
    ├── test_claude_runner.py
    ├── test_config_schema.py
    ├── test_feedback.py
    ├── test_git_manager.py
    ├── test_integration.py
    ├── test_orchestrator.py
    ├── test_safety.py
    ├── test_state.py
    ├── test_task_discovery.py
    ├── test_validator.py
    ├── test_model_resolver.py
    └── test_agent_pipeline.py
```

| Module | Role |
|---|---|
| `main.py` | Entry point with argument parsing and a two-layer watchdog that recovers from import failures |
| `orchestrator.py` | Runs the core loop, coordinates all other modules |
| `agent_pipeline.py` | Orchestrates the multi-agent pipeline: Planner → Coder → Tester → Reviewer with revision loops |
| `task_discovery.py` | Discovers work: test failures, lint errors, TODOs, coverage gaps, quality issues, Claude ideas |
| `claude_runner.py` | Invokes the `claude` CLI and parses the JSON response |
| `model_resolver.py` | Resolves model aliases (e.g., "opus") to actual model IDs at startup |
| `validator.py` | Runs test/lint/build commands and reports pass/fail |
| `git_manager.py` | Manages git snapshots, rollbacks, and commits |
| `feedback.py` | Reads developer task files from `feedback/` and moves completed ones to `feedback/done/` |
| `state.py` | Persists cycle history to `state/history.json` |
| `safety.py` | Enforces lock file, rate limits, cost limits, disk space checks, and protected file rules |
| `config_schema.py` | Loads and validates `config.yaml`, applies defaults |

## Safety Features

- **Lock file** — `state/lock.pid` prevents concurrent runs of the system
- **Protected files** — `main.py` and `config.yaml` cannot be modified by Claude
- **Rate limiting** — configurable max cycles per hour (default: 30)
- **Cost limiting** — configurable max USD per hour (default: $10)
- **Consecutive failure circuit breaker** — stops after N consecutive failures (default: 5)
- **Disk space check** — aborts if free space drops below threshold (default: 500 MB)
- **Stale lock cleanup** — automatically detects and cleans up lock files from crashed processes
- **Exponential backoff** — rate-limited API requests use exponential backoff for retry delays
- **Two-layer watchdog** — `main.py` catches import failures and keeps the system running

## Self-Improvement Mode

When `self_improve: true` in `config.yaml`:

- The system can modify its own Python modules (everything except protected files)
- All orchestrator source files are backed up to `state/backups/` before each cycle
- Every modified `.py` file is syntax-checked before committing
- If syntax checking fails, changes are rolled back

## Multi-Agent Pipeline

When `agent_pipeline.enabled: true` in `config.yaml`, each cycle uses a pipeline of specialized agents instead of a single Claude invocation:

1. **Planner** — analyzes the task and writes a detailed implementation plan (does not modify code)
2. **Coder** — implements the plan by modifying source files
3. **Tester** — writes or updates tests to cover the changes
4. **Reviewer** — reviews all changes and approves or requests revisions

If the Reviewer requests revisions, changes are rolled back and the Coder tries again with the feedback (up to `max_revisions` times, default: 2). Each agent can use a different model and timeout.

Example configuration:

```yaml
agent_pipeline:
  enabled: true
  max_revisions: 2
  planner:
    model: opus
    max_turns: 10
  coder:
    model: opus
    max_turns: 25
  tester:
    model: opus
    max_turns: 15
  reviewer:
    model: opus
    max_turns: 10
```

Agents communicate via files in `state/agent_workspace/`. The pipeline integrates with the existing validation and git management — changes are only committed if tests/lint/build pass.

## Testing

```bash
python3 -m pytest tests/ -v
```

The test suite covers all modules with unit and integration tests.
