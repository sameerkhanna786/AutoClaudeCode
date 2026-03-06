# Auto Claude Code

### The Self-Improving Autonomous Developer

Point it at a repo, go to sleep, wake up to committed, tested, validated improvements. No PRDs, no spec files, no manual orchestration — Auto Claude Code discovers what needs fixing, fixes it, validates the fix, and commits. Continuously.

---

## Quick Start

```bash
pip install -r requirements.txt    # only pyyaml
cp config.yaml.example config.yaml # edit target_dir
python3 main.py                    # autonomous mode
```

Three commands. Zero configuration required for defaults. It discovers tasks automatically.

---

## How It Works

```
                ┌───────────────────────────────────────────────────────┐
                │                    AUTO CLAUDE CODE                   │
                │                                                       │
feedback/*.md ─►│  ┌────────┐   ┌────────┐   ┌──────────┐  ┌────────┐ │
                │  │  TASK  │──►│ CLAUDE │──►│ VALIDATE │─►│ COMMIT │ │
auto-discover ─►│  │ GATHER │   │ INVOKE │   │test/lint │  │or RETRY│ │
(tests, lint,   │  └───┬────┘   └───┬────┘   └──────────┘  └───┬────┘ │
 TODOs, ideas)  │      │            │                            │      │
                │      ▼            ▼                            ▼      │
                │  ┌────────┐   ┌────────┐                  ┌────────┐ │
                │  │ DEDUP  │   │  LLM   │                  │HISTORY │ │
                │  │ + DAG  │   │ JUDGES │                  │+ LEARN │ │
                │  └────────┘   └────────┘                  └────────┘ │
                └───────────────────────────────────────────────────────┘
```

### The 13-Step Loop

1. **Safety checks** — lock file, disk space, rate limits, cost budget, memory
2. **Check feedback/** — developer-submitted priority tasks
3. **Auto-discover** — test failures, lint errors, TODOs, coverage gaps, quality issues, Claude ideas
4. **Deduplicate** — skip tasks already addressed in recent history
5. **Pick tasks** — adaptive batch sizing based on success rate
6. **Git snapshot** — record HEAD for potential rollback
7. **Invoke LLM** — Claude CLI, OpenAI API, or Gemini API (plan-then-execute optional)
8. **LLM Judges** — independent security, quality, and architecture review
9. **Validate** — run test, lint, build commands (short-circuit on failure)
10. **Retry on failure** — re-invoke with failure output (up to N retries)
11. **Commit or rollback** — commit valid changes; rollback if all retries exhausted
12. **Record history** — append to `state/history.json`, update learning
13. **Smart split & sleep** — split exhausted tasks, sleep, repeat

---

## Feature Highlights

### Auto Task Discovery
```yaml
discovery:
  enable_test_failures: true    # Parse pytest failures
  enable_lint_errors: true      # Parse ruff/flake8 JSON output
  enable_todos: true            # Scan TODO/FIXME/HACK comments
  enable_coverage: false        # Low-coverage modules
  enable_claude_ideas: false    # AI-generated improvements
  enable_quality_review: false  # Large file detection
  discovery_prompt: "Focus on performance and security"  # Custom focus
```
Zero setup required. Discovers work from your existing test suite, linter, and codebase.

### Parallel Workers
```yaml
parallel:
  enabled: true
  max_workers: 3
  merge_strategy: rebase  # "rebase" or "merge"
```
Each worker runs in an isolated git worktree. AI-powered merge conflict resolution handles concurrent changes automatically.

### AI Merge Conflict Resolution
When parallel workers produce conflicting changes, Claude analyzes the conflict markers and resolves them intelligently — understanding the intent of both changes rather than blindly picking sides.

### LLM Judges
```yaml
judges:
  enabled: true
  security:
    enabled: true
    model: sonnet
  quality:
    enabled: true
  architecture:
    enabled: false
  fail_action: retry  # "retry" | "rollback" | "warn"
```
Independent AI quality evaluation beyond "tests pass." SecurityJudge checks for secrets, injection, and auth bypass. QualityJudge catches test slop, dead code, and naming issues. ArchitectureJudge detects circular deps and layer violations.

### Multi-Agent Pipeline
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
Specialized Planner → Coder → Tester → Reviewer pipeline with revision loops. Each agent can use a different model and timeout.

### GitHub PR Integration
```yaml
github:
  enabled: true
  create_prs: true
  auto_merge: false
  label: auto-claude
```
Automatically create pull requests for each change instead of committing directly to main.

### Config Auto-Tuning
Analyzes cycle history and recommends config adjustments. Detects when batch sizes are too large, timeouts too short, or retry counts too high.

### Learning from History
Adaptive priority boosting based on historical success rates. Task types with high success rates get prioritized; struggling task types get deprioritized to save cost.

### Smart Zone Enforcement
```yaml
orchestrator:
  smart_split: true
  max_split_depth: 3
  max_context_pct: 80.0
```
Detects when Claude exhausts its context window (>80% usage or max_turns hit) and automatically splits incomplete tasks into follow-up tasks with dependency tracking.

### Session Recovery
```yaml
orchestrator:
  session_recovery: true
```
Saves progress at key checkpoints. On crash, resumes from last checkpoint. Detects and cleans up orphaned worktrees from crashed parallel workers.

### Provider Agnostic
```yaml
claude:
  provider: claude    # "claude" | "openai" | "gemini"
  api_key_env: ""     # env var name for API key
```
Switch between Claude CLI, OpenAI chat completions, and Gemini generateContent with a single config change. Uses stdlib `urllib` — no additional dependencies.

### PRD Generation & Import
```bash
python3 prd_cli.py generate --config config.yaml --output prd.yaml
python3 prd_cli.py import my-prd.yaml --feedback-dir feedback/
```
Auto-generate PRDs from discovered tasks with performance data. Import Tomacco-format or custom PRDs. Drop `.prd.yaml` files in `feedback/` for auto-import.

### Task DAG with Dependencies
```markdown
---
task_id: setup-auth
depends_on: [init-db]
---
Implement OAuth2 authentication flow
```
Express task dependencies via YAML frontmatter in feedback files. Topological sort ensures correct execution order. Cycle detection prevents deadlocks.

### Context Isolation
```yaml
orchestrator:
  context_isolation: true
  max_context_pct: 80.0
```
Tracks token usage per invocation. Warns when context window usage exceeds threshold to prevent quality degradation from context overload.

---

## Competitive Comparison

| Capability | Auto Claude Code | Ralph Wiggum | Tomacco |
|---|---|---|---|
| **Setup** | Zero (auto-discovers) | Manual spec.md | Manual PRD |
| **Task Source** | Automatic discovery | Manual | Manual |
| **Quality Gates** | LLM judges + tests + lint | None | LLM judges + gates |
| **Cost Control** | Per-hour budgets, prediction, circuit breaker | None | None |
| **Context Management** | Smart Zone auto-split | Fresh sessions (manual) | Fresh per task |
| **Merge Conflicts** | AI-resolved | Manual | Manual |
| **Learning** | Adaptive priority, config tuning | None | None |
| **Providers** | Claude, OpenAI, Gemini | Claude only | Claude, Gemini, Codex, DevMate |
| **Dependencies** | Python + pyyaml | Bash | Node.js 22+ |
| **Dashboard** | Web UI | None | Terminal TUI |
| **Session Recovery** | Automatic resume | None | Built-in |
| **Task Dependencies** | DAG with topological sort | None | JSON PRD |
| **PRD Support** | Generate + import + auto-detect | None | Manual authoring required |
| **Parallel Execution** | Git worktrees + AI merge | None | None |
| **Self-Improvement** | Modifies own code safely | None | None |

---

## Why Auto Claude Code Wins

**"Point it at a repo and go to sleep."** No PRDs to write, no spec files to maintain, no manual task lists to curate. Auto Claude Code discovers what needs doing by running your existing test suite, linter, and analyzing your codebase. Drop a feedback file if you want to steer it. Otherwise, it figures it out.

**Self-improving.** It learns from its own history — boosting priority for task types it succeeds at, tuning its own configuration, and adapting batch sizes based on recent performance.

**Safest autonomous system.** Disk space guards, memory checks, cost budgets with circuit breakers, rate limiting, protected files, two-layer watchdog, exponential backoff, graceful degradation. It won't burn through your API budget or fill your disk.

**Most portable.** Python + one dependency (pyyaml). Works on macOS and Linux. No Node.js, no Docker, no complex build systems. Uses stdlib `urllib` for API providers — no `requests`, no `httpx`.

**Production-grade quality gates.** Tests pass? Great. But also: did the LLM judge panel approve the security, quality, and architecture? Only then does it commit. Configurable fail actions let you tune strictness.

**Handles complexity.** Task dependencies with DAG ordering. Context window monitoring with auto-splitting. Session recovery across crashes. AI merge conflict resolution for parallel workers. These aren't demos — they're battle-tested features.

---

## Configuration Reference

All settings live in `config.yaml`:

| Section | Key Settings |
|---|---|
| `claude` | `model`, `max_turns`, `timeout_seconds`, `provider`, `api_key_env` |
| `orchestrator` | `loop_interval_seconds`, `plan_changes`, `batch_mode`, `adaptive_batch_*`, `max_validation_retries`, `context_isolation`, `smart_split`, `session_recovery` |
| `validation` | `test_command`, `lint_command`, `build_command`, `incremental_tests` |
| `discovery` | `enable_*` toggles, `todo_patterns`, `discovery_prompt`, `adaptive_priority` |
| `safety` | `max_consecutive_failures`, `max_cycles_per_hour`, `max_cost_usd_per_hour`, `protected_files` |
| `paths` | `feedback_dir`, `state_dir`, `history_file`, `backup_dir` |
| `logging` | `level`, `file`, `format` (text/json) |
| `agent_pipeline` | `enabled`, `max_revisions`, per-agent `model`/`max_turns`/`timeout` |
| `parallel` | `enabled`, `max_workers`, `merge_strategy`, `ai_conflict_resolution` |
| `notifications` | `enabled`, `webhooks`, `events`, `nl_summaries` |
| `judges` | `enabled`, per-judge `model`/`max_turns`, `fail_action` |
| `pricing` | `cost_per_million_input_tokens`, `output_cost_multiplier` |
| `github` | `enabled`, `create_prs`, `auto_merge`, `label` |

---

## Architecture Deep Dive

```
auto_claude_code/
├── main.py              # Entry point + two-layer watchdog (PROTECTED)
├── config.yaml          # Configuration (PROTECTED)
├── config_schema.py     # Load/validate config, apply defaults
├── orchestrator.py      # Main loop tying everything together
├── coordinator.py       # Parallel coordinator: worktree workers + merge
├── worker.py            # Parallel worker: isolated git worktree execution
├── agent_pipeline.py    # Multi-agent: Planner → Coder → Tester → Reviewer
├── task_discovery.py    # Auto-discover tasks (tests, lint, TODOs, ideas)
├── claude_runner.py     # Claude CLI invocation with circuit breaker
├── provider_runner.py   # Provider-agnostic: Claude, OpenAI, Gemini
├── model_resolver.py    # Resolve model aliases to IDs at startup
├── validator.py         # Run test/lint/build, determine pass/fail
├── git_manager.py       # Snapshot, rollback, commit, worktree management
├── feedback.py          # Feedback directory watcher + PRD auto-import
├── state.py             # Persist history to state/history.json
├── cycle_state.py       # Live cycle state for dashboard visibility
├── state_lock.py        # Thread-safe StateManager for parallel mode
├── safety.py            # Lock file, rate/cost/disk guards, circuit breaker
├── cost_predictor.py    # Estimate token cost before execution
├── llm_judges.py        # LLM-powered security/quality/architecture judges
├── context_monitor.py   # Smart zone: detect exhaustion, auto-split tasks
├── session_manager.py   # Crash recovery: save/restore session state
├── prd_generator.py     # PRD generation, import, export
├── prd_cli.py           # CLI for PRD operations
├── config_tuner.py      # Analyze history, recommend config changes
├── conflict_resolver.py # AI merge conflict resolution
├── github_integration.py # GitHub PR creation and management
├── notifications.py     # Webhook notifications (Slack, Discord, generic)
├── structured_logging.py # JSON log formatter
├── process_utils.py     # Process-group-aware subprocess management
├── telemetry.py         # Metrics computation from cycle history
├── dashboard.py         # Web dashboard for monitoring
└── tests/               # Comprehensive test suite (750+ tests)
```

| Module | Role |
|---|---|
| `orchestrator.py` | Core loop: gather tasks → invoke LLM → validate → commit/rollback |
| `coordinator.py` | Parallel mode: distribute tasks to worktree workers, merge results |
| `worker.py` | Execute tasks in isolated git worktrees with validation |
| `agent_pipeline.py` | Multi-agent: Planner → Coder → Tester → Reviewer with revisions |
| `provider_runner.py` | Factory for Claude CLI, OpenAI, and Gemini runners |
| `llm_judges.py` | Independent AI code review panel (security, quality, architecture) |
| `context_monitor.py` | Detect context exhaustion, auto-split tasks with dependencies |
| `session_manager.py` | Crash recovery with atomic session state persistence |
| `prd_generator.py` | Generate/import/export Product Requirement Documents |
| `task_discovery.py` | Discover tasks from tests, lint, TODOs, coverage, AI ideas |
| `claude_runner.py` | Claude CLI with retry, circuit breaker, token tracking |
| `git_manager.py` | Git operations with retry, worktree management |
| `safety.py` | Rate limits, cost limits, disk/memory checks, graceful degradation |

---

## Developer Tasks

Submit priority tasks by dropping files into `feedback/`:

```markdown
---
task_id: my-task
depends_on: [setup-db, init-config]
---
Implement the user authentication flow using OAuth2.
Add tests for all error cases.
```

- `.md` or `.txt` files are picked up automatically
- `.prd.yaml` and `.prd.json` files are auto-imported as task lists
- Prefix filenames with numbers for priority: `01-fix-bug.md` before `02-add-feature.md`
- YAML frontmatter supports `task_id` and `depends_on` for DAG ordering
- Completed tasks move to `feedback/done/`, failed tasks to `feedback/failed/`

---

## Safety Features

- **Lock file** — prevents concurrent runs
- **Protected files** — `main.py` and `config.yaml` cannot be modified
- **Rate limiting** — configurable max cycles per hour
- **Cost limiting** — per-hour budget with circuit breaker
- **Disk space guard** — aborts below threshold
- **Memory guard** — checks available memory before each cycle
- **Consecutive failure circuit breaker** — stops after N failures
- **API circuit breaker** — exponential backoff with jitter on rate limits
- **Graceful degradation** — reduces batch size and increases sleep on pressure
- **Two-layer watchdog** — `main.py` catches import failures
- **Syntax checking** — validates modified Python files before commit
- **File count limits** — prevents runaway changes
- **Cost prediction** — estimates cost before execution

---

## Testing

```bash
python3 -m pytest tests/ -v
```

750+ tests covering all modules with unit and integration tests.

---

## License

MIT
