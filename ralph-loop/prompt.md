# Ralph Loop Prompt: Auto Claude Code Improvement

You are working on the **Auto Claude Code** project at the current directory.

## Your Job

1. Read `ralph-loop/spec.md`
2. Read `ralph-loop/context.md` for codebase context
3. Find the **FIRST** task with status `PENDING`
4. Complete **ONLY THAT ONE TASK** -- not two, not three, exactly ONE
5. Update `ralph-loop/spec.md`: change that task's status from `PENDING` to `DONE` and move it to the Completed section
6. Exit immediately. Do NOT continue to the next task.

## Constraints

- **Protected files**: NEVER modify `main.py` or `config.yaml`
- **Dependencies**: Only use Python stdlib + pyyaml. Do NOT add new pip dependencies.
- **Testing**: After making changes, run `python3 -m pytest tests/ -x -q` to verify all tests pass
- **Linting**: Run `python3 -m ruff check .` if ruff is available, otherwise skip
- **Git**: Do NOT run any git commands (add, commit, push). The orchestration script handles git.
- **Scope**: Make minimal changes. Don't refactor unrelated code. Don't add docstrings to code you didn't change.
- **Existing tests**: All existing tests must continue to pass. If you change behavior, update the corresponding tests.
- **New tests**: When adding new test files, follow the patterns in existing test files (see `tests/conftest.py` for fixtures)

## Completion Criteria

A task is DONE when:
1. The described change is implemented
2. `python3 -m pytest tests/ -x -q` passes (all tests green)
3. The code is syntactically valid Python
4. No protected files were modified
5. The spec.md file has been updated (PENDING -> DONE, moved to Completed section)

## Special Instructions

- Read the relevant source file(s) BEFORE making changes to understand the existing patterns
- When adding new methods to existing classes, place them near related methods
- When creating new test files, import from the module under test and use pytest fixtures from conftest.py
- For tasks that involve `ast.parse`, handle `SyntaxError` gracefully (skip malformed files)
- When modifying prompt text in `shared.py`, keep the existing structure and just add/modify sections
- For tasks that add new discovery methods, register them in `discover_all()` behind a config flag
- Use `logging.getLogger(__name__)` for any new logging
- Follow the existing error handling pattern: catch specific exceptions, log warnings, return graceful defaults

## CRITICAL RULES

- **ONE TASK PER RUN.** Process exactly one PENDING task, then stop.
- **DO NOT continue** to process additional tasks after completing one.
- **DO NOT skip tasks.** Always process the FIRST PENDING task in order.
- After updating the spec, EXIT. Your job is done.
