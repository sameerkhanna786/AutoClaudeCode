# Ralph Loop Spec: Auto Claude Code Comprehensive Improvement

## Tasks

1. DONE: In `shared.py`, the `build_retry_prompt()` function only passes `validation.summary` (a one-line string like "tests: FAIL") to the retry prompt. Change the worker retry flow in `worker.py:316` to pass the full `validation.steps` output (including test failure tracebacks) instead of just the summary, so Claude has enough information to actually fix the failure. Cap the output at 8000 chars to avoid prompt overflow.

2. DONE: In `task_discovery.py:_discover_test_failures()`, the `context` field for test failure tasks contains either the per-test traceback or the full pytest output. Improve `_extract_test_traceback()` to also include the assertion details and the last 5 lines of the traceback, and append the file path + line number of the failing assertion as `source_file` and `line_number` on the Task so that Claude can jump directly to the relevant code.

3. DONE: In `task_discovery.py:_discover_claude_ideas()`, the generated tasks have empty `context` fields. After extracting IDEA lines, use `_FILE_REF_RE` to find any file references in the idea description, then call `_read_file_snippet()` to populate the task's `context` with the relevant code snippet. This gives Claude concrete code to work with instead of just a vague description.

4. DONE: Add a new discovery method `_discover_complexity_issues()` in `task_discovery.py` that scans Python files for functions longer than 50 lines (using `ast.parse` to walk `FunctionDef` nodes) and creates `quality` tasks with priority 5. Include the function name, file path, line number, and line count in the task description. Cap at 5 tasks.

5. DONE: In `validator.py:validate_with_baseline()`, when test failures are all pre-existing, the method returns `passed=True` but the `output` in the step still contains all the failure text. Add a note to the output like "NOTE: All N failure(s) are pre-existing baseline failures" so that downstream consumers (retry prompts, logs) don't confuse baseline failures with new issues.

6. DONE: In `claude_runner.py`, the `_parse_json_response()` method has three nearly-identical JSON parsing strategies. Refactor into a single `_try_parse_json(text) -> Optional[dict]` helper that tries line-by-line parsing, then multi-line, then raw_decode, reducing the method from ~50 lines to ~20.

7. DONE: In `task_discovery.py:_discover_claude_ideas()`, lines 552-630 have three nearly-identical JSON extraction strategies (line-by-line, multi-line join, raw_decode). Extract this into a shared `_extract_json_text(raw_output) -> str` utility function in `task_discovery.py` that both this method and `claude_runner.py:_parse_json_response()` could reference, reducing duplication.

8. DONE: Add a `_discover_import_issues()` method to `task_discovery.py` that uses `ast.parse` to find unused imports in Python files. Create `lint` tasks with priority 3 for files with unused imports. Only scan files not in `exclude_dirs`.

9. DONE: In `worker.py:execute()`, when Claude modifies files in the main repo instead of the worktree (lines 229-244), the error message is generic. Enhance it to include what files were modified and suggest the likely cause (Claude used relative paths instead of absolute paths to the worktree). Also log the prompt's working directory preamble for debugging.

10. DONE: In `shared.py:build_task_prompt()`, add a section that lists the 3-5 most relevant files for the task by extracting file references from the task description and context using `_FILE_REF_RE`. Format as "RELEVANT FILES:\n- path/to/file.py\n" so Claude knows where to look first.

11. DONE: Add test coverage for `conflict_resolver.py` by creating `tests/test_conflict_resolver.py` with tests for: successful resolution, failed resolution, timeout handling, and empty conflict list. Mock the Claude CLI calls.

12. DONE: In `coordinator.py:_merge_worker_branch()`, after a failed rebase (line 365), the method doesn't attempt the merge strategy as a fallback. Add fallback logic: if rebase fails and strategy is "rebase", try a regular merge before giving up.

13. DONE: Add tests for the circuit breaker exponential backoff in `claude_runner.py`. Create test cases in `tests/test_claude_runner.py` that verify: (a) recovery timeout doubles after each re-open, (b) timeout is capped at max_recovery_timeout, (c) jitter is applied within the expected range, (d) successful call resets backoff.

14. DONE: In `safety.py:check_memory()`, the macOS implementation only counts "Pages free" + "Pages speculative" + "Pages purgeable". Add "Pages inactive" to the count since macOS treats inactive pages as available memory, which currently causes false low-memory warnings.

15. DONE: Add a `validate_task_feasibility()` method to `task_discovery.py:TaskDiscovery` that estimates task complexity by checking: (a) number of files referenced, (b) whether referenced files exist, (c) whether the task description is specific enough (>20 chars, contains a file reference). Return a feasibility score 0-1. Use this in `discover_all()` to filter out low-feasibility tasks (score < 0.3).

16. DONE: In `agent_pipeline.py`, the planner agent always runs even for simple tasks like lint fixes. Added a `skip_planning_for` config list (default: `["lint", "todo"]`) that skips the planner for simple task types and goes straight to the coder, saving one Claude invocation per simple task.

17. DONE: Add tests for `session_manager.py` in `tests/test_session_manager.py`. Test: session file creation/loading, orphaned worktree detection, cleanup of stale sessions, and recovery flow.

18. DONE: In `coordinator.py:_partition_tasks()`, tasks are assigned one-per-worker but there's no consideration of task independence. Added a check that avoids assigning two tasks that reference the same `source_file` to different workers (they'd create merge conflicts). Same-file tasks are now grouped together.

19. DONE: In `shared.py:build_retry_prompt()`, added a "COMMON FAILURE PATTERNS" section with tips based on the task type. Added `_COMMON_FAILURE_PATTERNS` dict and `_common_failure_patterns()` helper.

20. DONE: Add a `get_task_success_history()` method to `state.py:StateManager` that returns the last N attempts for a given task_key, including what error occurred each time. Use this in `shared.py:build_retry_prompt()` to include previous failure reasons so Claude doesn't repeat the same mistake.

21. DONE: In `validator.py`, add a `validate_syntax_only()` method that just runs `ast.parse()` on all changed `.py` files. This can be used as a fast pre-check before running the full test suite, catching obvious syntax errors in <1 second instead of waiting for pytest to fail.

22. DONE: In `orchestrator.py`, the `_run_cycle()` method is ~200 lines. Extract the validation+retry loop into a separate `_validate_with_retries()` method to improve readability and make it easier to test the retry logic in isolation.

23. DONE: Add tests for `GracefulDegradation` in `tests/test_safety.py` that verify all four degradation levels (normal, mild, moderate, severe) return correct `batch_size_factor` and `sleep_multiplier` values at the exact threshold boundaries (69%, 70%, 85%, 95%).

24. DONE: In `task_discovery.py:_discover_quality_issues()`, the only quality check is file length >500 lines. Add checks for: (a) functions with more than 5 parameters (using `ast.parse`), (b) deeply nested code (indentation level >4), (c) files with no docstring on the module level. Create separate tasks for each finding.

25. DONE: In `worker.py`, the `_build_retry_prompt()` passes `validation.summary` which is just "tests: FAIL, lint: PASS". Change it to pass the full output from the failed validation step (the actual pytest traceback or lint errors) so Claude can see what specifically went wrong.

26. DONE: Add an `--analyze` CLI flag to `main.py`... wait, main.py is protected. Instead, create a standalone `analyze.py` script that runs `TaskDiscovery.discover_all()` and prints a formatted report of all discovered tasks grouped by source type, with counts and priorities. This helps users understand what the system would work on.

27. DONE: In `claude_runner.py:ClaudeRunner.run()`, the rate limit detection only checked for "rate limit", "429", and "too many requests" in stderr. Added "quota exceeded" and "capacity" to `_CB_ERROR_PATTERNS` ("overloaded" was already present). Updated the retry-loop rate limit detection to also trigger rate-limit-style backoff for "quota exceeded", "capacity", and "overloaded" patterns.

28. DONE: Add a `_discover_test_coverage_gaps()` method to `task_discovery.py` that doesn't require pytest-cov. Instead, scan for Python source files that have no corresponding `test_*.py` file in the tests directory. Create `coverage` tasks with priority 4 for untested modules. This is cheaper than running pytest-cov.

29. DONE: In `shared.py:build_plan_prompt()`, the planning prompt asks Claude to "Output your complete plan within 5 turns" but doesn't specify a format. Add a structured output format requirement: "Output your plan as a numbered list where each item specifies: FILE, CHANGE_TYPE (add/modify/delete), and DESCRIPTION of the change."

30. DONE: In `coordinator.py`, add a `_log_cycle_summary()` method called at the end of `_run_cycle()` that logs: number of tasks dispatched, number succeeded, number failed, total cost, total duration, and which task types succeeded/failed. This provides quick visibility into each parallel cycle's effectiveness.

31. DONE: Add tests for `worker._cost_limit_exceeded()` in `tests/test_worker.py` that verify: (a) returns False when well under budget, (b) returns True at 90% threshold, (c) handles state.get_total_cost() exceptions gracefully, (d) logs appropriate warning message.

32. DONE: In `agent_pipeline.py:AgentPipeline.run()`, the planner prompt (line 416) is generic. Enhance it to include task-type-specific planning instructions from `TASK_TYPE_INSTRUCTIONS` in `shared.py`, so the planner knows the conventions for the specific task type.

33. DONE: Add a `_discover_dead_code()` method to `task_discovery.py` that uses `ast.parse` to find functions/methods that are defined but never called within the same module (simple intra-module dead code detection). Create `quality` tasks with priority 5. Only flag functions not starting with `_` (public API) or starting with `test_`.

34. PENDING: In `safety.py:pre_flight_checks()`, the checks run sequentially. The `check_memory()` and `check_disk_space()` checks are independent and could provide better error messages if both fail simultaneously. Collect all check failures and raise a single SafetyError with all issues listed.

35. PENDING: In `state.py`, add a `get_strategy_performance_report()` method that returns a formatted string showing success rate, average cost, and average duration per task type over the last 24 hours. Call this from the orchestrator/coordinator at startup to log which strategies are working best.

## Completed

13. DONE: Add tests for the circuit breaker exponential backoff in `claude_runner.py`. Create test cases in `tests/test_claude_runner.py` that verify: (a) recovery timeout doubles after each re-open, (b) timeout is capped at max_recovery_timeout, (c) jitter is applied within the expected range, (d) successful call resets backoff.

1. DONE: In `shared.py`, the `build_retry_prompt()` function only passes `validation.summary` (a one-line string like "tests: FAIL") to the retry prompt. Change the worker retry flow in `worker.py:316` to pass the full `validation.steps` output (including test failure tracebacks) instead of just the summary, so Claude has enough information to actually fix the failure. Cap the output at 8000 chars to avoid prompt overflow.

2. DONE: In `task_discovery.py:_discover_test_failures()`, the `context` field for test failure tasks contains either the per-test traceback or the full pytest output. Improve `_extract_test_traceback()` to also include the assertion details and the last 5 lines of the traceback, and append the file path + line number of the failing assertion as `source_file` and `line_number` on the Task so that Claude can jump directly to the relevant code.

3. DONE: In `task_discovery.py:_discover_claude_ideas()`, the generated tasks have empty `context` fields. After extracting IDEA lines, use `_FILE_REF_RE` to find any file references in the idea description, then call `_read_file_snippet()` to populate the task's `context` with the relevant code snippet. This gives Claude concrete code to work with instead of just a vague description.

4. DONE: Add a new discovery method `_discover_complexity_issues()` in `task_discovery.py` that scans Python files for functions longer than 50 lines (using `ast.parse` to walk `FunctionDef` nodes) and creates `quality` tasks with priority 5. Include the function name, file path, line number, and line count in the task description. Cap at 5 tasks.

5. DONE: In `validator.py:validate_with_baseline()`, when test failures are all pre-existing, the method returns `passed=True` but the `output` in the step still contains all the failure text. Add a note to the output like "NOTE: All N failure(s) are pre-existing baseline failures" so that downstream consumers (retry prompts, logs) don't confuse baseline failures with new issues.

6. DONE: In `claude_runner.py`, the `_parse_json_response()` method has three nearly-identical JSON parsing strategies. Refactor into a single `_try_parse_json(text) -> Optional[dict]` helper that tries line-by-line parsing, then multi-line, then raw_decode, reducing the method from ~50 lines to ~20.

7. DONE: In `task_discovery.py:_discover_claude_ideas()`, lines 552-630 have three nearly-identical JSON extraction strategies (line-by-line, multi-line join, raw_decode). Extract this into a shared `_extract_json_text(raw_output) -> str` utility function in `task_discovery.py` that both this method and `claude_runner.py:_parse_json_response()` could reference, reducing duplication.

8. DONE: Add a `_discover_import_issues()` method to `task_discovery.py` that uses `ast.parse` to find unused imports in Python files. Create `lint` tasks with priority 3 for files with unused imports. Only scan files not in `exclude_dirs`.

9. DONE: In `worker.py:execute()`, when Claude modifies files in the main repo instead of the worktree (lines 229-244), the error message is generic. Enhance it to include what files were modified and suggest the likely cause (Claude used relative paths instead of absolute paths to the worktree). Also log the prompt's working directory preamble for debugging.

10. DONE: In `shared.py:build_task_prompt()`, add a section that lists the 3-5 most relevant files for the task by extracting file references from the task description and context using `_FILE_REF_RE`. Format as "RELEVANT FILES:\n- path/to/file.py\n" so Claude knows where to look first.

11. DONE: Add test coverage for `conflict_resolver.py` by creating `tests/test_conflict_resolver.py` with tests for: successful resolution, failed resolution, timeout handling, and empty conflict list. Mock the Claude CLI calls.

12. DONE: In `coordinator.py:_merge_worker_branch()`, after a failed rebase (line 365), the method doesn't attempt the merge strategy as a fallback. Add fallback logic: if rebase fails and strategy is "rebase", try a regular merge before giving up.

14. DONE: In `safety.py:check_memory()`, the macOS implementation only counts "Pages free" + "Pages speculative" + "Pages purgeable". Add "Pages inactive" to the count since macOS treats inactive pages as available memory, which currently causes false low-memory warnings.

15. DONE: Add a `validate_task_feasibility()` method to `task_discovery.py:TaskDiscovery` that estimates task complexity by checking: (a) number of files referenced, (b) whether referenced files exist, (c) whether the task description is specific enough (>20 chars, contains a file reference). Return a feasibility score 0-1. Use this in `discover_all()` to filter out low-feasibility tasks (score < 0.3).

16. DONE: In `agent_pipeline.py`, the planner agent always runs even for simple tasks like lint fixes. Added a `skip_planning_for` config list (default: `["lint", "todo"]`) that skips the planner for simple task types and goes straight to the coder, saving one Claude invocation per simple task.

17. DONE: Add tests for `session_manager.py` in `tests/test_session_manager.py`. Test: session file creation/loading, orphaned worktree detection, cleanup of stale sessions, and recovery flow.

18. DONE: In `coordinator.py:_partition_tasks()`, tasks are assigned one-per-worker but there's no consideration of task independence. Added a check that avoids assigning two tasks that reference the same `source_file` to different workers (they'd create merge conflicts). Same-file tasks are now grouped together.

19. DONE: In `shared.py:build_retry_prompt()`, added a "COMMON FAILURE PATTERNS" section with a `_COMMON_FAILURE_PATTERNS` dict mapping task sources (test_failure, lint, todo, quality, coverage, claude_idea) to common failure causes, and a `_common_failure_patterns()` helper that appends relevant tips to the retry prompt based on task type.

20. DONE: Added `get_task_success_history()` method to `state.py:StateManager` that returns the last N attempts for a given task_key/description, including error and validation_summary. Added `_format_task_history()` helper and optional `task_history` parameter to `shared.py:build_retry_prompt()`. Updated `orchestrator.py` and `worker.py` to pass task history to retry prompts. Added tests in both `test_state.py` and `test_shared.py`.

21. DONE: Added `validate_syntax_only()` method to `validator.py` that runs `ast.parse()` on all changed `.py` files. Handles missing files, non-Python files, and `SyntaxError`/`UnicodeDecodeError` gracefully. Returns a `ValidationResult` with a "syntax" step. Added 7 tests in `tests/test_validator.py`.

22. DONE: In `orchestrator.py`, the validation+retry loop was already extracted into `_validate_with_retries()` (lines 244-496). The `_cycle()` method delegates to it at line 866 for single-agent mode and line 940 for multi-agent mode. No code changes needed — the extraction was already in place.

23. DONE: Added 12 tests for `GracefulDegradation` in `tests/test_safety.py` covering all four degradation levels (normal at <70%, mild at 70%, moderate at 85%, severe at 95%) with exact boundary checks for `batch_size_factor` and `sleep_multiplier`. Also tests cost-driven degradation, combined rate+cost reasons, and recovery from degraded to normal state.

24. DONE: In `task_discovery.py:_discover_quality_issues()`, added three new quality checks beyond file length: (a) functions with more than 5 parameters using `ast.parse` (excluding self/cls), (b) deeply nested code (indentation level >4 based on 4-space indent), (c) files with no module-level docstring. Each creates a separate `quality` task with priority 5. Capped at 5 total tasks.

25. DONE: In `worker.py`, the retry flow already passes full validation output (not just `validation.summary`). The `_format_validation_errors()` method (lines 507-520) extracts the full output from each failed validation step including command, exit code, and actual pytest traceback/lint errors, capped at 8000 chars per step. This was already implemented as part of task 1's changes to the worker retry flow.

26. DONE: Created standalone `analyze.py` script that runs `TaskDiscovery.discover_all()` and prints a formatted report of all discovered tasks grouped by source type, with counts and priorities. Supports `--config` for custom config and `--verbose` for detailed context output.

27. DONE: In `claude_runner.py:ClaudeRunner.run()`, the rate limit detection only checked for "rate limit", "429", and "too many requests" in stderr. Added "quota exceeded" and "capacity" to `_CB_ERROR_PATTERNS` ("overloaded" was already present). Updated the retry-loop rate limit detection to also trigger rate-limit-style backoff for "quota exceeded", "capacity", and "overloaded" patterns.

28. DONE: Added `_discover_test_coverage_gaps()` method to `task_discovery.py` that scans for Python source files with no corresponding `test_*.py` file in the tests directory. Creates `coverage` tasks with priority 4 for untested modules. Skips `__init__.py`, `conftest.py`, `setup.py`, test files, and files in `exclude_dirs`. Capped at 10 tasks. Added `enable_test_coverage_gaps` config flag to `config_schema.py` and registered in `discover_all()`.

29. DONE: In `shared.py:build_plan_prompt()`, added a structured OUTPUT FORMAT section to both single-task and multi-task branches. The format requires a numbered list where each item specifies FILE, CHANGE_TYPE (add/modify/delete), and DESCRIPTION of the change, giving Claude a concrete structure to follow when outputting plans.

30. DONE: In `coordinator.py`, added a `_log_cycle_summary()` method called at the end of `_run_cycle()` that logs: number of tasks dispatched, number succeeded, number failed, total cost, total duration, and which task types succeeded/failed. Added `cycle_start = time.time()` at the top of `_run_cycle()` to track duration. The summary is logged via `logger.info` with all requested metrics.

31. DONE: Added 5 tests for `worker._cost_limit_exceeded()` in `tests/test_worker.py` in a new `TestCostLimitExceeded` class: (a) returns False when well under budget, (b) returns True at exact 90% threshold, (c) returns True when over threshold, (d) returns False and handles exceptions gracefully when `state.get_total_cost()` raises, (e) logs appropriate warning message when cost guard is triggered.

32. DONE: In `agent_pipeline.py:AgentPipeline.run()`, enhanced the planner prompt to include task-type-specific planning instructions from `TASK_TYPE_INSTRUCTIONS` in `shared.py`. Added import of `TASK_TYPE_INSTRUCTIONS` and logic to gather guidelines for each task source type, appending them as a "TASK-TYPE-SPECIFIC GUIDELINES" section in the planner prompt.

33. DONE: Added `_discover_dead_code()` method to `task_discovery.py` that uses `ast.parse` to find functions/methods defined but never called within the same module (intra-module dead code detection). Collects all defined public functions (not starting with `_` or `test_`), then scans for all name references (calls, attributes, name nodes) and flags unreferenced functions as `quality` tasks with priority 5. Capped at 5 tasks. Added `enable_dead_code_check` config flag to `config_schema.py` and registered in `discover_all()`.
