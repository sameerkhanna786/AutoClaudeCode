#!/usr/bin/env bash
set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
SPEC_FILE="ralph-loop/spec.md"
PROMPT_FILE="ralph-loop/prompt.md"
LOG_FILE="ralph-loop/ralph.log"
MAX_FAILURES=5          # Stop after this many consecutive failures
PAUSE_SECONDS=2         # Pause between iterations

# ─── State ───────────────────────────────────────────────────────────────────
consecutive_failures=0
tasks_completed=0
tasks_failed=0

# ─── Stream Filter ───────────────────────────────────────────────────────────
# Parses --output-format stream-json events and displays human-readable status
stream_filter() {
    local log_file="$1"
    while IFS= read -r line; do
        # Log raw JSON for debugging
        echo "$line" >> "$log_file"

        # Extract and display human-readable events
        if echo "$line" | python3 -c "
import sys, json
try:
    event = json.loads(sys.stdin.readline())
    t = event.get('type', '')
    if t == 'assistant' and 'message' in event:
        msg = event['message']
        if 'content' in msg:
            for block in msg['content']:
                if block.get('type') == 'tool_use':
                    name = block.get('name', '')
                    inp = block.get('input', {})
                    if name == 'Read':
                        print(f'  Reading {inp.get(\"file_path\", \"?\")[:80]}')
                    elif name == 'Edit':
                        print(f'  Editing {inp.get(\"file_path\", \"?\")[:80]}')
                    elif name == 'Write':
                        print(f'  Writing {inp.get(\"file_path\", \"?\")[:80]}')
                    elif name == 'Bash':
                        cmd = inp.get('command', '')[:80]
                        print(f'  Running: {cmd}')
                    elif name == 'Grep':
                        print(f'  Searching: {inp.get(\"pattern\", \"?\")[:60]}')
                    elif name == 'Glob':
                        print(f'  Globbing: {inp.get(\"pattern\", \"?\")[:60]}')
                    else:
                        print(f'  [{name}]')
                elif block.get('type') == 'text':
                    text = block.get('text', '')[:120]
                    if text.strip():
                        print(f'  {text}')
    elif t == 'result':
        cost = event.get('cost_usd', 0)
        duration = event.get('duration_ms', 0) / 1000
        print(f'  Done (\${cost:.3f}, {duration:.0f}s)')
except (json.JSONDecodeError, KeyError, TypeError):
    pass
" 2>/dev/null; then
            :
        fi
    done
}

# ─── Main Loop ───────────────────────────────────────────────────────────────
echo "=== Ralph Loop Starting ==="
echo "Spec: $SPEC_FILE"
echo "Log:  $LOG_FILE"
echo ""

while true; do
    # Check if any PENDING tasks remain
    if ! grep -q "^[0-9]\+\. PENDING:" "$SPEC_FILE" 2>/dev/null; then
        echo ""
        echo "=== All tasks complete! ==="
        echo "Completed: $tasks_completed | Failed: $tasks_failed"
        exit 0
    fi

    # Check consecutive failure limit
    if [ "$consecutive_failures" -ge "$MAX_FAILURES" ]; then
        echo ""
        echo "=== Stopping: $MAX_FAILURES consecutive failures ==="
        echo "Completed: $tasks_completed | Failed: $tasks_failed"
        echo "Review $LOG_FILE for details, fix the issue, then re-run."
        exit 1
    fi

    # Get the next pending task for display
    next_task=$(grep -m1 "^[0-9]\+\. PENDING:" "$SPEC_FILE" | sed 's/^[0-9]\+\. PENDING: //')
    pending_count=$(grep -c "^[0-9]\+\. PENDING:" "$SPEC_FILE" || true)
    echo "──────────────────────────────────────────────────────────────"
    echo "[$pending_count remaining] Next: ${next_task:0:80}"
    echo "──────────────────────────────────────────────────────────────"

    # Run Claude Code on the prompt
    if claude --dangerously-skip-permissions \
        --output-format stream-json --verbose \
        -p "$(cat "$PROMPT_FILE")" 2>&1 | stream_filter "$LOG_FILE"; then

        # Check if the task was actually marked DONE in the spec
        if git diff --name-only | grep -q "$SPEC_FILE"; then
            echo "  Task completed successfully"
            tasks_completed=$((tasks_completed + 1))
            consecutive_failures=0

            # Commit the changes
            git add -A
            git commit -m "Ralph loop: complete task - ${next_task:0:60}" --no-verify
            git push
        else
            echo "  WARNING: Claude exited OK but spec was not updated"
            echo "  Marking as failed and continuing..."
            tasks_failed=$((tasks_failed + 1))
            consecutive_failures=$((consecutive_failures + 1))

            # Mark as FAILED in spec to skip it
            sed -i '' "s/^\\([0-9]\\+\\)\\. PENDING: $(echo "$next_task" | sed 's/[\/&]/\\&/g' | head -c 60)/\\1. FAILED:/" "$SPEC_FILE" 2>/dev/null || true
            git add -A
            git commit -m "Ralph loop: mark failed - ${next_task:0:60}" --no-verify || true
        fi
    else
        echo "  Claude exited with error"
        tasks_failed=$((tasks_failed + 1))
        consecutive_failures=$((consecutive_failures + 1))

        # Mark as FAILED in spec
        sed -i '' "s/^\\([0-9]\\+\\)\\. PENDING: $(echo "$next_task" | sed 's/[\/&]/\\&/g' | head -c 60)/\\1. FAILED:/" "$SPEC_FILE" 2>/dev/null || true

        # Revert any partial changes
        git checkout -- . 2>/dev/null || true
        git clean -fd 2>/dev/null || true

        git add "$SPEC_FILE"
        git commit -m "Ralph loop: mark failed - ${next_task:0:60}" --no-verify || true
    fi

    # Brief pause between iterations
    echo "  Pausing ${PAUSE_SECONDS}s..."
    sleep "$PAUSE_SECONDS"
done
