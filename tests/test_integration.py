"""End-to-end integration test.

Creates a temp git repo with a buggy Python file, runs one orchestration cycle
with a mocked Claude response, and verifies the fix + commit.
"""

import json
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from claude_runner import ClaudeResult
from config_schema import Config
from orchestrator import Orchestrator


@pytest.fixture
def integration_repo(tmp_path):
    """Create a temporary git repo with a buggy Python file and tests."""
    repo = tmp_path / "project"
    repo.mkdir()

    # Init git repo
    subprocess.run(["git", "init"], cwd=str(repo), capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(repo), capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(repo), capture_output=True, check=True,
    )

    # Create a buggy file
    (repo / "app.py").write_text(
        'def add(a, b):\n'
        '    return a - b  # BUG: should be a + b\n'
    )

    # Create a test file
    tests_dir = repo / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_app.py").write_text(
        'import sys\n'
        'sys.path.insert(0, "..")\n'
        'from app import add\n'
        '\n'
        'def test_add():\n'
        '    assert add(2, 3) == 5\n'
    )

    # Initial commit
    subprocess.run(["git", "add", "-A"], cwd=str(repo), capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit with bug"],
        cwd=str(repo), capture_output=True, check=True,
    )

    return repo


def test_full_cycle_fix_and_commit(integration_repo):
    """Simulate a full cycle: discover test failure, Claude fixes it, validate, commit."""
    repo = str(integration_repo)

    # Configure for this repo
    config = Config()
    config.target_dir = repo
    config.validation.test_command = f"python3 -m pytest {repo}/tests/ -x -q"
    config.validation.lint_command = ""
    config.validation.build_command = ""
    config.paths.history_file = str(integration_repo / "state" / "history.json")
    config.paths.lock_file = str(integration_repo / "state" / "lock.pid")
    config.paths.feedback_dir = str(integration_repo / "feedback")
    config.paths.feedback_done_dir = str(integration_repo / "feedback" / "done")
    config.paths.backup_dir = str(integration_repo / "state" / "backups")

    # Create the orchestrator
    orch = Orchestrator(config)

    # Mock Claude to "fix" the bug by writing the correct file
    def mock_claude_run(prompt, working_dir=None):
        # Simulate Claude fixing the bug
        app_path = Path(repo) / "app.py"
        app_path.write_text(
            'def add(a, b):\n'
            '    return a + b\n'
        )
        return ClaudeResult(
            success=True,
            result_text="Fixed the add function",
            cost_usd=0.03,
            duration_seconds=5.0,
        )

    orch.claude.run = mock_claude_run

    # Run a single cycle
    orch._cycle()

    # Verify the fix was committed
    result = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=repo, capture_output=True, text=True, check=True,
    )
    assert "[auto]" in result.stdout

    # Verify the file is fixed
    app_content = (integration_repo / "app.py").read_text()
    assert "a + b" in app_content
    assert "a - b" not in app_content

    # Verify history was recorded
    history = json.loads((integration_repo / "state" / "history.json").read_text())
    assert len(history) == 1
    assert history[0]["success"] is True


def test_full_cycle_rollback_on_bad_fix(integration_repo):
    """Simulate a cycle where Claude's fix doesn't pass validation."""
    repo = str(integration_repo)

    config = Config()
    config.target_dir = repo
    config.validation.test_command = f"python3 -m pytest {repo}/tests/ -x -q"
    config.validation.lint_command = ""
    config.validation.build_command = ""
    config.paths.history_file = str(integration_repo / "state" / "history.json")
    config.paths.lock_file = str(integration_repo / "state" / "lock.pid")
    config.paths.feedback_dir = str(integration_repo / "feedback")
    config.paths.feedback_done_dir = str(integration_repo / "feedback" / "done")
    config.paths.backup_dir = str(integration_repo / "state" / "backups")

    orch = Orchestrator(config)

    # Mock Claude to make a bad fix
    def mock_claude_run(prompt, working_dir=None):
        app_path = Path(repo) / "app.py"
        app_path.write_text(
            'def add(a, b):\n'
            '    return a * b  # Still wrong!\n'
        )
        return ClaudeResult(
            success=True,
            result_text="Tried to fix",
            cost_usd=0.02,
            duration_seconds=3.0,
        )

    orch.claude.run = mock_claude_run

    # Run a single cycle
    orch._cycle()

    # Verify rollback happened: file should be back to original
    app_content = (integration_repo / "app.py").read_text()
    assert "a - b" in app_content  # Original buggy code

    # Verify no new commit was made
    result = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=repo, capture_output=True, text=True, check=True,
    )
    assert "[auto]" not in result.stdout

    # Verify history recorded the failure
    history = json.loads((integration_repo / "state" / "history.json").read_text())
    assert len(history) == 1
    assert history[0]["success"] is False
