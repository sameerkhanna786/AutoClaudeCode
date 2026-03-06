"""Shared test fixtures."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure project root is on sys.path so all modules are importable
# regardless of the working directory when pytest is invoked.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ---------------------------------------------------------------------------
# Custom marker: requires_subprocess
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_subprocess: mark test as needing real subprocess execution "
        "(skipped in sandboxed environments)",
    )


def _is_sandbox_restricted():
    """Detect if subprocess spawning is restricted (e.g. macOS sandbox-exec)."""
    try:
        result = subprocess.run(
            ["echo", "test"], capture_output=True, timeout=5,
        )
        return result.returncode != 0
    except (OSError, subprocess.TimeoutExpired, PermissionError):
        return True


def pytest_collection_modifyitems(config, items):
    if _is_sandbox_restricted():
        skip_marker = pytest.mark.skip(
            reason="Subprocess execution restricted in this environment",
        )
        for item in items:
            if "requires_subprocess" in item.keywords:
                item.add_marker(skip_marker)

from config_schema import Config, load_config


@pytest.fixture
def default_config():
    """Return a Config with all defaults."""
    return Config()


@pytest.fixture
def tmp_dir(tmp_path):
    """Return a temporary directory path as a string."""
    return str(tmp_path)


@pytest.fixture
def tmp_git_repo(tmp_path):
    """Create a temporary git repo and return its path."""
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo), capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(repo), capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(repo), capture_output=True, check=True,
    )
    # Create an initial commit
    (repo / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "-A"], cwd=str(repo), capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=str(repo), capture_output=True, check=True,
    )
    return str(repo)
