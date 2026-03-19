#!/usr/bin/env python3
"""CLI for PRD generation and import.

Usage:
    python3 prd_cli.py generate --config config.yaml --output prd.yaml
    python3 prd_cli.py import my-prd.yaml --feedback-dir feedback/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_generate(args):
    """Generate a PRD from the current project state."""
    from config_schema import load_config
    from task_discovery import TaskDiscovery
    from state import StateManager
    from prd_generator import generate_prd, export_prd

    config = load_config(args.config)
    state = StateManager(config)
    discovery = TaskDiscovery(config, state_manager=state)
    tasks = discovery.discover_all()

    prd = generate_prd(tasks, config=config, state_manager=state)

    fmt = "json" if args.output.endswith(".json") else "yaml"
    export_prd(prd, args.output, fmt=fmt)
    print(f"Generated PRD with {len(tasks)} tasks -> {args.output}")


def cmd_import(args):
    """Import a PRD file into the feedback directory."""
    from prd_generator import import_prd

    tasks = import_prd(args.prd_file)
    if not tasks:
        print("No tasks found in PRD file")
        return

    feedback_dir = Path(args.feedback_dir)
    feedback_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        deps_str = ", ".join(f'"{d}"' for d in task.depends_on)
        frontmatter = (
            f"---\n"
            f"task_id: {task.task_id}\n"
            f"depends_on: [{deps_str}]\n"
            f"---\n"
        )
        content = frontmatter + task.description

        import re as _re
        safe_id = _re.sub(r'[^a-zA-Z0-9_\-]', '_', task.task_id)
        filename = f"prd-{safe_id}.md"
        filepath = feedback_dir / filename
        filepath.write_text(content)

    print(f"Imported {len(tasks)} tasks -> {args.feedback_dir}/")


def main():
    parser = argparse.ArgumentParser(description="PRD generation and import")
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="Generate a PRD from the project")
    gen.add_argument("--config", default="config.yaml", help="Config file path")
    gen.add_argument("--output", default="prd.yaml", help="Output file path")

    imp = sub.add_parser("import", help="Import a PRD into feedback/")
    imp.add_argument("prd_file", help="Path to PRD file (YAML or JSON)")
    imp.add_argument("--feedback-dir", default="feedback", help="Feedback directory")

    args = parser.parse_args()
    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "import":
        cmd_import(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
