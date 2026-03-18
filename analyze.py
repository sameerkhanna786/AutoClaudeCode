#!/usr/bin/env python3
"""Standalone analysis script that discovers tasks and prints a formatted report.

Usage:
    python3 analyze.py [--config path/to/config.yaml]

Runs TaskDiscovery.discover_all() and prints all discovered tasks grouped by
source type, with counts and priorities.
"""

import argparse
import logging
import sys
from collections import defaultdict

from config_schema import load_config
from task_discovery import TaskDiscovery


def main():
    parser = argparse.ArgumentParser(
        description="Analyze the target project and report discovered tasks."
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to config.yaml (uses defaults if not provided)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show task context and details",
    )
    args = parser.parse_args()

    # Suppress noisy logs unless verbose
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    config = load_config(args.config)
    discovery = TaskDiscovery(config)
    tasks = discovery.discover_all()

    if not tasks:
        print("No tasks discovered.")
        return

    # Group by source
    by_source = defaultdict(list)
    for task in tasks:
        by_source[task.source].append(task)

    # Print report
    print(f"{'=' * 60}")
    print(f"  Task Discovery Report  —  {len(tasks)} task(s) found")
    print(f"{'=' * 60}")
    print()

    # Summary table
    print(f"  {'Source':<20} {'Count':>6}   Priority Range")
    print(f"  {'-' * 20} {'-' * 6}   {'-' * 14}")
    for source in sorted(by_source):
        group = by_source[source]
        priorities = [t.priority for t in group]
        lo, hi = min(priorities), max(priorities)
        prange = str(lo) if lo == hi else f"{lo}-{hi}"
        print(f"  {source:<20} {len(group):>6}   {prange}")
    print()

    # Detailed listing per source
    for source in sorted(by_source):
        group = by_source[source]
        group.sort(key=lambda t: t.priority)
        print(f"--- {source} ({len(group)}) ---")
        for i, task in enumerate(group, 1):
            loc = ""
            if task.source_file:
                loc = f"  [{task.source_file}"
                if task.line_number:
                    loc += f":{task.line_number}"
                loc += "]"
            print(f"  {i}. [P{task.priority}] {task.description}{loc}")
            if args.verbose and task.context:
                # Indent context lines
                for line in task.context.splitlines()[:10]:
                    print(f"        {line}")
                if len(task.context.splitlines()) > 10:
                    print(f"        ... ({len(task.context.splitlines())} lines total)")
                print()
        print()


if __name__ == "__main__":
    main()
