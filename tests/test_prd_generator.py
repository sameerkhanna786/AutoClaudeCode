"""Tests for prd_generator.py — PRD generation and import."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import yaml

from task_discovery import Task
from prd_generator import generate_prd, import_prd, export_prd


class TestGeneratePrd(unittest.TestCase):

    def test_basic_generation(self):
        tasks = [
            Task(description="Fix bug A", priority=2, source="test_failure",
                 task_id="t1"),
            Task(description="Add feature B", priority=3, source="feedback",
                 task_id="t2", depends_on=["t1"]),
        ]
        prd = generate_prd(tasks)
        self.assertEqual(prd["version"], "1.0")
        self.assertEqual(len(prd["tasks"]), 2)
        self.assertEqual(prd["tasks"][0]["id"], "t1")
        self.assertEqual(prd["tasks"][1]["depends_on"], ["t1"])

    def test_with_config(self):
        config = MagicMock()
        config.target_dir = "/project"
        config.claude.model = "opus"
        tasks = [Task(description="Fix X", priority=1, source="lint", task_id="x")]
        prd = generate_prd(tasks, config=config)
        self.assertEqual(prd["metadata"]["target_dir"], "/project")

    def test_with_state_manager(self):
        state = MagicMock()
        state.get_strategy_performance.return_value = {"lint": {"success_rate": 0.8}}
        tasks = [Task(description="Fix X", priority=1, source="lint", task_id="x")]
        prd = generate_prd(tasks, state_manager=state)
        self.assertIn("strategy_performance", prd["metadata"])

    def test_empty_tasks(self):
        prd = generate_prd([])
        self.assertEqual(len(prd["tasks"]), 0)


class TestExportPrd(unittest.TestCase):

    def test_export_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prd = {"version": "1.0", "tasks": [{"id": "t1", "description": "Fix"}]}
            output = str(Path(tmpdir) / "prd.yaml")
            export_prd(prd, output, format="yaml")
            content = Path(output).read_text()
            loaded = yaml.safe_load(content)
            self.assertEqual(loaded["version"], "1.0")

    def test_export_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prd = {"version": "1.0", "tasks": [{"id": "t1", "description": "Fix"}]}
            output = str(Path(tmpdir) / "prd.json")
            export_prd(prd, output, format="json")
            content = Path(output).read_text()
            loaded = json.loads(content)
            self.assertEqual(loaded["version"], "1.0")


class TestImportPrd(unittest.TestCase):

    def test_import_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prd = {
                "version": "1.0",
                "tasks": [
                    {"id": "t1", "description": "Fix bug", "priority": 2,
                     "source": "test_failure"},
                    {"id": "t2", "description": "Add feature", "priority": 3,
                     "depends_on": ["t1"]},
                ],
            }
            prd_path = str(Path(tmpdir) / "prd.yaml")
            Path(prd_path).write_text(yaml.dump(prd))
            tasks = import_prd(prd_path)
            self.assertEqual(len(tasks), 2)
            self.assertEqual(tasks[0].task_id, "t1")
            self.assertEqual(tasks[1].depends_on, ["t1"])

    def test_import_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prd = {
                "version": "1.0",
                "tasks": [
                    {"id": "t1", "description": "Fix bug", "priority": 2},
                ],
            }
            prd_path = str(Path(tmpdir) / "prd.json")
            Path(prd_path).write_text(json.dumps(prd))
            tasks = import_prd(prd_path)
            self.assertEqual(len(tasks), 1)

    def test_import_tomacco_format(self):
        """Import Tomacco-style PRD with phases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd = {
                "phases": [
                    {
                        "name": "Phase 1",
                        "tasks": [
                            {"title": "Setup auth", "priority": 1},
                            {"title": "Add login page", "priority": 2},
                        ],
                    },
                    {
                        "name": "Phase 2",
                        "tasks": [
                            {"title": "Deploy", "priority": 3},
                        ],
                    },
                ],
            }
            prd_path = str(Path(tmpdir) / "prd.yaml")
            Path(prd_path).write_text(yaml.dump(prd))
            tasks = import_prd(prd_path)
            self.assertEqual(len(tasks), 3)
            self.assertEqual(tasks[0].description, "Setup auth")

    def test_round_trip(self):
        """Generate -> export -> import should preserve tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_tasks = [
                Task(description="Fix bug", priority=2, source="test_failure",
                     task_id="t1"),
                Task(description="Add feature", priority=3, source="feedback",
                     task_id="t2", depends_on=["t1"]),
            ]
            prd = generate_prd(original_tasks)
            prd_path = str(Path(tmpdir) / "prd.yaml")
            export_prd(prd, prd_path)

            imported = import_prd(prd_path)
            self.assertEqual(len(imported), 2)
            self.assertEqual(imported[0].task_id, "t1")
            self.assertEqual(imported[1].depends_on, ["t1"])

    def test_import_nonexistent(self):
        tasks = import_prd("/nonexistent/prd.yaml")
        self.assertEqual(tasks, [])

    def test_import_invalid_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = str(Path(tmpdir) / "bad.yaml")
            Path(prd_path).write_text("{{{{invalid yaml")
            tasks = import_prd(prd_path)
            self.assertEqual(tasks, [])


if __name__ == "__main__":
    unittest.main()
