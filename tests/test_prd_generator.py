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
            export_prd(prd, output, fmt="yaml")
            content = Path(output).read_text()
            loaded = yaml.safe_load(content)
            self.assertEqual(loaded["version"], "1.0")

    def test_export_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prd = {"version": "1.0", "tasks": [{"id": "t1", "description": "Fix"}]}
            output = str(Path(tmpdir) / "prd.json")
            export_prd(prd, output, fmt="json")
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


class TestImportPrdInvalidPriority(unittest.TestCase):
    """Tests for robust priority parsing in PRD import."""

    def test_non_integer_priority_defaults_to_3(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = str(Path(tmpdir) / "prd.yaml")
            data = {"tasks": [{"description": "task1", "priority": "high"}]}
            Path(prd_path).write_text(yaml.dump(data))
            tasks = import_prd(prd_path)
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].priority, 3)

    def test_null_priority_defaults_to_3(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = str(Path(tmpdir) / "prd.yaml")
            data = {"tasks": [{"description": "task1", "priority": None}]}
            Path(prd_path).write_text(yaml.dump(data))
            tasks = import_prd(prd_path)
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].priority, 3)

    def test_float_priority_converts_to_int(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = str(Path(tmpdir) / "prd.yaml")
            data = {"tasks": [{"description": "task1", "priority": 2.7}]}
            Path(prd_path).write_text(yaml.dump(data))
            tasks = import_prd(prd_path)
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].priority, 2)

    def test_missing_priority_defaults_to_3(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = str(Path(tmpdir) / "prd.yaml")
            data = {"tasks": [{"description": "task1"}]}
            Path(prd_path).write_text(yaml.dump(data))
            tasks = import_prd(prd_path)
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].priority, 3)


class TestPrdCliAtomicWrite(unittest.TestCase):
    """Tests that prd_cli.cmd_import writes files atomically."""

    def test_import_writes_files_atomically(self):
        """cmd_import should use temp+replace instead of write_text."""
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = str(Path(tmpdir) / "prd.yaml")
            feedback_dir = str(Path(tmpdir) / "feedback")
            data = {
                "tasks": [
                    {"id": "t1", "description": "Fix bug", "priority": 2},
                ],
            }
            Path(prd_path).write_text(yaml.dump(data))

            import argparse
            from prd_cli import cmd_import
            args = argparse.Namespace(prd_file=prd_path, feedback_dir=feedback_dir)
            cmd_import(args)

            # Verify file was created with correct content
            feedback_path = Path(feedback_dir)
            files = list(feedback_path.glob("prd-*.md"))
            self.assertEqual(len(files), 1)
            content = files[0].read_text()
            self.assertIn("Fix bug", content)
            self.assertIn("task_id: t1", content)

    def test_import_no_leftover_temp_files(self):
        """Atomic write should not leave .tmp files behind."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = str(Path(tmpdir) / "prd.yaml")
            feedback_dir = str(Path(tmpdir) / "feedback")
            data = {
                "tasks": [
                    {"id": "t1", "description": "Task 1", "priority": 1},
                    {"id": "t2", "description": "Task 2", "priority": 2},
                ],
            }
            Path(prd_path).write_text(yaml.dump(data))

            import argparse
            from prd_cli import cmd_import
            args = argparse.Namespace(prd_file=prd_path, feedback_dir=feedback_dir)
            cmd_import(args)

            # No .tmp files should remain
            feedback_path = Path(feedback_dir)
            tmp_files = list(feedback_path.glob("*.tmp"))
            self.assertEqual(len(tmp_files), 0)
            # Both task files should exist
            prd_files = list(feedback_path.glob("prd-*.md"))
            self.assertEqual(len(prd_files), 2)


class TestImportPrdSizeLimit(unittest.TestCase):

    def test_rejects_oversized_prd_file(self):
        """PRD files larger than 1 MB should be rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "huge.prd.yaml"
            # Write a file slightly over 1 MB
            prd_path.write_text("x" * (1024 * 1024 + 1))
            result = import_prd(str(prd_path))
            self.assertEqual(result, [])

    def test_accepts_normal_prd_file(self):
        """PRD files under 1 MB should be accepted normally."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "normal.prd.yaml"
            prd_data = {
                "version": "1.0",
                "tasks": [
                    {"id": "t1", "description": "Fix bug", "priority": 1, "source": "test_failure"}
                ],
            }
            import yaml
            prd_path.write_text(yaml.dump(prd_data))
            result = import_prd(str(prd_path))
            self.assertEqual(len(result), 1)


class TestPrdNonAsciiEncoding(unittest.TestCase):
    """Tests that import_prd and export_prd handle non-ASCII content with utf-8 encoding."""

    def test_import_prd_non_ascii_content(self):
        """import_prd should correctly read files containing non-ASCII characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = str(Path(tmpdir) / "prd.yaml")
            data = {
                "tasks": [
                    {"id": "t1", "description": "Corriger le bogue café ☕", "priority": 1},
                    {"id": "t2", "description": "日本語のタスク", "priority": 2},
                ],
            }
            Path(prd_path).write_text(yaml.dump(data, allow_unicode=True), encoding="utf-8")
            tasks = import_prd(prd_path)
            self.assertEqual(len(tasks), 2)
            self.assertIn("café", tasks[0].description)
            self.assertIn("日本語", tasks[1].description)

    def test_export_prd_non_ascii_roundtrip(self):
        """export_prd should write non-ASCII content that import_prd can read back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_tasks = [
                Task(description="Résoudre le problème über", priority=1,
                     source="feedback", task_id="t1"),
                Task(description="修复错误 🐛", priority=2,
                     source="lint", task_id="t2"),
            ]
            prd = generate_prd(original_tasks)
            prd_path = str(Path(tmpdir) / "prd.yaml")
            export_prd(prd, prd_path)
            imported = import_prd(prd_path)
            self.assertEqual(len(imported), 2)
            self.assertIn("über", imported[0].description)
            self.assertIn("修复错误", imported[1].description)

    def test_export_json_non_ascii(self):
        """export_prd in JSON format should handle non-ASCII content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd = {
                "version": "1.0",
                "tasks": [{"id": "t1", "description": "Ñoño señor 🎉"}],
            }
            output = str(Path(tmpdir) / "prd.json")
            export_prd(prd, output, fmt="json")
            content = Path(output).read_text(encoding="utf-8")
            loaded = json.loads(content)
            self.assertIn("Ñoño", loaded["tasks"][0]["description"])


class TestExportPrdAtomicWrite(unittest.TestCase):
    """export_prd must use atomic write (tempfile + os.replace) to prevent corruption."""

    def test_export_uses_atomic_write(self):
        """export_prd should use os.replace for atomic writes, not plain write_text."""
        import inspect
        source = inspect.getsource(export_prd)
        assert "os.replace" in source or "replace(" in source, (
            "export_prd should use atomic write (tempfile + os.replace) "
            "to prevent corruption on crash"
        )


class TestImportPrdSanitization(unittest.TestCase):
    """PRD task descriptions must be sanitized like feedback files."""

    def test_dangerous_pattern_stripped_from_prd_task(self):
        """Command substitution in PRD description must be removed."""
        prd_data = {
            "tasks": [
                {"description": "Fix bug $(rm -rf /)", "priority": 1},
            ]
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".prd.yaml", delete=False
        ) as f:
            yaml.dump(prd_data, f)
            f.flush()
            tasks = import_prd(f.name)

        self.assertEqual(len(tasks), 1)
        self.assertNotIn("$(", tasks[0].description)
        self.assertNotIn("rm -rf", tasks[0].description)

    def test_empty_after_sanitization_skipped(self):
        """A PRD task that becomes empty after sanitization should be skipped."""
        prd_data = {
            "tasks": [
                {"description": "$(malicious_command)", "priority": 1},
            ]
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".prd.yaml", delete=False
        ) as f:
            yaml.dump(prd_data, f)
            f.flush()
            tasks = import_prd(f.name)

        # Task should be skipped since description is empty after sanitization
        self.assertEqual(len(tasks), 0)


if __name__ == "__main__":
    unittest.main()
