"""Tests for cost_predictor.py — token estimation and budget enforcement."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from cost_predictor import (
    CHARS_PER_TOKEN,
    OUTPUT_COST_MULTIPLIER,
    OUTPUT_TO_INPUT_RATIO,
    _MODEL_COST_PER_M_INPUT_TOKENS,
    check_cost_budget,
    estimate_prompt_tokens,
    estimate_task_cost,
)
from task_discovery import Task


def _make_task(description: str = "test task", context: str = "") -> Task:
    """Helper to create a Task with known fields."""
    return Task(description=description, priority=3, source="claude_idea", context=context)


class TestEstimatePromptTokens(unittest.TestCase):

    def test_basic(self):
        result = estimate_prompt_tokens("hello")
        self.assertEqual(result, max(1, len("hello") // CHARS_PER_TOKEN))

    def test_empty_string(self):
        result = estimate_prompt_tokens("")
        self.assertEqual(result, 1)  # max(1, 0) = 1

    def test_long_string(self):
        text = "a" * 400
        result = estimate_prompt_tokens(text)
        self.assertEqual(result, 100)


class TestEstimateTaskCost(unittest.TestCase):

    def test_opus_pricing(self):
        task = _make_task("a" * 100, context="b" * 100)
        cost = estimate_task_cost([task], "opus", prompt_overhead=0)
        total_chars = 200
        input_tokens = total_chars // CHARS_PER_TOKEN
        output_tokens = int(input_tokens * OUTPUT_TO_INPUT_RATIO)
        input_cost_per_m = _MODEL_COST_PER_M_INPUT_TOKENS["opus"]
        output_cost_per_m = input_cost_per_m * OUTPUT_COST_MULTIPLIER
        expected = (input_tokens / 1_000_000) * input_cost_per_m + \
                   (output_tokens / 1_000_000) * output_cost_per_m
        self.assertAlmostEqual(cost, expected)

    def test_sonnet_pricing(self):
        task = _make_task("a" * 100)
        cost = estimate_task_cost([task], "sonnet", prompt_overhead=0)
        total_chars = len(task.description) + len(task.context)
        input_tokens = total_chars // CHARS_PER_TOKEN
        output_tokens = int(input_tokens * OUTPUT_TO_INPUT_RATIO)
        input_cost_per_m = _MODEL_COST_PER_M_INPUT_TOKENS["sonnet"]
        output_cost_per_m = input_cost_per_m * OUTPUT_COST_MULTIPLIER
        expected = (input_tokens / 1_000_000) * input_cost_per_m + \
                   (output_tokens / 1_000_000) * output_cost_per_m
        self.assertAlmostEqual(cost, expected)

    def test_haiku_pricing(self):
        task = _make_task("a" * 100)
        cost = estimate_task_cost([task], "haiku", prompt_overhead=0)
        input_cost_per_m = _MODEL_COST_PER_M_INPUT_TOKENS["haiku"]
        self.assertGreater(cost, 0)
        # haiku should be cheaper than opus
        opus_cost = estimate_task_cost([task], "opus", prompt_overhead=0)
        self.assertLess(cost, opus_cost)

    def test_unknown_model_defaults_to_opus(self):
        task = _make_task("hello world")
        cost_unknown = estimate_task_cost([task], "gpt-4", prompt_overhead=0)
        cost_opus = estimate_task_cost([task], "opus", prompt_overhead=0)
        self.assertAlmostEqual(cost_unknown, cost_opus)

    def test_model_id_prefix_match(self):
        task = _make_task("hello world")
        cost_prefix = estimate_task_cost([task], "claude-opus-4-6", prompt_overhead=0)
        cost_opus = estimate_task_cost([task], "opus", prompt_overhead=0)
        self.assertAlmostEqual(cost_prefix, cost_opus)

    def test_empty_tasks(self):
        cost = estimate_task_cost([], "opus", prompt_overhead=500)
        # Cost comes from prompt_overhead only
        input_tokens = 500
        output_tokens = int(input_tokens * OUTPUT_TO_INPUT_RATIO)
        input_cost_per_m = _MODEL_COST_PER_M_INPUT_TOKENS["opus"]
        output_cost_per_m = input_cost_per_m * OUTPUT_COST_MULTIPLIER
        expected = (input_tokens / 1_000_000) * input_cost_per_m + \
                   (output_tokens / 1_000_000) * output_cost_per_m
        self.assertAlmostEqual(cost, expected)

    def test_prompt_overhead_adds_to_cost(self):
        task = _make_task("test")
        cost_no_overhead = estimate_task_cost([task], "opus", prompt_overhead=0)
        cost_with_overhead = estimate_task_cost([task], "opus", prompt_overhead=1000)
        self.assertGreater(cost_with_overhead, cost_no_overhead)


class TestCheckCostBudget(unittest.TestCase):

    def _make_config(self, model="opus", resolved_model="", max_cost=10.0):
        config = MagicMock()
        config.claude.model = model
        config.claude.resolved_model = resolved_model
        config.safety.max_cost_usd_per_hour = max_cost
        return config

    def _make_state(self, total_cost=0.0):
        state = MagicMock()
        state.get_total_cost.return_value = total_cost
        return state

    def test_allowed_within_budget(self):
        config = self._make_config(max_cost=10.0)
        state = self._make_state(total_cost=0.0)
        tasks = [_make_task("test")]
        allowed, estimated, remaining = check_cost_budget(tasks, config, state)
        self.assertTrue(allowed)
        self.assertGreater(estimated, 0)
        self.assertAlmostEqual(remaining, 10.0)

    def test_denied_exceeds_budget(self):
        config = self._make_config(max_cost=0.0001)
        state = self._make_state(total_cost=0.0)
        # Create tasks with enough text to generate meaningful cost
        tasks = [_make_task("a" * 10000, context="b" * 10000)]
        allowed, estimated, remaining = check_cost_budget(tasks, config, state)
        self.assertFalse(allowed)

    def test_uses_resolved_model(self):
        config = self._make_config(model="opus", resolved_model="sonnet", max_cost=10.0)
        state = self._make_state(total_cost=0.0)
        tasks = [_make_task("hello world")]
        allowed, estimated, remaining = check_cost_budget(tasks, config, state)
        self.assertTrue(allowed)
        # Verify that resolved_model was used (sonnet pricing is cheaper)
        config_opus = self._make_config(model="opus", resolved_model="", max_cost=10.0)
        _, est_opus, _ = check_cost_budget(tasks, config_opus, state)
        # sonnet cost should be less than opus cost
        self.assertLess(estimated, est_opus)

    def test_remaining_budget_accounts_for_spent(self):
        config = self._make_config(max_cost=10.0)
        state = self._make_state(total_cost=8.0)
        tasks = [_make_task("test")]
        allowed, estimated, remaining = check_cost_budget(tasks, config, state)
        self.assertAlmostEqual(remaining, 2.0)


if __name__ == "__main__":
    unittest.main()
