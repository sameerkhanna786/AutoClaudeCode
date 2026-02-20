"""Cost prediction: estimate token count and cost before executing tasks."""

from __future__ import annotations

import logging
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from config_schema import Config
    from state import StateManager
    from task_discovery import Task

logger = logging.getLogger(__name__)

# Rough approximation: 1 token ≈ 4 characters for English text
CHARS_PER_TOKEN = 4

# Cost per million tokens (input) — conservative estimates for Claude models
# These are upper-bound estimates to err on the side of caution.
_MODEL_COST_PER_M_INPUT_TOKENS = {
    "opus": 15.0,
    "sonnet": 3.0,
    "haiku": 0.25,
}

# Output tokens are typically ~3x more expensive; assume output ≈ 50% of input
OUTPUT_TO_INPUT_RATIO = 0.5
OUTPUT_COST_MULTIPLIER = 5.0  # output tokens cost ~5x input for opus


def estimate_prompt_tokens(prompt_text: str) -> int:
    """Estimate the number of tokens in a prompt string."""
    return max(1, len(prompt_text) // CHARS_PER_TOKEN)


def estimate_task_cost(
    tasks: List["Task"],
    model: str,
    prompt_overhead: int = 500,
) -> float:
    """Estimate the cost in USD to process a batch of tasks.

    Args:
        tasks: List of tasks to estimate cost for.
        model: The model alias or ID being used.
        prompt_overhead: Fixed token overhead for prompt template, instructions, etc.

    Returns:
        Estimated cost in USD.
    """
    # Calculate total character count from task descriptions and context
    total_chars = 0
    for task in tasks:
        total_chars += len(task.description)
        total_chars += len(task.context)

    input_tokens = (total_chars // CHARS_PER_TOKEN) + prompt_overhead
    output_tokens = int(input_tokens * OUTPUT_TO_INPUT_RATIO)

    # Look up per-token cost for the model
    model_lower = model.lower()
    input_cost_per_m = _MODEL_COST_PER_M_INPUT_TOKENS.get(model_lower)
    if input_cost_per_m is None:
        # Try to match by model ID prefix
        for alias, cost in _MODEL_COST_PER_M_INPUT_TOKENS.items():
            if alias in model_lower:
                input_cost_per_m = cost
                break
        else:
            # Default to opus pricing (most expensive, safest estimate)
            input_cost_per_m = _MODEL_COST_PER_M_INPUT_TOKENS["opus"]

    output_cost_per_m = input_cost_per_m * OUTPUT_COST_MULTIPLIER

    estimated_cost = (
        (input_tokens / 1_000_000) * input_cost_per_m
        + (output_tokens / 1_000_000) * output_cost_per_m
    )

    return estimated_cost


def check_cost_budget(
    tasks: List["Task"],
    config: "Config",
    state: "StateManager",
) -> tuple[bool, float, float]:
    """Check whether executing the given tasks would likely exceed the hourly budget.

    Returns:
        (allowed, estimated_cost, remaining_budget)
        - allowed: True if the estimated cost fits within remaining budget
        - estimated_cost: the predicted cost in USD
        - remaining_budget: USD remaining in the current hourly window
    """
    model = config.claude.resolved_model or config.claude.model
    estimated = estimate_task_cost(tasks, model)

    hourly_spent = state.get_total_cost(lookback_seconds=3600)
    hourly_limit = config.safety.max_cost_usd_per_hour
    remaining = hourly_limit - hourly_spent

    allowed = estimated < remaining

    if not allowed:
        logger.warning(
            "Cost prediction: estimated $%.4f for %d task(s) exceeds "
            "remaining budget $%.4f (spent $%.4f of $%.4f/hr limit)",
            estimated, len(tasks), remaining, hourly_spent, hourly_limit,
        )
    else:
        logger.debug(
            "Cost prediction: estimated $%.4f for %d task(s), "
            "remaining budget $%.4f",
            estimated, len(tasks), remaining,
        )

    return allowed, estimated, remaining
