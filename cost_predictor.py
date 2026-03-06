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

# Default cost per million tokens (input) — conservative estimates for Claude models.
# These are upper-bound estimates to err on the side of caution.
# Can be overridden via config.pricing.cost_per_million_input_tokens.
_DEFAULT_MODEL_COST_PER_M_INPUT_TOKENS = {
    "opus": 15.0,
    "sonnet": 3.0,
    "haiku": 0.25,
}

# Output tokens are typically ~3x more expensive; assume output ≈ 50% of input.
# Can be overridden via config.pricing.output_cost_multiplier.
OUTPUT_TO_INPUT_RATIO = 0.5
_DEFAULT_OUTPUT_COST_MULTIPLIER = 5.0  # output tokens cost ~5x input for opus


def estimate_prompt_tokens(prompt_text: str) -> int:
    """Estimate the number of tokens in a prompt string."""
    return max(1, len(prompt_text) // CHARS_PER_TOKEN)


def estimate_task_cost(
    tasks: List["Task"],
    model: str,
    prompt_overhead: int = 500,
    model_costs: dict | None = None,
    output_cost_multiplier: float | None = None,
) -> float:
    """Estimate the cost in USD to process a batch of tasks.

    Args:
        tasks: List of tasks to estimate cost for.
        model: The model alias or ID being used.
        prompt_overhead: Fixed token overhead for prompt template, instructions, etc.
        model_costs: Optional dict of model alias -> cost per million input tokens.
        output_cost_multiplier: Optional multiplier for output token cost.

    Returns:
        Estimated cost in USD.
    """
    costs = model_costs or _DEFAULT_MODEL_COST_PER_M_INPUT_TOKENS
    out_mult = output_cost_multiplier if output_cost_multiplier is not None else _DEFAULT_OUTPUT_COST_MULTIPLIER

    # Calculate total character count from task descriptions and context
    total_chars = 0
    for task in tasks:
        total_chars += len(task.description)
        total_chars += len(task.context)

    input_tokens = (total_chars // CHARS_PER_TOKEN) + prompt_overhead
    output_tokens = int(input_tokens * OUTPUT_TO_INPUT_RATIO)

    # Look up per-token cost for the model
    model_lower = model.lower()
    input_cost_per_m = costs.get(model_lower)
    if input_cost_per_m is None:
        # Try to match by model ID prefix
        for alias, cost in costs.items():
            if alias in model_lower:
                input_cost_per_m = cost
                break
        else:
            # Default to opus pricing (most expensive, safest estimate)
            input_cost_per_m = costs.get("opus", _DEFAULT_MODEL_COST_PER_M_INPUT_TOKENS["opus"])

    output_cost_per_m = input_cost_per_m * out_mult

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

    # Read pricing from config if available, falling back to defaults
    model_costs = None
    output_cost_multiplier = None
    pricing = getattr(config, "pricing", None)
    if pricing is not None:
        costs_dict = getattr(pricing, "cost_per_million_input_tokens", None)
        if isinstance(costs_dict, dict) and costs_dict:
            model_costs = costs_dict
        out_mult = getattr(pricing, "output_cost_multiplier", None)
        if isinstance(out_mult, (int, float)):
            output_cost_multiplier = float(out_mult)

    estimated = estimate_task_cost(
        tasks, model,
        model_costs=model_costs,
        output_cost_multiplier=output_cost_multiplier,
    )

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
