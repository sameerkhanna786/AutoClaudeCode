"""Telemetry and metrics export: track orchestrator performance over time.

Computes cycle duration distribution, validation retry patterns, cost trends,
and success/failure rates from the cycle history.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def compute_metrics(
    records: List[Dict[str, Any]],
    lookback_seconds: int = 86400,
) -> Dict[str, Any]:
    """Compute orchestrator performance metrics from cycle history records.

    Args:
        records: List of cycle record dicts from history.json.
        lookback_seconds: Only consider records within this window (default 24h).

    Returns:
        Dict containing duration stats, retry stats, cost trends, and rates.
    """
    now = time.time()
    cutoff = now - lookback_seconds

    recent = [r for r in records if r.get("timestamp", 0) >= cutoff]

    if not recent:
        return _empty_metrics()

    # Duration distribution
    durations = [r.get("duration_seconds", 0.0) for r in recent if r.get("duration_seconds", 0.0) > 0]
    duration_stats = _compute_distribution(durations)

    # Validation retry patterns
    retry_counts = [r.get("validation_retry_count", 0) for r in recent]
    retry_stats = _compute_distribution(retry_counts)
    retry_zero_pct = (
        sum(1 for c in retry_counts if c == 0) / len(retry_counts) * 100
        if retry_counts else 0
    )

    # Cost trends
    costs = [r.get("cost_usd", 0.0) for r in recent]
    cost_stats = _compute_distribution(costs)
    total_cost = sum(costs)

    # Success/failure rates
    total = len(recent)
    successes = sum(1 for r in recent if r.get("success", False))
    failures = total - successes
    success_rate = (successes / total * 100) if total > 0 else 0.0

    # Hourly breakdown (last 24 hours in 1-hour buckets)
    hourly_buckets = _compute_hourly_buckets(recent, now)

    # Task type breakdown
    type_breakdown = _compute_type_breakdown(recent)

    # Pipeline mode stats
    pipeline_cycles = [r for r in recent if r.get("pipeline_mode")]
    pipeline_count = len(pipeline_cycles)

    # Batch size distribution
    batch_sizes = [r.get("batch_size", 1) for r in recent]
    batch_stats = _compute_distribution(batch_sizes)

    return {
        "lookback_seconds": lookback_seconds,
        "total_cycles": total,
        "successes": successes,
        "failures": failures,
        "success_rate": round(success_rate, 1),
        "duration": {
            **duration_stats,
            "unit": "seconds",
        },
        "validation_retries": {
            **retry_stats,
            "zero_retry_pct": round(retry_zero_pct, 1),
        },
        "cost": {
            **cost_stats,
            "total": round(total_cost, 4),
            "unit": "USD",
        },
        "batch_size": batch_stats,
        "hourly_buckets": hourly_buckets,
        "type_breakdown": type_breakdown,
        "pipeline_cycles": pipeline_count,
    }


def _compute_distribution(values: List[float]) -> Dict[str, Any]:
    """Compute min, max, mean, median, p90, p95 for a list of values."""
    if not values:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "p90": 0, "p95": 0, "count": 0}

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mean_val = sum(sorted_vals) / n

    def percentile(pct: float) -> float:
        k = pct / 100 * (n - 1)
        floor_idx = int(k)
        ceil_idx = min(floor_idx + 1, n - 1)
        frac = k - floor_idx
        return sorted_vals[floor_idx] + frac * (sorted_vals[ceil_idx] - sorted_vals[floor_idx])

    return {
        "min": round(sorted_vals[0], 4),
        "max": round(sorted_vals[-1], 4),
        "mean": round(mean_val, 4),
        "median": round(percentile(50), 4),
        "p90": round(percentile(90), 4),
        "p95": round(percentile(95), 4),
        "count": n,
    }


def _compute_hourly_buckets(
    records: List[Dict[str, Any]],
    now: float,
) -> List[Dict[str, Any]]:
    """Compute per-hour aggregates for the last 24 hours.

    Uses a single O(n) pass over records instead of scanning all records
    24 times (once per bucket).
    """
    # Initialize accumulators for each bucket
    cycles = [0] * 24
    successes = [0] * 24
    costs = [0.0] * 24

    cutoff = now - 24 * 3600
    for r in records:
        ts = r.get("timestamp", 0)
        if ts < cutoff or ts > now:
            continue
        hours_ago = int((now - ts) / 3600)
        if 0 <= hours_ago < 24:
            cycles[hours_ago] += 1
            if r.get("success", False):
                successes[hours_ago] += 1
            costs[hours_ago] += r.get("cost_usd", 0.0)

    return [
        {
            "hours_ago": h,
            "cycles": cycles[h],
            "successes": successes[h],
            "failures": cycles[h] - successes[h],
            "cost_usd": round(costs[h], 4),
        }
        for h in range(24)
    ]


def _compute_type_breakdown(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Compute success/failure counts per task type."""
    breakdown: Dict[str, Dict[str, int]] = {}

    for r in records:
        task_type = r.get("task_type", "unknown")
        if task_type not in breakdown:
            breakdown[task_type] = {"total": 0, "successes": 0, "failures": 0}
        breakdown[task_type]["total"] += 1
        if r.get("success", False):
            breakdown[task_type]["successes"] += 1
        else:
            breakdown[task_type]["failures"] += 1

    return breakdown


def _empty_metrics() -> Dict[str, Any]:
    """Return empty metrics structure when no records are available."""
    empty_dist = {"min": 0, "max": 0, "mean": 0, "median": 0, "p90": 0, "p95": 0, "count": 0}
    return {
        "lookback_seconds": 0,
        "total_cycles": 0,
        "successes": 0,
        "failures": 0,
        "success_rate": 0.0,
        "duration": {**empty_dist, "unit": "seconds"},
        "validation_retries": {**empty_dist, "zero_retry_pct": 0.0},
        "cost": {**empty_dist, "total": 0.0, "unit": "USD"},
        "batch_size": empty_dist,
        "hourly_buckets": [],
        "type_breakdown": {},
        "pipeline_cycles": 0,
    }
