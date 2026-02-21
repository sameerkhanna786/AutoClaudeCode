"""Tests for telemetry module."""

import time

import pytest

from telemetry import (
    _compute_distribution,
    _compute_hourly_buckets,
    _compute_type_breakdown,
    _empty_metrics,
    compute_metrics,
)


class TestComputeDistribution:
    def test_empty_list(self):
        result = _compute_distribution([])
        assert result["count"] == 0
        assert result["min"] == 0
        assert result["max"] == 0
        assert result["mean"] == 0
        assert result["median"] == 0
        assert result["p90"] == 0
        assert result["p95"] == 0

    def test_single_value(self):
        result = _compute_distribution([5.0])
        assert result["count"] == 1
        assert result["min"] == 5.0
        assert result["max"] == 5.0
        assert result["mean"] == 5.0
        assert result["median"] == 5.0

    def test_two_values(self):
        result = _compute_distribution([2.0, 8.0])
        assert result["count"] == 2
        assert result["min"] == 2.0
        assert result["max"] == 8.0
        assert result["mean"] == 5.0

    def test_multiple_values(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = _compute_distribution(values)
        assert result["count"] == 10
        assert result["min"] == 1.0
        assert result["max"] == 10.0
        assert result["mean"] == 5.5
        assert result["median"] == 5.0  # idx = int(0.5 * 9) = 4 -> 5th element = 5.0
        assert result["p90"] == 9.0     # idx = int(0.9 * 9) = 8 -> 9th element = 9.0
        assert result["p95"] == 9.0     # idx = int(0.95 * 9) = 8 -> 9th element = 9.0

    def test_unsorted_input(self):
        values = [10.0, 1.0, 5.0, 3.0, 7.0]
        result = _compute_distribution(values)
        assert result["min"] == 1.0
        assert result["max"] == 10.0
        assert result["count"] == 5

    def test_values_are_rounded(self):
        values = [1.11111, 2.22222, 3.33333]
        result = _compute_distribution(values)
        assert result["min"] == 1.1111
        assert result["max"] == 3.3333
        assert result["mean"] == 2.2222


class TestComputeHourlyBuckets:
    def test_empty_records(self):
        now = time.time()
        buckets = _compute_hourly_buckets([], now)
        assert len(buckets) == 24
        for b in buckets:
            assert b["cycles"] == 0
            assert b["successes"] == 0
            assert b["failures"] == 0
            assert b["cost_usd"] == 0

    def test_records_in_current_hour(self):
        now = time.time()
        records = [
            {"timestamp": now - 100, "success": True, "cost_usd": 0.5},
            {"timestamp": now - 200, "success": False, "cost_usd": 0.3},
        ]
        buckets = _compute_hourly_buckets(records, now)
        # hours_ago=0 is the current hour bucket
        assert buckets[0]["cycles"] == 2
        assert buckets[0]["successes"] == 1
        assert buckets[0]["failures"] == 1
        assert buckets[0]["cost_usd"] == 0.8

    def test_records_spread_across_hours(self):
        now = time.time()
        records = [
            {"timestamp": now - 100, "success": True, "cost_usd": 0.1},
            {"timestamp": now - 3700, "success": True, "cost_usd": 0.2},  # 1 hour ago
            {"timestamp": now - 7300, "success": False, "cost_usd": 0.3},  # 2 hours ago
        ]
        buckets = _compute_hourly_buckets(records, now)
        assert buckets[0]["cycles"] == 1
        assert buckets[1]["cycles"] == 1
        assert buckets[2]["cycles"] == 1

    def test_hours_ago_field(self):
        now = time.time()
        buckets = _compute_hourly_buckets([], now)
        for i, b in enumerate(buckets):
            assert b["hours_ago"] == i


class TestComputeTypeBreakdown:
    def test_empty_records(self):
        result = _compute_type_breakdown([])
        assert result == {}

    def test_single_type(self):
        records = [
            {"task_type": "lint", "success": True},
            {"task_type": "lint", "success": False},
            {"task_type": "lint", "success": True},
        ]
        result = _compute_type_breakdown(records)
        assert "lint" in result
        assert result["lint"]["total"] == 3
        assert result["lint"]["successes"] == 2
        assert result["lint"]["failures"] == 1

    def test_multiple_types(self):
        records = [
            {"task_type": "lint", "success": True},
            {"task_type": "test_failure", "success": False},
            {"task_type": "todo", "success": True},
        ]
        result = _compute_type_breakdown(records)
        assert len(result) == 3
        assert result["lint"]["total"] == 1
        assert result["test_failure"]["total"] == 1
        assert result["todo"]["total"] == 1

    def test_missing_task_type_defaults_to_unknown(self):
        records = [{"success": True}]
        result = _compute_type_breakdown(records)
        assert "unknown" in result
        assert result["unknown"]["total"] == 1

    def test_missing_success_defaults_to_false(self):
        records = [{"task_type": "lint"}]
        result = _compute_type_breakdown(records)
        assert result["lint"]["failures"] == 1
        assert result["lint"]["successes"] == 0


class TestEmptyMetrics:
    def test_structure(self):
        result = _empty_metrics()
        assert result["total_cycles"] == 0
        assert result["successes"] == 0
        assert result["failures"] == 0
        assert result["success_rate"] == 0.0
        assert result["duration"]["count"] == 0
        assert result["duration"]["unit"] == "seconds"
        assert result["validation_retries"]["count"] == 0
        assert result["validation_retries"]["zero_retry_pct"] == 0.0
        assert result["cost"]["count"] == 0
        assert result["cost"]["total"] == 0.0
        assert result["cost"]["unit"] == "USD"
        assert result["batch_size"]["count"] == 0
        assert result["hourly_buckets"] == []
        assert result["type_breakdown"] == {}
        assert result["pipeline_cycles"] == 0


class TestComputeMetrics:
    def test_no_records(self):
        result = compute_metrics([])
        assert result["total_cycles"] == 0

    def test_records_outside_lookback(self):
        old_ts = time.time() - 200000  # well outside 24h window
        records = [{"timestamp": old_ts, "success": True}]
        result = compute_metrics(records, lookback_seconds=86400)
        assert result["total_cycles"] == 0

    def test_basic_metrics(self):
        now = time.time()
        records = [
            {
                "timestamp": now - 100,
                "success": True,
                "duration_seconds": 10.0,
                "validation_retry_count": 0,
                "cost_usd": 0.5,
                "task_type": "lint",
                "batch_size": 2,
            },
            {
                "timestamp": now - 200,
                "success": False,
                "duration_seconds": 20.0,
                "validation_retry_count": 2,
                "cost_usd": 0.3,
                "task_type": "test_failure",
                "batch_size": 1,
            },
        ]
        result = compute_metrics(records, lookback_seconds=86400)

        assert result["total_cycles"] == 2
        assert result["successes"] == 1
        assert result["failures"] == 1
        assert result["success_rate"] == 50.0
        assert result["cost"]["total"] == 0.8
        assert result["cost"]["unit"] == "USD"
        assert result["duration"]["unit"] == "seconds"
        assert result["duration"]["count"] == 2

    def test_success_rate_all_success(self):
        now = time.time()
        records = [
            {"timestamp": now - 100, "success": True, "duration_seconds": 5.0},
            {"timestamp": now - 200, "success": True, "duration_seconds": 10.0},
        ]
        result = compute_metrics(records)
        assert result["success_rate"] == 100.0

    def test_success_rate_all_failure(self):
        now = time.time()
        records = [
            {"timestamp": now - 100, "success": False},
            {"timestamp": now - 200, "success": False},
        ]
        result = compute_metrics(records)
        assert result["success_rate"] == 0.0

    def test_validation_retry_zero_pct(self):
        now = time.time()
        records = [
            {"timestamp": now - 100, "validation_retry_count": 0},
            {"timestamp": now - 200, "validation_retry_count": 0},
            {"timestamp": now - 300, "validation_retry_count": 3},
        ]
        result = compute_metrics(records)
        # 2 out of 3 had zero retries = 66.7%
        assert result["validation_retries"]["zero_retry_pct"] == 66.7

    def test_pipeline_cycles_counted(self):
        now = time.time()
        records = [
            {"timestamp": now - 100, "pipeline_mode": True},
            {"timestamp": now - 200, "pipeline_mode": False},
            {"timestamp": now - 300, "pipeline_mode": True},
        ]
        result = compute_metrics(records)
        assert result["pipeline_cycles"] == 2

    def test_batch_size_stats(self):
        now = time.time()
        records = [
            {"timestamp": now - 100, "batch_size": 3},
            {"timestamp": now - 200, "batch_size": 5},
            {"timestamp": now - 300, "batch_size": 1},
        ]
        result = compute_metrics(records)
        assert result["batch_size"]["min"] == 1
        assert result["batch_size"]["max"] == 5
        assert result["batch_size"]["count"] == 3

    def test_custom_lookback(self):
        now = time.time()
        records = [
            {"timestamp": now - 100, "success": True},
            {"timestamp": now - 5000, "success": True},  # outside 1h lookback
        ]
        result = compute_metrics(records, lookback_seconds=3600)
        assert result["total_cycles"] == 1
        assert result["lookback_seconds"] == 3600

    def test_hourly_buckets_present(self):
        now = time.time()
        records = [{"timestamp": now - 100, "success": True}]
        result = compute_metrics(records)
        assert len(result["hourly_buckets"]) == 24

    def test_type_breakdown_present(self):
        now = time.time()
        records = [
            {"timestamp": now - 100, "task_type": "lint", "success": True},
            {"timestamp": now - 200, "task_type": "lint", "success": False},
        ]
        result = compute_metrics(records)
        assert "lint" in result["type_breakdown"]
        assert result["type_breakdown"]["lint"]["total"] == 2

    def test_duration_filters_zero(self):
        """Records with duration_seconds=0 are excluded from duration stats."""
        now = time.time()
        records = [
            {"timestamp": now - 100, "duration_seconds": 0},
            {"timestamp": now - 200, "duration_seconds": 10.0},
        ]
        result = compute_metrics(records)
        # Only the non-zero duration record should be counted
        assert result["duration"]["count"] == 1
        assert result["duration"]["min"] == 10.0

    def test_missing_fields_use_defaults(self):
        """Records missing optional fields should not cause errors."""
        now = time.time()
        records = [{"timestamp": now - 100}]
        result = compute_metrics(records)
        assert result["total_cycles"] == 1
        assert result["successes"] == 0
        assert result["cost"]["total"] == 0.0
