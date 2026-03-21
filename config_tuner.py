"""Config auto-tuning: analyze history and recommend configuration changes."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TuningRecommendation:
    """A single configuration tuning recommendation."""
    field: str
    current_value: Any
    recommended_value: Any
    reason: str
    confidence: float  # 0.0 to 1.0


class ConfigTuner:
    """Analyzes cycle history and recommends configuration tuning.

    Examines patterns in historical records (success rates, durations,
    costs) and produces recommendations for adjusting orchestrator
    parameters to improve throughput and reduce waste.
    """

    def __init__(self, state_dir: str):
        self._state_dir = Path(state_dir)
        self._recommendations_file = self._state_dir / "tuning_recommendations.json"

    def analyze(self, records: List[Dict[str, Any]], config: Any) -> List[TuningRecommendation]:
        """Analyze history records and generate tuning recommendations.

        Args:
            records: List of cycle history records (dicts).
            config: The current Config object.

        Returns:
            List of TuningRecommendation objects.
        """
        if not records or len(records) < 5:
            return []

        recs: List[TuningRecommendation] = []
        recs.extend(self._analyze_max_turns(records, config))
        recs.extend(self._analyze_batch_size(records, config))
        recs.extend(self._analyze_loop_interval(records, config))
        recs.extend(self._analyze_retries(records, config))
        return recs

    def save_recommendations(self, recs: List[TuningRecommendation]) -> None:
        """Save recommendations to a JSON file in the state directory.

        Uses atomic temp-file + os.replace to prevent corrupt files on crash.
        """
        if not recs:
            return
        self._state_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "timestamp": time.time(),
            "recommendations": [asdict(r) for r in recs],
        }
        tmp_path = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(self._state_dir), suffix=".tmp",
            )
            try:
                f = os.fdopen(tmp_fd, "w", encoding="utf-8")
            except Exception:
                os.close(tmp_fd)
                raise
            with f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp_path, str(self._recommendations_file))
            logger.info(
                "Saved %d tuning recommendations to %s",
                len(recs), self._recommendations_file,
            )
        except Exception as e:
            logger.warning("Failed to save tuning recommendations: %s", e)
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _analyze_max_turns(
        self, records: List[Dict[str, Any]], config: Any,
    ) -> List[TuningRecommendation]:
        """Recommend max_turns adjustments based on duration patterns."""
        recs: List[TuningRecommendation] = []
        recent = records[-20:]
        successful = [r for r in recent if r.get("success", False)]

        if len(successful) < 3:
            return recs

        durations = [r.get("duration_seconds", 0.0) for r in successful]
        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)

        current_max_turns = config.claude.max_turns

        # If successful cycles are consistently fast, max_turns might be too high
        if avg_duration < 60 and current_max_turns > 15:
            recs.append(TuningRecommendation(
                field="claude.max_turns",
                current_value=current_max_turns,
                recommended_value=max(15, current_max_turns - 5),
                reason=f"Successful cycles average {avg_duration:.0f}s — "
                       f"reducing max_turns may save cost without impacting success",
                confidence=0.5,
            ))
        # If cycles are slow and hitting high durations, max_turns may be too low
        elif max_duration > 600 and current_max_turns < 40:
            recs.append(TuningRecommendation(
                field="claude.max_turns",
                current_value=current_max_turns,
                recommended_value=min(40, current_max_turns + 5),
                reason=f"Some cycles take {max_duration:.0f}s — "
                       f"increasing max_turns may help complex tasks complete",
                confidence=0.4,
            ))

        return recs

    def _analyze_batch_size(
        self, records: List[Dict[str, Any]], config: Any,
    ) -> List[TuningRecommendation]:
        """Recommend batch size adjustments based on success patterns."""
        recs: List[TuningRecommendation] = []
        recent = records[-20:]

        if len(recent) < 5:
            return recs

        # Analyze batch success rates
        batch_records = [r for r in recent if r.get("batch_size", 1) > 1]
        single_records = [r for r in recent if r.get("batch_size", 1) == 1]

        if batch_records and single_records:
            batch_success = sum(1 for r in batch_records if r.get("success")) / len(batch_records)
            single_success = sum(1 for r in single_records if r.get("success")) / len(single_records)

            current_max = config.orchestrator.max_batch_size

            if batch_success < 0.3 and single_success > 0.7 and current_max > 3:
                recs.append(TuningRecommendation(
                    field="orchestrator.max_batch_size",
                    current_value=current_max,
                    recommended_value=max(3, current_max - 2),
                    reason=f"Batch success rate ({batch_success:.0%}) much lower than "
                           f"single ({single_success:.0%}) — reduce batch size",
                    confidence=0.7,
                ))
            elif batch_success > 0.8 and current_max < 15:
                recs.append(TuningRecommendation(
                    field="orchestrator.max_batch_size",
                    current_value=current_max,
                    recommended_value=min(15, current_max + 2),
                    reason=f"Batch success rate is high ({batch_success:.0%}) — "
                           f"increasing max_batch_size may improve throughput",
                    confidence=0.5,
                ))

        return recs

    def _analyze_loop_interval(
        self, records: List[Dict[str, Any]], config: Any,
    ) -> List[TuningRecommendation]:
        """Recommend loop interval adjustments based on idle patterns."""
        recs: List[TuningRecommendation] = []
        recent = records[-20:]

        if len(recent) < 5:
            return recs

        # Count cycles with no tasks — use dedicated 'no_tasks' field first
        # (the orchestrator returns early without recording a CycleRecord when
        # no tasks are found, so error-string matching alone never fires).
        # Fall back to error-string matching for backwards compatibility with
        # any manually-recorded records.
        no_task_count = sum(
            1 for r in recent
            if r.get("no_tasks") or (
                not r.get("success") and re.search(
                    r'\bno\s+(?:actionable\s+)?tasks?\b',
                    r.get("error") or "", re.IGNORECASE,
                )
            )
        )

        current_interval = config.orchestrator.loop_interval_seconds

        if no_task_count > len(recent) * 0.5 and current_interval < 120:
            recs.append(TuningRecommendation(
                field="orchestrator.loop_interval_seconds",
                current_value=current_interval,
                recommended_value=min(120, current_interval * 2),
                reason=f"{no_task_count}/{len(recent)} recent cycles found no tasks — "
                       f"increasing interval reduces unnecessary cycles",
                confidence=0.6,
            ))
        elif no_task_count == 0 and current_interval > 60:
            recs.append(TuningRecommendation(
                field="orchestrator.loop_interval_seconds",
                current_value=current_interval,
                recommended_value=max(30, current_interval // 2),
                reason="All recent cycles found tasks — "
                       "decreasing interval may improve responsiveness",
                confidence=0.4,
            ))

        return recs

    def _analyze_retries(
        self, records: List[Dict[str, Any]], config: Any,
    ) -> List[TuningRecommendation]:
        """Recommend validation retry adjustments based on retry patterns."""
        recs: List[TuningRecommendation] = []
        recent = records[-20:]

        if len(recent) < 5:
            return recs

        retry_counts = [r.get("validation_retry_count", 0) for r in recent]
        max_retries_seen = max(retry_counts)

        current_max_retries = config.orchestrator.max_validation_retries

        # If retries are consistently exhausted but rarely succeed after retry
        high_retry = [r for r in recent if r.get("validation_retry_count", 0) >= current_max_retries]
        if high_retry:
            retry_then_fail = sum(1 for r in high_retry if not r.get("success"))
            if retry_then_fail == len(high_retry) and current_max_retries > 2:
                recs.append(TuningRecommendation(
                    field="orchestrator.max_validation_retries",
                    current_value=current_max_retries,
                    recommended_value=max(2, current_max_retries - 1),
                    reason=f"All {len(high_retry)} cycles that hit max retries still failed — "
                           f"reducing retries saves cost on hopeless attempts",
                    confidence=0.6,
                ))

        # If retries never needed
        if max_retries_seen == 0 and current_max_retries > 3:
            recs.append(TuningRecommendation(
                field="orchestrator.max_validation_retries",
                current_value=current_max_retries,
                recommended_value=3,
                reason="No recent cycles needed validation retries — "
                       "reducing max_validation_retries is safe",
                confidence=0.5,
            ))

        return recs
