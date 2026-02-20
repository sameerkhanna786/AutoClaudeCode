"""Structured JSON logging formatter for log aggregation integration."""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON objects.

    Output format:
        {"timestamp": "...", "level": "...", "logger": "...", "message": "...", ...}

    Includes exception info when present.
    """

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)

        if record.stack_info:
            entry["stack_info"] = record.stack_info

        return json.dumps(entry, default=str)


def apply_json_logging() -> None:
    """Replace all handlers on the root logger with JSONFormatter.

    Call this after main.py's setup_logging() has configured the handlers.
    Existing handlers are preserved but their formatter is switched to JSON.
    """
    formatter = JSONFormatter()
    root = logging.getLogger()
    for handler in root.handlers:
        handler.setFormatter(formatter)
