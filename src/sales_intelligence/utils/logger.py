from __future__ import annotations

import json
import logging
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    """Serialize a log record to a structured JSON string with timestamp and extra fields."""

    def format(
        self,
        record: logging.LogRecord,
    ) -> str:
        log_data = {
            "timestamp": datetime.now(
                UTC,
            ).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        extra_fields = getattr(record, "extra_fields", None)
        if extra_fields:
            log_data.update(extra_fields)
        return json.dumps(log_data)


def get_logger(
    name: str,
) -> logging.Logger:
    """Return a named logger with JSON formatting attached; idempotent on repeated calls."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_request(
    logger: logging.Logger,
    request_id: str,
    event: str,
    **kwargs: str | float | int | bool | None,
) -> None:
    """Emit a structured INFO log entry binding request_id and any extra kwargs to the event."""
    logger.info(
        event,
        extra={
            "extra_fields": {
                "request_id": request_id,
                **kwargs,
            }
        },
    )
