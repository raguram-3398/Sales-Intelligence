from __future__ import annotations

import math

from sales_intelligence.utils.logger import get_logger

WHISPER_COST_PER_MINUTE: float = 0.006
CLAUDE_INPUT_COST_PER_TOKEN: float = 3.0 / 1_000_000
CLAUDE_OUTPUT_COST_PER_TOKEN: float = 15.0 / 1_000_000
DAILY_SPEND_ALERT_THRESHOLD: float = 1.0
# Resets on process restart — in production persist to Database
_daily_spend: float = 0.0
_threshold_alerted: bool = False

logger = get_logger(__name__)


def calculate_claude_cost(input_tokens: int, output_tokens: int) -> tuple[float, float]:
    """Convert Claude input/output token counts to USD costs; returns (input_cost, output_cost)."""
    return (
        input_tokens * CLAUDE_INPUT_COST_PER_TOKEN,
        output_tokens * CLAUDE_OUTPUT_COST_PER_TOKEN,
    )


def calculate_whisper_cost(duration_seconds: float | None) -> float:
    """Return Whisper cost in USD billed in whole minutes; returns 0.0 if duration unknown."""
    if duration_seconds is None:
        return 0.0
    billed_minutes = math.ceil(duration_seconds / 60)
    return billed_minutes * WHISPER_COST_PER_MINUTE


def record_spend(amount: float, request_id: str) -> float:
    """Accumulate spend against the daily total, log every call, and warn once when threshold crossed."""
    global _daily_spend, _threshold_alerted
    _daily_spend += amount
    logger.info(
        "cost_recorded",
        extra={
            "extra_fields": {
                "request_id": request_id,
                "amount_usd": round(amount, 6),
                "daily_total_usd": round(_daily_spend, 6),
            }
        },
    )
    if _daily_spend >= DAILY_SPEND_ALERT_THRESHOLD and not _threshold_alerted:
        _threshold_alerted = True
        logger.warning(
            "daily_spend_threshold_exceeded",
            extra={
                "extra_fields": {
                    "daily_spend_usd": round(_daily_spend, 6),
                    "threshold_usd": DAILY_SPEND_ALERT_THRESHOLD,
                    "request_id": request_id,
                }
            },
        )
    return _daily_spend
