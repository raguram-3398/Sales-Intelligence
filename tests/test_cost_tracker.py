from __future__ import annotations

import logging

import pytest

import sales_intelligence.services.cost_tracker as ct
from sales_intelligence.api.routes import trim_transcript
from sales_intelligence.services.cost_tracker import calculate_claude_cost, record_spend


@pytest.fixture(autouse=True)
def reset_daily_spend():
    """Reset module-level spend state before each test to prevent bleed-through."""
    ct._daily_spend = 0.0
    ct._threshold_alerted = False
    yield
    ct._daily_spend = 0.0
    ct._threshold_alerted = False


def test_trim_transcript_short_unchanged():
    text = "This is a short transcript."
    result, was_trimmed = trim_transcript(text)
    assert result == text and was_trimmed is False


def test_trim_transcript_long_trimmed():
    """Transcript over 150K token budget is trimmed and notice appended."""
    text = "word " * 160_000
    result, was_trimmed = trim_transcript(text)
    assert was_trimmed is True
    assert result.endswith("[Transcript trimmed - original exceeded context budget]")


def test_calculate_claude_cost():
    """1M input tokens = $3, 1M output tokens = $15 at current pricing."""
    input_cost, output_cost = calculate_claude_cost(1_000_000, 1_000_000)
    assert abs(input_cost - 3.0) < 1e-9
    assert abs(output_cost - 15.0) < 1e-9


def test_record_spend_accumulates_and_alerts_once(caplog):
    """Daily spend accumulates correctly and threshold warning fires exactly once."""
    with caplog.at_level(logging.WARNING, logger="services.cost_tracker"):
        record_spend(0.80, request_id="req-1")
        record_spend(0.30, request_id="req-2")  # crosses $1.00 threshold
        record_spend(0.50, request_id="req-3")  # must NOT fire again

    assert abs(ct._daily_spend - 1.60) < 1e-9
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
