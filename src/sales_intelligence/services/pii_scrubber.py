from __future__ import annotations

import re

from sales_intelligence.models.schemas import GuardResult

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(\+1[-\s]?)?(\(?\d{3}\)?[-\s]?)\d{3}[-\s]?\d{4}")
CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
AMOUNT_RE = re.compile(r"\$\d{1,3}(?:,\d{3})*(?:\.\d+)?|\$\d+(?:\.\d+)?[MK]?")

INJECTION_PATTERNS = [
    "ignore previous instructions",
    "disregard your system prompt",
    "you are now",
    "new instruction:",
    "forget everything",
]


def scrub_pii(transcript: str) -> str:
    """Replace emails, phones, card numbers, and financial amounts with typed placeholders.

    Note: name scrubbing is intentionally omitted — title-cased regex cannot reliably
    distinguish personal names from company names, products, and locations without NLP.
    Emails and phone numbers provide sufficient contact-detail protection for this use case.
    """
    transcript = EMAIL_RE.sub("[EMAIL]", transcript)
    transcript = PHONE_RE.sub("[PHONE]", transcript)
    transcript = CARD_RE.sub("[CARD]", transcript)
    transcript = AMOUNT_RE.sub("[AMOUNT]", transcript)
    return transcript


def count_scrubbed_entities(scrubbed: str) -> dict[str, int]:
    """Count each placeholder type in an already-scrubbed string; used for README metrics."""
    return {
        "EMAIL": scrubbed.count("[EMAIL]"),
        "PHONE": scrubbed.count("[PHONE]"),
        "CARD": scrubbed.count("[CARD]"),
        "AMOUNT": scrubbed.count("[AMOUNT]"),
    }


def check_injection(text: str, max_length: int = 600_000) -> GuardResult:
    """Reject input that exceeds length limit or matches known prompt injection patterns."""
    if len(text) > max_length:
        return GuardResult(
            is_safe=False,
            reason="Input exceeds maximum length",
        )
    lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if pattern in lower:
            return GuardResult(
                is_safe=False,
                reason="Potential prompt injection detected",
            )
    return GuardResult(
        is_safe=True,
        reason=None,
    )
