from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field


class SalesCallAnalysis(BaseModel):
    summary: str = Field(description="Executive summary of the sales call")
    objections: list[str] = Field(
        description="Customer objections raised during the call"
    )
    action_items: list[str] = Field(description="Follow-up actions for the sales team")
    sentiment: Literal["Positive", "Neutral", "Negative"] = Field(
        description="Overall sentiment of the sales call"
    )


class StreamMetadata(BaseModel):
    request_id: str
    transcript: str
    analysis: SalesCallAnalysis
    low_confidence: bool
    confidence_message: str | None
    whisper_cost: float
    claude_cost: float
    total_cost: float


@dataclass
class TranscriptionResult:
    transcript: str
    duration_seconds: float | None


@dataclass
class ConfidenceResult:
    is_low_confidence: bool
    flag_message: str | None


@dataclass
class GuardResult:
    is_safe: bool
    reason: str | None
