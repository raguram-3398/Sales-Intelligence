from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sales_intelligence.services.cost_tracker import calculate_whisper_cost
from sales_intelligence.services.transcription import (
    _get_client as transcription_get_client,
)
from sales_intelligence.services.transcription import check_confidence, transcribe_audio


def test_calculate_whisper_cost():
    """None → 0.0; partial minute billed as full; two minutes correct."""
    assert calculate_whisper_cost(None) == 0.0
    assert abs(calculate_whisper_cost(45) - 0.006) < 1e-9  # ceil(45/60)=1 min
    assert abs(calculate_whisper_cost(90) - 0.012) < 1e-9  # ceil(90/60)=2 min


def test_check_confidence_clean():
    result = check_confidence(
        "The customer was interested in our pricing.", request_id="r1"
    )
    assert result.is_low_confidence is False


def test_check_confidence_flagged():
    """More than 30% consecutive repeated words triggers low-confidence flag."""
    result = check_confidence(
        "the the the the customer customer was interested", request_id="r1"
    )
    assert result.is_low_confidence is True
    assert result.flag_message is not None


async def test_transcribe_audio_happy_path():
    """Returns transcript and duration when Whisper succeeds."""
    mock_response = MagicMock()
    mock_response.text = "Hello this is a test transcript."
    mock_response.duration = 30.0

    mock_client = MagicMock()
    mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)

    with patch(
        "sales_intelligence.services.transcription._get_client",
        return_value=mock_client,
    ):
        result = await transcribe_audio(
            audio_bytes=b"fake_audio", filename="test.mp3", request_id="req-1"
        )

    assert result.transcript == "Hello this is a test transcript."
    assert result.duration_seconds == 30.0


async def test_transcribe_audio_timeout():
    """Raises TimeoutError when Whisper transcription exceeds timeout."""
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create = AsyncMock(side_effect=TimeoutError())

    with patch(
        "sales_intelligence.services.transcription._get_client",
        return_value=mock_client,
    ):
        with pytest.raises(TimeoutError, match="Whisper timed out"):
            await transcribe_audio(
                audio_bytes=b"fake_audio",
                filename="test.mp3",
                request_id="req-1",
                timeout=0.001,
            )


def test_get_client_uses_header_key():
    """Header key takes priority over everything."""
    client = transcription_get_client("sk-openai-header")
    assert client.api_key == "sk-openai-header"


def test_get_client_falls_back_to_env(monkeypatch):
    """No header key + no REQUIRE_USER_KEYS → uses env var."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-env")
    monkeypatch.delenv("REQUIRE_USER_KEYS", raising=False)
    client = transcription_get_client(None)
    assert client.api_key == "sk-openai-env"


def test_get_client_raises_when_require_user_keys_set(monkeypatch):
    """No header key + REQUIRE_USER_KEYS=true → ValueError."""
    monkeypatch.setenv("REQUIRE_USER_KEYS", "true")
    with pytest.raises(ValueError, match="OpenAI API key is required"):
        transcription_get_client(None)
