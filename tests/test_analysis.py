from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sales_intelligence.models.schemas import SalesCallAnalysis
from sales_intelligence.services.analysis import (
    _get_client as analysis_get_client,
)
from sales_intelligence.services.analysis import (
    analyse_streaming,
    extract_json,
    get_prompt,
)
from sales_intelligence.utils.exceptions import AnalysisError

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

VALID_JSON = json.dumps(
    {
        "summary": "Good call.",
        "objections": ["Price too high"],
        "action_items": ["Send proposal"],
        "sentiment": "Positive",
    }
)

INVALID_JSON = "not json at all"

WRONG_SCHEMA_JSON = json.dumps(
    {
        "summary": "Good call.",
        "objections": [],
        "action_items": [],
        "sentiment": "Happy",  # invalid Literal — triggers Pydantic retry
    }
)


def _make_stream(text: str):
    """Build a minimal Anthropic streaming context manager mock."""
    mock_message = MagicMock()
    mock_message.usage.input_tokens = 100
    mock_message.usage.output_tokens = 50

    async def _chunks():
        for char in text:
            yield char

    mock_stream = MagicMock()
    mock_stream.text_stream = _chunks()
    mock_stream.get_final_message = AsyncMock(return_value=mock_message)
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.messages.stream.return_value = mock_stream
    return mock_client


# ---------------------------------------------------------------------------
# Prompt registry — baseline regression
# ---------------------------------------------------------------------------


def test_prompt_registry_v1_baseline():
    """v1 must exist and contain the JSON-only instruction — regression guard."""
    prompt = get_prompt("v1")
    assert "ONLY valid JSON" in prompt["system"]


def test_get_prompt_unknown_raises():
    with pytest.raises(KeyError, match="does not exist"):
        get_prompt("v99")


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------


def test_extract_json_strips_markdown_fence():
    """Markdown code fences must be stripped before JSON parsing."""
    assert extract_json('```json\n{"k": "v"}\n```') == '{"k": "v"}'


# ---------------------------------------------------------------------------
# analyse_streaming — happy path + tokens on queue
# ---------------------------------------------------------------------------


async def test_analyse_streaming_happy_path_and_tokens():
    """Valid JSON → SalesCallAnalysis returned; tokens pushed onto queue."""
    mock_client = _make_stream(VALID_JSON)
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    with patch(
        "sales_intelligence.services.analysis._get_client", return_value=mock_client
    ):
        with patch("sales_intelligence.services.analysis.record_spend"):
            result, _, _ = await analyse_streaming(
                transcript="Test",
                request_id="req-1",
                token_queue=queue,
                anthropic_api_key="sk-test",
            )

    assert isinstance(result, SalesCallAnalysis)
    assert result.sentiment == "Positive"

    # Tokens must have been pushed to the queue during attempt 1
    tokens = []
    while not queue.empty():
        t = queue.get_nowait()
        if t is not None:
            tokens.append(t)
    assert "".join(tokens) == VALID_JSON


# ---------------------------------------------------------------------------
# analyse_streaming — retry on invalid then valid
# ---------------------------------------------------------------------------


async def test_analyse_streaming_retries_and_succeeds():
    """Attempt 1 invalid JSON → attempt 2 valid JSON → success on 2nd call."""

    def _make_stream_for(text):
        msg = MagicMock()
        msg.usage.input_tokens = 100
        msg.usage.output_tokens = 50

        async def _chunks():
            for char in text:
                yield char

        s = MagicMock()
        s.text_stream = _chunks()
        s.get_final_message = AsyncMock(return_value=msg)
        s.__aenter__ = AsyncMock(return_value=s)
        s.__aexit__ = AsyncMock(return_value=False)
        return s

    mock_client = MagicMock()
    mock_client.messages.stream.side_effect = [
        _make_stream_for(WRONG_SCHEMA_JSON),  # attempt 1 — wrong schema
        _make_stream_for(VALID_JSON),  # attempt 2 — correct
    ]

    queue: asyncio.Queue[str | None] = asyncio.Queue()

    with patch(
        "sales_intelligence.services.analysis._get_client", return_value=mock_client
    ):
        with patch("sales_intelligence.services.analysis.record_spend"):
            result, _, _ = await analyse_streaming(
                transcript="Test",
                request_id="req-1",
                token_queue=queue,
                anthropic_api_key="sk-test",
            )

    assert result.sentiment == "Positive"
    assert mock_client.messages.stream.call_count == 2


# ---------------------------------------------------------------------------
# analyse_streaming — all retries exhausted → AnalysisError
# ---------------------------------------------------------------------------


async def test_analyse_streaming_raises_after_max_attempts():
    """Three consecutive invalid responses must raise AnalysisError with attempts=3."""

    def _bad():
        msg = MagicMock()
        msg.usage.input_tokens = 10
        msg.usage.output_tokens = 10

        async def _chunks():
            yield INVALID_JSON

        s = MagicMock()
        s.text_stream = _chunks()
        s.get_final_message = AsyncMock(return_value=msg)
        s.__aenter__ = AsyncMock(return_value=s)
        s.__aexit__ = AsyncMock(return_value=False)
        return s

    mock_client = MagicMock()
    mock_client.messages.stream.side_effect = [_bad(), _bad(), _bad()]

    queue: asyncio.Queue[str | None] = asyncio.Queue()

    with patch(
        "sales_intelligence.services.analysis._get_client", return_value=mock_client
    ):
        with patch("sales_intelligence.services.analysis.record_spend"):
            with pytest.raises(AnalysisError) as exc_info:
                await analyse_streaming(
                    transcript="Test",
                    request_id="req-1",
                    token_queue=queue,
                    anthropic_api_key="sk-test",
                )

    assert exc_info.value.attempts == 3


# ---------------------------------------------------------------------------
# get_client
# ---------------------------------------------------------------------------


def test_get_client_uses_header_key():
    """Header key takes priority over everything."""
    client = analysis_get_client("sk-ant-header")
    assert client.api_key == "sk-ant-header"


def test_get_client_falls_back_to_env(monkeypatch):
    """No header key + no REQUIRE_USER_KEYS → uses env var."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    monkeypatch.delenv("REQUIRE_USER_KEYS", raising=False)
    client = analysis_get_client(None)
    assert client.api_key == "sk-ant-env"


def test_get_client_raises_when_require_user_keys_set(monkeypatch):
    """No header key + REQUIRE_USER_KEYS=true → ValueError."""
    monkeypatch.setenv("REQUIRE_USER_KEYS", "true")
    with pytest.raises(ValueError, match="Anthropic API key is required"):
        analysis_get_client(None)
