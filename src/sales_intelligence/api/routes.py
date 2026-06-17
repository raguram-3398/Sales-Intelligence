from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from sales_intelligence.models.schemas import StreamMetadata
from sales_intelligence.services.analysis import analyse_streaming
from sales_intelligence.services.cost_tracker import (
    calculate_whisper_cost,
    record_spend,
)
from sales_intelligence.services.pii_scrubber import check_injection, scrub_pii
from sales_intelligence.services.transcription import check_confidence, transcribe_audio
from sales_intelligence.utils.exceptions import AnalysisError
from sales_intelligence.utils.logger import get_logger, log_request

logger = get_logger(__name__)
limiter = Limiter(key_func=get_remote_address)
router = APIRouter()

_MAX_FILE_SIZE = 25 * 1024 * 1024
_ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".mp3", ".mp4", ".wav", ".m4a"})
_ALLOWED_MIME_TYPES: frozenset[str] = frozenset(
    {
        "audio/mpeg",
        "audio/mp3",
        "audio/wav",
        "audio/x-wav",
        "audio/mp4",
        "audio/x-m4a",
        "audio/m4a",
    }
)


async def rate_limit_handler(request: Request, exc: Exception):
    """Return rate-limit or server error responses."""
    if isinstance(exc, RateLimitExceeded):
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests"},
        )

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


def _validate_audio_file(audio_file: UploadFile, audio_bytes: bytes) -> None:
    """Reject unsupported file extensions, MIME types, and files over 25 MB before any external call."""
    suffix = Path(audio_file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported audio format. Allowed: {', '.join(_ALLOWED_EXTENSIONS)}",
        )
    if audio_file.content_type not in _ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=422,
            detail="unsupported MIME type.",
        )
    if len(audio_bytes) > _MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File exceeds 25MB limit",
        )


def trim_transcript(
    transcript: str, max_tokens: int = 150000, char_per_token: float = 4.0
) -> tuple[str, bool]:
    """Trim transcript to fit within the 150K token context budget; returns (text, was_trimmed)."""
    estimated_tokens = len(transcript) / char_per_token
    if estimated_tokens <= max_tokens:
        return transcript, False
    max_char = int(max_tokens * char_per_token)
    trimmed = transcript[:max_char].rstrip()
    trimmed += "\n\n [Transcript trimmed - original exceeded context budget]"
    return trimmed, True


def _prepare_transcript(raw_transcript: str, request_id: str) -> str:
    """Run context trim → injection check → PII scrub on the raw transcript."""
    trimmed, was_trimmed = trim_transcript(raw_transcript)
    if was_trimmed:
        log_request(logger, request_id, "transcript_trimmed")
    guard = check_injection(trimmed)
    if not guard.is_safe:
        raise HTTPException(status_code=400, detail=guard.reason)
    return scrub_pii(trimmed)


async def _analyse_stream(
    transcript: str,
    request_id: str,
    whisper_cost: float,
    low_confidence: bool,
    confidence_message: str | None,
    anthropic_api_key: str | None,
) -> AsyncGenerator[str, None]:
    """Orchestrate the full analysis pipeline as an async generator feeding a StreamingResponse.

    # Design note: asyncio.Queue decouples the producer (analyse_streaming background task)
    # from this consumer generator. Tokens flow to the HTTP client as they arrive.
    # After the queue drains, the validated result is awaited and emitted as a __META__ sentinel
    # so Streamlit receives structured data and raw tokens in a single HTTP connection.
    """
    token_queue: asyncio.Queue[str | None] = asyncio.Queue()
    analysis_task = asyncio.create_task(
        analyse_streaming(
            transcript=transcript,
            request_id=request_id,
            token_queue=token_queue,
            anthropic_api_key=anthropic_api_key,
        )
    )
    while True:
        token = await token_queue.get()
        if token is None:
            break
        yield token
    try:
        analysis, claude_input_cost, claude_output_cost = await analysis_task
    except AnalysisError as e:
        yield f"__ERROR__:{str(e)}\n"
        return
    except TimeoutError as e:
        yield f"__ERROR__: Claude Timed out - {str(e)}\n"
        return
    claude_cost = claude_input_cost + claude_output_cost
    total_cost = whisper_cost + claude_cost
    metadata = StreamMetadata(
        request_id=request_id,
        transcript=transcript,
        analysis=analysis,
        low_confidence=low_confidence,
        confidence_message=confidence_message,
        whisper_cost=whisper_cost,
        claude_cost=claude_cost,
        total_cost=total_cost,
    )
    yield f"__META__:{metadata.model_dump_json()}\n"
    log_request(
        logger, request_id, "analyse_request_complete", total_cost=round(total_cost, 6)
    )


@router.post("/analyse")
@limiter.limit("10/minute")
async def analyze_call(
    request: Request,
    audio_file: UploadFile,
    x_anthropic_key: str | None = Header(default=None),
    x_openai_key: str | None = Header(default=None),
) -> StreamingResponse:
    """Accept audio upload, run the full pipeline, and return a streaming SSE response.

    # Design note: request: Request is required by slowapi's @limiter.limit decorator
    # to extract client IP via get_remote_address — it is not used directly in the function body.
    # Whisper cost is recorded here immediately after transcription; Claude cost is recorded
    # inside analyse_streaming on validation success. Both feed the same daily accumulator.
    """
    request_id = str(uuid.uuid4())
    log_request(
        logger, request_id, "analyse_request_received", filename=audio_file.filename
    )
    audio_bytes = await audio_file.read()
    _validate_audio_file(audio_file, audio_bytes)
    try:
        transcription = await transcribe_audio(
            audio_bytes=audio_bytes,
            filename=audio_file.filename or "audio",
            request_id=request_id,
            openai_api_key=x_openai_key,
        )
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e)) from e
    whisper_cost = calculate_whisper_cost(transcription.duration_seconds or 0.0)
    if whisper_cost:
        record_spend(whisper_cost, request_id=request_id)
    confidence = check_confidence(transcription.transcript, request_id=request_id)
    trimmed = _prepare_transcript(transcription.transcript, request_id)
    log_request(logger, request_id, "transcription_done_starting_analysis")
    return StreamingResponse(
        _analyse_stream(
            transcript=trimmed,
            request_id=request_id,
            whisper_cost=whisper_cost,
            low_confidence=confidence.is_low_confidence,
            confidence_message=confidence.flag_message,
            anthropic_api_key=x_anthropic_key,
        ),
        media_type="text/event-stream",
    )
