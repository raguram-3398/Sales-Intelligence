from __future__ import annotations

import asyncio
import json
import os
import re

from anthropic import AsyncAnthropic
from pydantic import ValidationError

from sales_intelligence.models.schemas import SalesCallAnalysis
from sales_intelligence.services.cost_tracker import calculate_claude_cost, record_spend
from sales_intelligence.utils.exceptions import AnalysisError

_MODEL = "claude-sonnet-4-6"

PROMPT_REGISTRY: dict[str, dict[str, str]] = {
    "v1": {
        "system": (
            "You are an expert sales analyst.\n"
            "Return ONLY valid JSON with no markdown, no code fences, no extra keys.\n"
            "Analyse the provided sales call transcript and return a JSON object with exactly these fields:\n"
            "- summary: a single string containing an executive summary\n"
            "- objections: a list of strings, each string is one objection raised by the customer\n"
            "- action_items: a list of strings, each string is one follow-up action for the sales team\n"
            "- sentiment: exactly one of the strings Positive, Neutral, or Negative\n\n"
            "Example of correct format:\n"
            '{"summary": "The call went well.", "objections": ["Price too high", "Need more time"], '
            '"action_items": ["Send proposal", "Schedule demo"], "sentiment": "Positive"}\n\n'
            "Be concise, accurate and fact-based."
        ),
        "user": "Analyse the following sales call transcript:\n\n{transcript}",
    },
}


def get_prompt(version: str) -> dict[str, str]:
    """Look up a prompt version from PROMPT_REGISTRY; raises KeyError on unknown version."""
    try:
        return PROMPT_REGISTRY[version]
    except KeyError as e:
        raise KeyError(f"Prompt version '{version}' does not exist.") from e


def extract_json(raw: str) -> str:
    """Strip markdown code fences from a Claude response before JSON parsing."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


def _get_client(api_key: str | None) -> AsyncAnthropic:
    """Returns AsyncAnthropic from user key or env fallback"""
    if api_key:
        return AsyncAnthropic(api_key=api_key)
    if os.getenv("REQUIRE_USER_KEYS", "").lower() == "true":
        raise ValueError(
            "An Anthropic API key is required. Please provide one in the sidebar."
        )
    return AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


async def _call_claude(
    client: AsyncAnthropic,
    system: str,
    user_content: str,
    token_queue: asyncio.Queue[str | None] | None = None,
    timeout: float = 60.0,
) -> tuple[str, int, int]:
    """Stream a single Claude call; push tokens onto token_queue if provided, else collect silently.

    # Design note: one function handles both attempt 1 (token_queue supplied → tokens forwarded
    # live to the HTTP client) and retries (token_queue=None → tokens collected but not forwarded).
    # Always streams internally so usage stats are captured identically on every attempt.
    # None sentinel pushed in finally so the queue consumer in _analyse_stream always terminates.
    """
    collected: list[str] = []
    message = None
    try:
        async with asyncio.timeout(timeout):
            async with client.messages.stream(
                model=_MODEL,
                max_tokens=2048,
                system=system,
                messages=[{"role": "user", "content": user_content}],
            ) as stream:
                async for chunk in stream.text_stream:
                    collected.append(chunk)
                    if token_queue is not None:
                        await token_queue.put(chunk)
                message = await stream.get_final_message()
    except TimeoutError as e:
        raise TimeoutError("claude timed out") from e
    finally:
        if token_queue is not None:
            await token_queue.put(None)
    return (
        "".join(collected),
        message.usage.input_tokens if message else 0,
        message.usage.output_tokens if message else 0,
    )


async def analyse_streaming(
    transcript: str,
    request_id: str,
    token_queue: asyncio.Queue[str | None],
    prompt_version: str = "v1",
    max_attempts: int = 3,
    anthropic_api_key: str | None = None,
) -> tuple[SalesCallAnalysis, float, float]:
    """Stream attempt 1 live; validate collected JSON; retry up to max_attempts with error-fed correction prompt.

    # Design note: attempt 1 streams tokens onto token_queue so Streamlit can render fields
    # progressively via partial_json_parser. Retries are silent (token_queue=None) — Streamlit
    # clears and re-renders from the validated __META__ sentinel on completion.
    # Cost is recorded once on success, accumulating tokens across all attempts.
    """
    client = _get_client(anthropic_api_key)
    prompt = get_prompt(prompt_version)
    raw = ""
    last_error = ""
    total_input_tokens = 0
    total_output_tokens = 0
    for attempt in range(1, max_attempts + 1):
        if attempt == 1:
            user_content = prompt["user"].format(transcript=transcript)
            raw, in_tok, out_tok = await _call_claude(
                client=client,
                system=prompt["system"],
                user_content=user_content,
                token_queue=token_queue,
            )
        else:
            user_content = (
                "You are correcting a structured analysis output.\n\n"
                "TASK:\nReturn valid JSON matching the required schema.\n\n"
                f"ORIGINAL TRANSCRIPT:\n{transcript}\n\n"
                f"YOUR PREVIOUS RESPONSE:\n{raw}\n\n"
                f"ERROR:\n{last_error}\n\n"
                "RULES:\n"
                "- Output ONLY valid JSON\n"
                "- Must match schema exactly\n"
                "- sentiment must be: Positive | Neutral | Negative\n"
                "- No markdown, no code fences"
            )
            raw, in_tok, out_tok = await _call_claude(
                client=client,
                system=prompt["system"],
                user_content=user_content,
                token_queue=None,
            )
        total_input_tokens += in_tok
        total_output_tokens += out_tok
        cleaned = extract_json(raw)
        try:
            result = SalesCallAnalysis.model_validate(json.loads(cleaned))
            input_cost, output_cost = calculate_claude_cost(
                total_input_tokens, total_output_tokens
            )
            record_spend(input_cost + output_cost, request_id=request_id)
            return result, input_cost, output_cost
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)
    raise AnalysisError(
        message=f"Analysis failed after {max_attempts} attempts",
        attempts=max_attempts,
        last_error=last_error,
    )
