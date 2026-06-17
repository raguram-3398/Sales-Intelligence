from __future__ import annotations

import json
import os
from typing import Any

import requests
import streamlit as st
from partial_json_parser import Allow
from partial_json_parser import loads as partial_loads

API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000/api/v1")

st.set_page_config(
    page_title="Sales Call Intelligence",
    layout="wide",
)
st.title("Sales Call Intelligence")
with st.sidebar:
    st.header("API Keys")
    anthropic_key = st.text_input(
        "Anthropic API Key", type="password", placeholder="sk-ant-..."
    )
    openai_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    st.caption("keys are sent as request headers and are not logged or persisted")
if "metadata" not in st.session_state:
    st.session_state.metadata = None
uploaded_file = st.file_uploader(
    "Upload a sales call recording", type=["mp3", "mp4", "wav", "m4a"]
)


def _auth_headers() -> dict[str, str]:
    """Build X-Anthropic-Key / X-OpenAI-Key headers from sidebar inputs; omits keys if blank."""
    headers: dict[str, str] = {}
    if anthropic_key:
        headers["X-Anthropic-Key"] = anthropic_key
    if openai_key:
        headers["X-OpenAI-Key"] = openai_key
    return headers


def _stream_and_render(file) -> dict:
    """
    POST audio file to /analyse endpoint and consume streaming response.

    Stream yields token chunks with incremental JSON-parsed previews via partial_loads, and ends with "__META__:{json}" for final structured output or "__ERROR__:{msg}" on failure.
    Tokens are streamed live into Streamlit with st.write_stream + a live-updating markdown preview derived from partially parsed fields (summary, objections, action items, sentiment). A last_text guard prevents redundant re-renders. On __META__ arrival, streaming stops, UI is cleared, and final structured Markdown is rendered from authoritative metadata.
    Design note: hybrid streaming UX combining sentinel-based completion with best-effort partial JSON rendering for real-time structured feedback, followed by final deterministic UI rebuild from __META__ payload.
    """
    file.seek(0)
    files = {"audio_file": (file.name, file, file.type)}
    metadata: dict[str, Any] = {}
    buffer = ""
    last_text = ""
    stream_area = st.empty()

    with requests.post(
        f"{API_BASE_URL}/analyse",
        files=files,
        headers=_auth_headers(),
        stream=True,
        timeout=300,
    ) as response:
        response.raise_for_status()

        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if not chunk or not isinstance(chunk, str):
                continue
            if chunk.startswith("__META__:"):
                metadata.update(json.loads(chunk[len("__META__:") :].strip()))
                continue
            if chunk.startswith("__ERROR__:"):
                raise RuntimeError(chunk[len("__ERROR__:") :].strip())

            buffer += chunk
            try:
                partial = partial_loads(buffer, Allow.ALL)
                if not isinstance(partial, dict):
                    continue

                # Extract only the readable text values, no keys or structure
                parts = []
                if summary := partial.get("summary"):
                    parts.append(summary)
                if objections := partial.get("objections"):
                    parts.extend(o for o in objections if isinstance(o, str))  # type: ignore[reportOptionalIterable]
                if action_items := partial.get("action_items"):
                    parts.extend(a for a in action_items if isinstance(a, str))  # type: ignore[reportOptionalIterable]
                if sentiment := partial.get("sentiment"):
                    parts.append(f"Sentiment: {sentiment}")

                current_text = "\n\n".join(parts)
                if current_text != last_text:
                    stream_area.markdown(current_text)
                    last_text = current_text
            except Exception:
                pass

    if not metadata:
        raise RuntimeError("Stream ended without __META__ sentinel")

    # Replace progressive text with final structured view
    stream_area.empty()
    analysis = metadata.get("analysis", {})
    low_confidence = metadata.get("low_confidence", False)
    confidence_message = metadata.get("confidence_message")

    if low_confidence and confidence_message:
        st.warning(f"Low confidence: {confidence_message}")
    st.markdown(f"**Summary**\n\n{analysis.get('summary', '')}")
    st.markdown(
        "**Objections**\n\n"
        + "\n".join(f"- {o}" for o in analysis.get("objections", []))
    )
    st.markdown(
        "**Action Items**\n\n"
        + "\n".join(f"- {a}" for a in analysis.get("action_items", []))
    )
    st.markdown(f"**Sentiment**\n\n{analysis.get('sentiment', '')}")

    return metadata


if st.button("Analyse"):
    if uploaded_file is None:
        st.warning("Please upload an audio file.")
        st.stop()
    uploaded_file.seek(0)
    st.subheader("Analysis")
    with st.spinner("Transcribing and analysing audio..."):
        try:
            metadata = _stream_and_render(uploaded_file)
            st.session_state.metadata = metadata
        except RuntimeError as e:
            st.error(f"Analysis failed: {e}")
        except requests.HTTPError as e:
            if e.response is not None:
                if e.response.status_code == 401:
                    st.error(
                        f"API key required: {e.response.json().get('detail', 'Please provide your API keys in the sidebar.')}"
                    )
                else:
                    st.error(
                        f"Request failed {e.response.status_code}: {e.response.text}"
                    )
            else:
                st.error("Request failed: no response received from server.")

metadata = st.session_state.metadata

if metadata:
    st.subheader("Cost Breakdown")
    col1, col2, col3 = st.columns(3)
    col1.metric("Whisper Cost", f"${metadata['whisper_cost']:.6f}")
    col2.metric("Claude Cost", f"${metadata['claude_cost']:.6f}")
    col3.metric("Total Cost", f"${metadata['total_cost']:.6f}")
