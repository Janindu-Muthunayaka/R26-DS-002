# app.py
# Run with: streamlit run app.py
#
# This is a thin UI shim around your existing pipeline. It doesn't change any
# of your logic — it just collects the same fields that live in contracts.py
# (SAMPLE_OCR_INPUT, SAMPLE_VOICE_INPUT, SAMPLE_REQUEST_CONTEXT) via form
# widgets, builds the same dicts, and calls run_pipeline() exactly like
# main.py does.

# IMPORTANT: must run before importing pipeline/generate, since generate.py
# builds the ChatOpenAI client at import time (module-level `llm = ...`) and
# reads OPENAI_API_KEY from the environment. main.py does this too, via its
# own load_dotenv() call — app.py needs it independently since Streamlit
# runs this file directly, not through main.py.
from dotenv import load_dotenv
load_dotenv()

import json
import os
import uuid
import streamlit as st

from vectorstore import build_vectorstore
from pipeline import run_pipeline
from contracts import SAMPLE_OCR_INPUT, SAMPLE_VOICE_INPUT, SAMPLE_REQUEST_CONTEXT
from history import get_history, clear_history, history_path

st.set_page_config(page_title="Advanced Document Retrieval and Information Generation — RAG pipeline for Sinhala Assive Reading", layout="wide")


@st.cache_resource(show_spinner="Building / connecting vector store...")
def get_vectorstore():
    # build_vectorstore() is already idempotent (skips re-index if the
    # collection is populated), and st.cache_resource means this only
    # runs once per server process regardless of how many times the
    # user clicks "Run".
    return build_vectorstore()


# Every browser tab gets its own session id (st.session_state is per tab).
def _new_session_id() -> str:
    return uuid.uuid4().hex[:8]


def _start_new_session():
    st.session_state["session_id"] = _new_session_id()


if "session_id" not in st.session_state:
    st.session_state["session_id"] = _new_session_id()


st.title("Advanced Document Retrieval and Information Generation — RAG pipeline for Sinhala Assive Reading")
st.caption("Fill in the same fields your OCR/Voice modules would normally produce, then run the pipeline.")

col_ocr, col_voice, col_ctx = st.columns(3)

# --- OCR input -------------------------------------------------------
with col_ocr:
    st.subheader("OCR input")
    corrected_text = st.text_area(
        "corrected_text",
        value=SAMPLE_OCR_INPUT["corrected_text"],
        height=200,
    )

# --- Voice input -------------------------------------------------------
with col_voice:
    st.subheader("Voice input")
    route = st.selectbox("route", ["GENERATE", "OTHER"], index=0)
    intent = st.text_input("intent", value=SAMPLE_VOICE_INPUT["intent"])
    english_translation = st.text_area(
        "english_translation (query)",
        value=SAMPLE_VOICE_INPUT["english_translation"],
        height=100,
    )
    # NEW: optional. If Component 4 ever sends the original Sinhala question,
    # BM25 uses it directly. Empty = the pipeline translates the English query.
    original_text = st.text_input(
        "original_text (Sinhala query, optional)", value=""
    )
    style_class = st.selectbox(
        "style_class",
        ["Simple", "Moderate", "Detailed", "StepByStep"],
        index=3,
    )
    prompt_modifier = st.text_input(
        "prompt_modifier", value=SAMPLE_VOICE_INPUT["prompt_modifier"]
    )
    detail_level = st.selectbox(
        "personalization_flags.detail_level",
        ["brief", "moderate", "detailed", "structured", "full"],
        index=3,
    )
    retrieved_chunk_id = st.text_input(
        "retrieved_chunk_id (optional)",
        value=SAMPLE_VOICE_INPUT.get("retrieved_chunk_id") or "",
    )

# --- Request context -----------------------------------------------
with col_ctx:
    st.subheader("Request context")
    age_group = st.selectbox(
        "age_group",
        ["(none)", "10-13", "14-16", "17-18", "18+"],
        index=1,
    )
    use_web = st.checkbox(
        "Web search fallback (when articles can't answer)", value=True
    )
    # switch for the "history in the LLM prompt" feature (A/B testing)
    history_in_prompt = st.checkbox(
        "Send chat history to the LLM prompt", value=True
    )
    # NEW: switch for the retrieval method (A/B testing)
    retrieval_mode = st.selectbox(
        "retrieval_mode",
        ["hybrid", "dense", "bm25"],
        index=0,
        help="hybrid = dense + BM25 merged with RRF. dense = embeddings only "
             "(the old behaviour). bm25 = keyword search only.",
    )

    st.text_input(
        "session_id",
        key="session_id",
        help="Same id = same chat history. Change it (or click New session) to start a separate conversation.",
    )
    st.button("New session", on_click=_start_new_session)

session_id = (st.session_state.get("session_id") or "").strip()

st.divider()

col_run, col_clear, _ = st.columns([1, 1, 6])
run_clicked = col_run.button("Run pipeline", type="primary")
if col_clear.button("Clear chat history"):
    clear_history(session_id)            # only this session
    st.info(f"Chat history cleared for session '{session_id}'.")

if run_clicked:
    ocr_input = {"corrected_text": corrected_text, "tokens": []}

    voice_input = {
        "route": route,
        "intent": intent,
        "english_translation": english_translation,
        "original_text": original_text or None,                  # NEW
        "style_class": style_class,
        "prompt_modifier": prompt_modifier,
        "personalization_flags": {"detail_level": detail_level},
        "retrieved_chunk_id": retrieved_chunk_id or None,
        "correction_applied": None,
        "session_id": session_id or None,
    }

    request_context = {} if age_group == "(none)" else {"age_group": age_group}

    with st.spinner("Retrieving evidence and generating answer..."):
        vs = get_vectorstore()
        result = run_pipeline(
            vs, ocr_input, voice_input,
            raw_request_context=request_context,
            use_history=True,
            use_web_search=use_web,
            history_in_prompt=history_in_prompt,
            retrieval_mode=retrieval_mode,                       # NEW
        )

    st.subheader("Answer (Sinhala)")
    st.write(result["answer_si"])
    if result.get("web_used"):
        st.caption("🌐 This answer is based on web search results.")

    st.subheader("Details")
    st.json(
        {
            "session_id": session_id or None,
            "intent": result["intent"],
            "standalone_query": result.get("standalone_query"),
            "retrieval_mode": result.get("retrieval_mode"),                  # NEW
            "sparse_query (Sinhala, for BM25)": result.get("sparse_query"),  # NEW
            "history_in_prompt": history_in_prompt,
            "history_chars_in_prompt": len(result.get("chat_history_text") or ""),
            "web_searched": result.get("web_searched"),
            "web_used": result.get("web_used"),
            "perplexity": result["perplexity"],
            "num_sources": len(result["retrieved_sources"]),
        }
    )

    # shows exactly what previous-conversation text went into the prompt
    with st.expander("Chat history sent to the prompt"):
        st.text(result.get("chat_history_text") or "(none)")

    # Article chunks now also show rrf_score / dense_rank / bm25_rank
    with st.expander("Retrieved source metadata"):
        st.json(result["retrieved_sources"])

    with st.expander("Raw inputs sent to run_pipeline()"):
        st.code(
            json.dumps(
                {"ocr_input": ocr_input, "voice_input": voice_input, "request_context": request_context},
                ensure_ascii=False,
                indent=2,
            ),
            language="json",
        )

st.divider()
turns = get_history(session_id)           # this session only
st.subheader(f"Chat history ({len(turns)} turns kept) — session '{session_id or 'default'}'")
st.caption(f"Saved to: {os.path.relpath(history_path(session_id))}")
for i, t in enumerate(turns, 1):
    with st.expander(f"Turn {i}: {t['question']}"):
        st.write("Standalone query:", t["standalone_question"])
        st.write(t["answer"])
