# pipeline.py
try:
    from .adapters import parse_voice_input, parse_ocr_input, parse_request_context
    from .vectorstore import build_vectorstore, add_current_ocr_chunk
    from .retrieve import retrieve_context
    from .llm_generate import generate_answer
    from .prompt import resolve_max_words, resolve_age_profile
    from .history import get_history, add_turn
    from .history_prompt import format_history_for_prompt
    from .rewrite import rewrite_query
    from .websearch import web_search, is_refusal, WEB_PROMPT_NOTE
    from .query_si import get_sinhala_query
except (ImportError, ValueError):
    from adapters import parse_voice_input, parse_ocr_input, parse_request_context
    from vectorstore import build_vectorstore, add_current_ocr_chunk
    from retrieve import retrieve_context
    from llm_generate import generate_answer
    from prompt import resolve_max_words, resolve_age_profile
    from history import get_history, add_turn
    from history_prompt import format_history_for_prompt
    from rewrite import rewrite_query
    from websearch import web_search, is_refusal, WEB_PROMPT_NOTE
    from query_si import get_sinhala_query

_DEFAULT_VECTORSTORE = None


def get_default_vectorstore():
    global _DEFAULT_VECTORSTORE
    if _DEFAULT_VECTORSTORE is None:
        try:
            _DEFAULT_VECTORSTORE = build_vectorstore()
        except Exception as e:
            print(f"[pipeline] build_vectorstore failed ({e})")
            _DEFAULT_VECTORSTORE = None
    return _DEFAULT_VECTORSTORE


def run_pipeline(vectorstore=None, raw_ocr_input: dict = None, raw_voice_input: dict = None,
                 raw_request_context: dict = None, skip_ocr: bool = False,
                 use_history: bool = True,
                 use_web_search: bool = True,
                 history_in_prompt: bool = True,
                 retrieval_mode: str = "hybrid"):
    if vectorstore is None:
        vectorstore = get_default_vectorstore()

    raw_voice_input = raw_voice_input or {}
    raw_ocr_input = raw_ocr_input or {}
    voice = parse_voice_input(raw_voice_input)
    ocr = parse_ocr_input(raw_ocr_input)
    ctx = parse_request_context(raw_request_context)

    # If vectorstore is unavailable, gracefully fall back to bm25
    if vectorstore is None and retrieval_mode in ("hybrid", "dense"):
        retrieval_mode = "bm25"
    elif retrieval_mode not in ("hybrid", "dense", "bm25"):
        retrieval_mode = "hybrid"

    if voice["route"] != "GENERATE":
        return {
            "ok": True,
            "intent": voice["intent"],
            "answer_si": f"'{voice['route']}' route සඳහා RAG පිළිතුරු ජනනය නොකරයි.",
            "retrieved_sources": [],
            "contexts": [],
            "speakable_text": None,
            "perplexity": None,
            "standalone_query": None,
            "web_searched": False,
            "web_used": False,
            "chat_history_text": "",
            "retrieval_mode": retrieval_mode,
            "sparse_query": None,
        }

    effective_age_group = ctx["age_group"] or voice["personalization_flags"].get("age_group")
    age_profile = resolve_age_profile(effective_age_group)

    session_id = voice.get("session_id")

    original_query = voice["query_text"]
    caller_history = raw_voice_input.get("chat_history")
    if raw_voice_input.get("history_override") and isinstance(caller_history, list):
        # Re-generating an old turn: exactly the turns before it, nothing later.
        history = caller_history if use_history else []
    else:
        history = get_history(session_id) if use_history else []
    standalone_query = rewrite_query(original_query, history)

    chat_history_text = (
        format_history_for_prompt(history) if (use_history and history_in_prompt) else ""
    )

    if not skip_ocr and vectorstore is not None:
        add_current_ocr_chunk(vectorstore, ocr)

    # Sinhala version of the query for BM25 (not needed for dense-only).
    sparse_query = None
    if retrieval_mode in ("hybrid", "bm25"):
        sparse_query = get_sinhala_query(standalone_query, voice.get("original_text"))

    docs = retrieve_context(
        vectorstore,
        query_text=standalone_query,
        intent=voice["intent"],
        retrieved_chunk_id=voice.get("retrieved_chunk_id"),
        use_ocr=not skip_ocr,
        sparse_query=sparse_query,
        retrieval_mode=retrieval_mode,
    )

    max_words = resolve_max_words(
        voice["style_class"], voice["personalization_flags"], age_profile["word_cap"]
    )

    generation_ok = True
    try:
        answer, perplexity = generate_answer(
            query_text=standalone_query,
            intent=voice["intent"],
            style_class=voice["style_class"],
            prompt_modifier=voice["prompt_modifier"],
            age_instruction=age_profile["instruction"],
            max_words=max_words,
            evidence_docs=docs,
            chat_history=chat_history_text,
        )
    except Exception as e:
        print(f"[pipeline] generate_answer failed: {e}")
        generation_ok = False
        answer = "පිළිතුර ලබාගැනීමට නොහැකි විය. කරුණාකර නැවත උත්සාහ කරන්න."
        perplexity = None

    # Web search fallback. Only when the articles could not answer
    web_searched = False
    web_used = False
    if use_web_search and generation_ok and is_refusal(answer):
        web_searched = True
        web_docs = web_search(standalone_query)
        if web_docs:
            try:
                web_answer, web_perplexity = generate_answer(
                    query_text=standalone_query,
                    intent=voice["intent"],
                    style_class=voice["style_class"],
                    prompt_modifier=(voice["prompt_modifier"] or "") + " " + WEB_PROMPT_NOTE,
                    age_instruction=age_profile["instruction"],
                    max_words=max_words,
                    evidence_docs=docs + web_docs,
                    chat_history=chat_history_text,
                )
                if not is_refusal(web_answer):
                    answer, perplexity = web_answer, web_perplexity
                    docs = docs + web_docs
                    web_used = True
            except Exception:
                pass  # keep the original refusal answer

    if use_history and generation_ok and not raw_voice_input.get("no_history_write"):
        add_turn(original_query, standalone_query, answer, session_id)

    return {
        "ok": generation_ok and not is_refusal(answer),
        "intent": voice["intent"],
        "answer_si": answer,
        "retrieved_sources": [getattr(d, "metadata", {}) for d in docs],
        "contexts": [getattr(d, "page_content", "") for d in docs],
        "speakable_text": answer,
        "perplexity": perplexity,
        "standalone_query": standalone_query,
        "web_searched": web_searched,
        "web_used": web_used,
        "chat_history_text": chat_history_text,
        "retrieval_mode": retrieval_mode,
        "sparse_query": sparse_query,
    }
