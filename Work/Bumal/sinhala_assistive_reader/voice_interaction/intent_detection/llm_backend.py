# intent_detection/llm_backend.py
#
# One place that decides HOW the Llama 3.2 1B model is run.
#
# TWO BACKENDS, same function signature:
#
#   "llamacpp"  — loads a .gguf file directly inside this Python process
#                 using llama-cpp-python. No separate application, no
#                 background server, no HTTP. This is the deployable option:
#                 the model file ships inside the project folder.
#
#   "ollama"    — the original approach: POSTs to the Ollama server running
#                 at localhost:11434. Kept working so the Approach 1 vs
#                 Approach 2 evaluation can still be reproduced, and as a
#                 fallback if the .gguf file is missing.
#
# WHICH ONE RUNS: controlled by LLM_BACKEND below, or by setting the
# environment variable LLM_BACKEND=ollama / LLM_BACKEND=llamacpp.
# Default is "auto": use llama.cpp if the model file is present and the
# library is installed, otherwise fall back to Ollama automatically.
#
# The model is loaded ONCE, lazily, on first use — not at import time — so
# scripts that never call the LLM don't pay the loading cost.

import os
import re
import json

# ─── Configuration ──────────────────────────────────────────────────────
LLM_BACKEND = os.environ.get("LLM_BACKEND", "auto")   # "auto" | "llamacpp" | "ollama"

# Where the .gguf file lives: intent_detection/model/ (see MODEL_DIR below)
# Where the .gguf file lives: intent_detection/model/ (a subfolder right
# next to this file, since llm_backend.py itself lives in intent_detection/)
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model"
)
MODEL_FILENAME = os.environ.get(
    "LLM_GGUF_FILE", "llama-3.2-1b-instruct-q4_k_m.gguf"
)
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# Generation settings — temperature 0 for deterministic intent classification
TEMPERATURE = 0.0
MAX_TOKENS = 100
N_CTX = 2048

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:1b"

_llm = None            # cached llama_cpp.Llama instance
_active_backend = None  # resolved backend name, for reporting


def _resolve_backend():
    """Decides which backend to use, once."""
    if LLM_BACKEND == "ollama":
        return "ollama"
    if LLM_BACKEND == "llamacpp":
        return "llamacpp"

    # auto
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        print("  llama-cpp-python not installed — falling back to Ollama.")
        return "ollama"

    if not os.path.exists(MODEL_PATH):
        print(f"  GGUF model not found at {MODEL_PATH} — falling back to Ollama.")
        return "ollama"

    return "llamacpp"


def _get_llm():
    """Loads the GGUF model once and caches it."""
    global _llm
    if _llm is None:
        from llama_cpp import Llama
        print(f"Loading Llama 3.2 1B from {MODEL_FILENAME} ...")
        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            verbose=False,       # hide the C++ loader logs
        )
        print("✅ Llama model loaded (local, no Ollama needed)")
    return _llm


def get_active_backend():
    """Returns which backend is actually in use ('llamacpp' or 'ollama')."""
    global _active_backend
    if _active_backend is None:
        _active_backend = _resolve_backend()
    return _active_backend


def _chat_llamacpp(system_prompt, user_prompt):
    llm = _get_llm()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return response["choices"][0]["message"]["content"].strip()


def _chat_ollama(system_prompt, user_prompt):
    import requests
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
        },
    )
    return response.json()["message"]["content"].strip()


def chat(system_prompt, user_prompt):
    """
    Sends one system+user prompt pair to Llama 3.2 1B and returns the raw
    text reply. Identical behaviour regardless of backend.
    """
    if get_active_backend() == "llamacpp":
        return _chat_llamacpp(system_prompt, user_prompt)
    return _chat_ollama(system_prompt, user_prompt)


def normalize_label(label):
    """
    Puts an intent label into one canonical form so that writing style never
    changes its meaning: upper case, no surrounding spaces, and spaces or
    hyphens turned into underscores.
        "Stop", "stop ", "STOP"            -> "STOP"
        "step by step", "Step-By-Step"     -> "STEP_BY_STEP"
    Used for every model output and every comparison against an expected
    label, in all approaches and evaluation scripts.
    """
    return re.sub(r"[\s\-]+", "_", str(label or "").strip().upper()).strip("_")


def labels_match(predicted, expected):
    """True if two intent labels are the same after normalize_label()."""
    return normalize_label(predicted) == normalize_label(expected)


def parse_json_reply(raw):
    """
    Both backends occasionally wrap JSON in markdown fences. Strips them and
    parses. Returns {} if the reply isn't valid JSON, so one malformed
    response can't crash the whole pipeline.
    """
    if not raw:
        return {}

    text = raw.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()

    # Some small models emit a leading sentence before the JSON — take the
    # outermost {...} block if a bare parse fails.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return {}


# ─── Quick standalone test ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("LLM BACKEND TEST")
    print("=" * 60)
    print(f"Configured : {LLM_BACKEND}")
    print(f"Model path : {MODEL_PATH}")
    print(f"File exists: {os.path.exists(MODEL_PATH)}")
    print(f"Resolved   : {get_active_backend()}")
    print("-" * 60)

    reply = chat(
        "You reply with only a JSON object, nothing else.",
        'Return {"status": "ok"}',
    )
    print("Raw reply  :", reply)
    print("Parsed     :", parse_json_reply(reply))
