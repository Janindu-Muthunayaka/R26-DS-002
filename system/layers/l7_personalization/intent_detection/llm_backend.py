# intent_detection/llm_backend.py
#
# Dedicated OpenAI backend for Intent Extraction.
# Uses OpenAI Chat Completions API with the key and model defined in .env.

import os
import re
import json
import requests
from dotenv import load_dotenv

# Load .env if present (searching up directory tree to system/ or repo root)
_cur = os.path.abspath(__file__)
for _ in range(5):
    _cur = os.path.dirname(_cur)
    _env_file = os.path.join(_cur, ".env")
    if os.path.exists(_env_file):
        load_dotenv(_env_file)
        break

TEMPERATURE = 0.0
MAX_TOKENS = 100


def get_active_backend():
    """Returns the active backend name."""
    return "openai"


def chat(system_prompt, user_prompt):
    """
    Sends one system+user prompt pair to OpenAI API and returns the raw text reply.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from environment and .env")

    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=15.0
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def normalize_label(label):
    """
    Puts an intent label into one canonical form so that writing style never
    changes its meaning: upper case, no surrounding spaces, and spaces or
    hyphens turned into underscores.
        "Stop", "stop ", "STOP"            -> "STOP"
        "step by step", "Step-By-Step"     -> "STEP_BY_STEP"
    """
    return re.sub(r"[\s\-]+", "_", str(label or "").strip().upper()).strip("_")


def labels_match(predicted, expected):
    """True if two intent labels are the same after normalize_label()."""
    return normalize_label(predicted) == normalize_label(expected)


def parse_json_reply(raw):
    """
    Strips markdown fences and parses JSON. Returns {} if the reply isn't
    valid JSON, so one malformed response can't crash the whole pipeline.
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


if __name__ == "__main__":
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("=" * 60)
    print("OPENAI INTENT BACKEND TEST")
    print("=" * 60)
    print(f"Backend  : {get_active_backend()}")
    print(f"Model    : {os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini')}")
    print(f"API Key  : {'Set (' + os.getenv('OPENAI_API_KEY')[:7] + '...)' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
    print("-" * 60)

    reply = chat(
        "You reply with only a JSON object, nothing else.",
        'Return {"status": "ok"}',
    )
    print("Raw reply  :", reply)
    print("Parsed     :", parse_json_reply(reply))
