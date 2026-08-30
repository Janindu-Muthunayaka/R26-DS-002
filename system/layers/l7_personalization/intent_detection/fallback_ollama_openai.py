import os
import requests
import json
from dotenv import load_dotenv

# Try to load .env from the system root
sys_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
load_dotenv(os.path.join(sys_root, '.env'))

_OLLAMA_AVAILABLE = False

# Ollama is explicitly disabled per user request
# Check at startup (module load) is commented out
# try:
#     print("[Intent Detection] Checking Ollama availability on startup...")
#     # A simple GET to the root endpoint checks if the Ollama daemon is running
#     res = requests.get("http://localhost:11434/", timeout=2)
#     if res.status_code == 200:
#         print("[Intent Detection] Ollama is running and available.")
#     else:
#         _OLLAMA_AVAILABLE = False
#         print("[Intent Detection] Ollama returned non-200. Will use OpenAI fallback.")
# except requests.exceptions.RequestException:
#     print("[Intent Detection] Ollama unreachable at startup. Will use OpenAI API fallback permanently.")
#     _OLLAMA_AVAILABLE = False

def extract_intent_fallback(system_prompt: str, user_prompt: str) -> str:
    global _OLLAMA_AVAILABLE
    
    # Try Ollama first if we think it's available
    if _OLLAMA_AVAILABLE:
        try:
            response = requests.post(
                "http://localhost:11434/api/chat",
            json={
                "model": "llama3.2:1b",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 100
                }
            },
            timeout=15.0  # Allow enough time for local generation
        )
            if response.status_code == 200:
                return response.json()["message"]["content"].strip()
        except requests.exceptions.RequestException:
            # If it was available at startup but failed now, we can still fallback
            print("[Intent Detection] Ollama failed during request, falling back to OpenAI API...")
            _OLLAMA_AVAILABLE = False

    # Fallback to OpenAI API
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise Exception("Ollama is not reachable and OPENAI_API_KEY is missing from .env")
        
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        "temperature": 0,
        "max_tokens": 100
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=10)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()
