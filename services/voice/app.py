"""
svc-voice — Component 4: Adaptive Conversational Personalization & Voice Interaction.
Based on the end-to-end specification in mem/end_to_end_working_process.md
Student: Sathsara S.P.Y.B — IT22004468 | R26-DS-002

Runs as an independent FastAPI microservice on port 8101:
    python services/voice/app.py --port 8101

Architecture (4-Stage Pipeline):
  Stage 1: Voice Input / STT Transcribed Text
  Stage 2: Translation (NLLB-200 / OpenAI fallback) + Intent Detection (Trained Classifier + LLM fallback)
  Stage 3: Personalization & Online Learning:
    - 3.0: System Command Interception (Navigation bypasses style model)
    - 3.1: Style Correction Detection (Relabels past turn & learns)
    - 3.2: Interaction Logging (TinyDB)
    - 3.3: 4-Tier Style Decision (Intent -> Grounded Flags -> Keywords -> River Online Model)
    - 3.4: Evidence-Only Learning (Self-training prevention)
  Stage 4: Prompt Construction for Component 3 (RAG)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

# Suppress harmless sklearn unpickle version warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ---------------------------------------------------------------------------
# PATH SETUP: make Bumal's code and System utilities importable
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_VOICE_ROOT = _REPO / 'Work' / 'Bumal' / 'sinhala_assistive_reader' / 'voice_interaction'
_SYSTEM = _REPO / 'system'

if str(_VOICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_VOICE_ROOT))
if str(_SYSTEM) not in sys.path:
    sys.path.insert(0, str(_SYSTEM))

# Load environment variables (.env)
from core.env import load as _load_env
_load_env()

# Import Bumal's Personalization Modules
from personalization.system_commands import detect_system_command, get_command_action
from personalization.diagnostic import (
    style_from_intent, style_from_flags, style_from_keywords, detect_correction_signal
)
from personalization.style_model import predict_style, learn_style, get_user_summary, _users, _save_all_users
from personalization.logger import (
    log_interaction, get_last_interaction,
    update_last_interaction_style, update_interaction_style_by_timestamp,
    interaction_logs, Query
)


def get_user_history(user_id: str, limit: int = 10) -> list:
    """Returns recent interaction records for a user from TinyDB."""
    User = Query()
    records = interaction_logs.search(User.user_id == user_id)
    return records[-limit:] if records else []

STYLE_PROMPT_MODIFIERS = {
    "Simple": "Use very simple everyday words, short sentences, avoid technical terms.",
    "Detailed": "Provide a thorough, in-depth explanation with full context, reasoning, and supporting detail. Do not shorten or oversimplify.",
    "StepByStep": "Break the explanation into clear numbered steps, one action or idea per step.",
}

# ---------------------------------------------------------------------------
# LAZY-LOAD HEAVY MODELS AT STARTUP
# ---------------------------------------------------------------------------
_nllb_model = None
_nllb_tokenizer = None
_classifier = None
_vectorizer = None


def _load_nllb():
    """Load NLLB translation model from local cache or prepare fallback."""
    global _nllb_model, _nllb_tokenizer
    if _nllb_model is not None:
        return

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    MODEL_NAME = "facebook/nllb-200-distilled-600M"
    try:
        _nllb_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        _nllb_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, local_files_only=True)
        print("[voice] [OK] NLLB-200 model loaded from local cache")
    except Exception:
        _nllb_model = None
        _nllb_tokenizer = None
        print("[voice] NLLB not in local cache; will use high-speed OpenAI GPT translation")


def _load_classifier():
    """Load the trained TF-IDF+LinearSVC intent classifier."""
    global _classifier, _vectorizer
    if _classifier is not None:
        return

    import joblib
    bundle_path = _VOICE_ROOT / 'intent_detection' / 'model' / 'intent_classifier_bundle.joblib'
    if not bundle_path.exists():
        print(f"[voice] ⚠ Trained classifier not found at {bundle_path}")
        return

    print("[voice] Loading trained intent classifier (TF-IDF + LinearSVC)...")
    bundle = joblib.load(str(bundle_path))
    _classifier = bundle["model"]
    _vectorizer = bundle["vectorizer"]
    print("[voice] [OK] Intent Classifier loaded successfully")


# ---------------------------------------------------------------------------
# STAGE 2: TRANSLATION & INTENT EXTRACTION
# ---------------------------------------------------------------------------
def translate_to_english(sinhala_text: str) -> tuple[str, float]:
    """Translate Sinhala to English using NLLB-200 or OpenAI GPT fallback."""
    if not (sinhala_text or '').strip():
        return '', 0.0
    t0 = time.time()
    try:
        _load_nllb()
        if _nllb_model is not None and _nllb_tokenizer is not None:
            inputs = _nllb_tokenizer(
                sinhala_text, return_tensors="pt",
                padding=True, truncation=True, max_length=200
            )
            translated = _nllb_model.generate(
                **inputs,
                forced_bos_token_id=_nllb_tokenizer.convert_tokens_to_ids("eng_Latn"),
                max_length=200
            )
            res = _nllb_tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
            return res, round(time.time() - t0, 3)
    except Exception as e:
        print(f"[voice] NLLB translation failed ({e}), trying OpenAI fallback...")

    try:
        from core import llm
        if llm.available()[0]:
            res, _ = llm.chat([
                {"role": "system", "content": "Translate the following Sinhala user voice command into concise English. Return ONLY the English translation with no extra text."},
                {"role": "user", "content": sinhala_text}
            ], temperature=0.0)
            if res and res.strip():
                return res.strip(), round(time.time() - t0, 3)
    except Exception:
        pass
    return sinhala_text, round(time.time() - t0, 3)


CONFIDENCE_THRESHOLD = 0.60


def _classify_trained(english_text: str) -> tuple:
    """Fast classifier inference (<10ms). Returns (intent, confidence)."""
    _load_classifier()
    if _classifier is None:
        return None, 0.0
    vec = _vectorizer.transform([english_text])
    probs = _classifier.predict_proba(vec)[0]
    top_idx = probs.argmax()
    return _classifier.classes_[top_idx], float(probs[top_idx])


def _extract_intent_llm(english_text: str) -> dict:
    """Fallback LLM intent and personalization flag extraction."""
    from core import llm
    ok, _ = llm.available()
    if not ok:
        return {"intent": "UNKNOWN", "personalization_flags": {}}

    system_prompt = """You are an intelligent assistant that understands what a user wants.

A visually impaired user is interacting with an assistive reading device.
They will give you a voice command. Your job is to understand the TRUE MEANING
of what they want and extract it as structured JSON.

You must return ONLY a valid JSON object with exactly two keys:

1. "intent": A short English verb phrase describing what the user wants to do.
   Examples: "SUMMARIZE", "EXPLAIN", "SIMPLIFY", "ELABORATE", "REPHRASE",
   "IDENTIFY_CONTENT", "READ_ALOUD", "STOP", "REPEAT", "NEXT", "STEP_BY_STEP"

2. "personalization_flags": A JSON object extracting HOW the user wants it done.
   Look for mentions of:
   - "speed": "fast" or "slow"
   - "detail_level": "brief", "detailed", or "step_by_step"
   - "language_style": "simple" or "technical"
   If none are mentioned, return empty object {}.

Return ONLY the JSON. No explanation. No markdown. No extra text."""

    reply, _ = llm.chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User command: '{english_text}'"}
    ], temperature=0.0)

    if reply is None:
        return {"intent": "UNKNOWN", "personalization_flags": {}}

    raw = reply.strip()
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {"intent": "UNKNOWN", "personalization_flags": {}}


def detect_intent(sinhala_text: str) -> dict:
    """Full Stage 2 translation + intent detection pipeline."""
    english_text, t_trans = translate_to_english(sinhala_text)
    t1 = time.time()
    intent, confidence = _classify_trained(english_text)
    t_classify = time.time() - t1

    flags = {}
    approach = "trained_classifier"
    t_llm = 0.0

    if intent is None or confidence < CONFIDENCE_THRESHOLD:
        t2 = time.time()
        llm_res = _extract_intent_llm(english_text)
        t_llm = round(time.time() - t2, 3)
        intent = llm_res.get("intent", "UNKNOWN")
        flags = llm_res.get("personalization_flags", {})
        approach = "llm_fallback"

    return {
        "approach": approach,
        "sinhala_input": sinhala_text,
        "english_translation": english_text,
        "intent": intent,
        "confidence": round(confidence, 3) if approach == "trained_classifier" else None,
        "personalization_flags": flags,
        "translation_time_sec": t_trans,
        "classify_time_sec": round(t_classify, 3),
        "llm_time_sec": t_llm,
        "total_time_sec": round(t_trans + t_classify + t_llm, 3),
    }


# ---------------------------------------------------------------------------
# STAGE 3 & 4: FULL PERSONALIZATION ORCHESTRATION (main_flow.py logic)
# ---------------------------------------------------------------------------
def handle_voice_command(sinhala_text: str, user_id: str = 'user_001',
                          retrieved_chunk_id: str = None) -> dict:
    """
    Executes the full end-to-end 4-stage pipeline matching end_to_end_working_process.md:
      Stage 1: STT / Input
      Stage 2: Translation & Intent Detection
      Stage 3.0: System Command Interception
      Stage 3.1: Correction Signal Detection
      Stage 3.2: Interaction Logging (TinyDB)
      Stage 3.3: 4-Tier Style Decision (Intent -> Grounded Flags -> Keywords -> Model)
      Stage 3.4: Evidence-Only Learning (No self-training)
      Stage 4: Structured Output for Component 3 (RAG)
    """
    # ── Stage 1: STT Output ──
    stt_stage = {"sinhala_input": sinhala_text}

    # ── Stage 2: Translation + Intent Detection ──
    intent_stage = detect_intent(sinhala_text)
    intent = intent_stage["intent"]
    english_text = intent_stage["english_translation"]
    flags = intent_stage.get("personalization_flags", {})

    # ── Stage 3.0: System Command Check ──
    command = detect_system_command(intent, english_text)
    if command:
        intent_stage["retrieved_chunk_id"] = retrieved_chunk_id
        log_interaction(user_id, intent_stage, style_class=None)
        return {
            "route": "SYSTEM_COMMAND",
            "command": command,
            "action": get_command_action(command),
            "intent": intent,
            "english_translation": english_text,
            "style_class": "Detailed",
            "prompt_modifier": "",
            "personalization_flags": flags,
            "retrieved_chunk_id": retrieved_chunk_id,
            "correction_applied": None,
            "user_profile": get_user_summary(user_id),
            "stage_breakdown": {
                "stt_stage": stt_stage,
                "intent_stage": intent_stage,
                "personalization_stage": {
                    "is_system_command": True,
                    "command": command,
                    "style_class": None,
                    "style_source": None,
                    "learned": False,
                    "correction_applied": None,
                },
            }
        }

    # ── Stage 3.1: Correction Check ──
    last = get_last_interaction(user_id)
    corrected_style = detect_correction_signal(intent, last)
    if corrected_style is not None and last is not None:
        update_interaction_style_by_timestamp(user_id, last.get("timestamp"), corrected_style)
        learn_style(user_id, last.get("english_translation", ""), corrected_style)

    # ── Stage 3.2: Log Interaction ──
    intent_stage["retrieved_chunk_id"] = retrieved_chunk_id
    log_interaction(user_id, intent_stage)

    # ── Stage 3.3: 4-Tier Style Decision ──
    intent_style = style_from_intent(intent)
    flag_style = style_from_flags(flags, english_text)
    keyword_style = style_from_keywords(english_text)

    if intent_style:
        style, style_source, is_evidence = intent_style, "explicit_intent (Tier 1)", True
    elif flag_style:
        style, style_source, is_evidence = flag_style, "explicit_flag (Tier 2)", True
    elif keyword_style:
        style, style_source, is_evidence = keyword_style, "text_keywords (Tier 3)", True
    else:
        style, src = predict_style(user_id, english_text)
        style_source = f"online_model:{src} (Tier 4)"
        is_evidence = False

    prompt_modifier = STYLE_PROMPT_MODIFIERS.get(style, STYLE_PROMPT_MODIFIERS["Detailed"])

    # ── Stage 3.4: Record Style & Evidence-Only Training ──
    update_last_interaction_style(user_id, style)
    if is_evidence:
        learn_style(user_id, english_text, style)

    user_profile = get_user_summary(user_id)

    return {
        "route": "GENERATE",
        "intent": intent,
        "english_translation": english_text,
        "style_class": style,
        "prompt_modifier": prompt_modifier,
        "personalization_flags": flags,
        "retrieved_chunk_id": retrieved_chunk_id,
        "correction_applied": corrected_style,
        "user_profile": user_profile,
        "stage_breakdown": {
            "stt_stage": stt_stage,
            "intent_stage": intent_stage,
            "personalization_stage": {
                "is_system_command": False,
                "command": None,
                "style_class": style,
                "style_source": style_source,
                "learned": is_evidence,
                "correction_applied": corrected_style,
                "user_profile": user_profile,
            },
        }
    }


# ---------------------------------------------------------------------------
# FASTAPI APPLICATION & ENDPOINTS
# ---------------------------------------------------------------------------
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


def build() -> FastAPI:
    svc = FastAPI(
        title='svc-voice · Component 4',
        description='Adaptive Conversational Personalization & Voice Interaction Module (Bumal)'
    )

    svc.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    @svc.get('/')
    def index():
        return {
            'ok': True,
            'service': 'svc-voice (Component 4 — Voice Interaction & Personalization)',
            'student': 'Sathsara S.P.Y.B — IT22004468',
            'endpoint': 'POST /interpret',
            'endpoints': ['GET /health', 'POST /interpret', 'GET /users', 'GET /user/{user_id}', 'POST /demo/compare'],
            'status': 'OPERATIONAL',
        }

    @svc.get('/health')
    def health():
        has_nllb = _nllb_model is not None
        has_classifier = _classifier is not None
        from core import llm
        llm_ok, llm_why = llm.available()
        return {
            'ok': True,
            'nllb_loaded': has_nllb,
            'classifier_loaded': has_classifier,
            'openai_available': llm_ok,
            'openai_reason': llm_why if not llm_ok else '',
            'active_users': list(_users.keys()),
        }

    @svc.post('/interpret')
    def interpret(body: dict):
        text = body.get('text', '')
        user_id = body.get('user_id', 'user_001')
        chunk_id = body.get('retrieved_chunk_id')

        if not text.strip():
            return {
                'route': 'GENERATE', 'intent': 'ASK',
                'english_translation': '',
                'style_class': 'Detailed', 'prompt_modifier': '',
                'personalization_flags': {},
                'retrieved_chunk_id': chunk_id,
                'correction_applied': None,
                'user_profile': get_user_summary(user_id),
            }

        try:
            result = handle_voice_command(text, user_id=user_id, retrieved_chunk_id=chunk_id)
            return result
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse({
                'route': 'GENERATE', 'intent': 'ASK',
                'english_translation': text,
                'style_class': 'Detailed', 'prompt_modifier': '',
                'personalization_flags': {},
                'retrieved_chunk_id': chunk_id,
                'correction_applied': None,
                'user_profile': get_user_summary(user_id),
                'warning': f'{type(e).__name__}: {e}',
            }, status_code=200)

    @svc.get('/users')
    def list_users():
        """Returns all tracked users, their confirmed counts, and preference summaries."""
        res = {}
        for uid in _users.keys():
            res[uid] = get_user_summary(uid)
        return {'ok': True, 'users': res}

    @svc.get('/user/{user_id}')
    def get_user(user_id: str):
        """Returns full profile, history weights, and recent interactions for a specific user."""
        summary = get_user_summary(user_id)
        history = get_user_history(user_id, limit=10)
        return {
            'ok': True,
            'user_id': user_id,
            'summary': summary,
            'history': history,
        }

    @svc.post('/demo/compare')
    def demo_compare():
        """Runs the 3-user Viva comparison demo dynamically."""
        teaching_commands = {
            "user_001": ["සරල කරන්න", "කෙටියෙන් කියන්න", "සරලව පැහැදිලි කරන්න"],
            "user_002": ["විස්තරාත්මකව කියන්න", "සම්පූර්ණ විස්තර දෙන්න", "තවදුරටත් පැහැදිලි කරන්න"],
            "user_003": ["පියවරෙන් පියවර කියන්න", "එකින් එක පැහැදිලි කරන්න", "පියවර මඟින් කියන්න"],
        }
        # Phase 1: Teach
        teach_results = {}
        for uid, cmds in teaching_commands.items():
            teach_results[uid] = []
            for cmd in cmds:
                r = handle_voice_command(cmd, user_id=uid)
                teach_results[uid].append({
                    "input": cmd,
                    "intent": r["intent"],
                    "style": r["style_class"],
                })

        # Phase 2: Test on identical neutral question
        neutral_query = "මෙහි ඇත්තේ කුමක්ද"
        comparison = {}
        for uid in ["user_001", "user_002", "user_003"]:
            res = handle_voice_command(neutral_query, user_id=uid)
            comparison[uid] = {
                "user_id": uid,
                "input": neutral_query,
                "predicted_style": res["style_class"],
                "prompt_modifier": res["prompt_modifier"],
                "style_source": res.get("stage_breakdown", {}).get("personalization_stage", {}).get("style_source", "online_model"),
                "profile": get_user_summary(uid),
            }

        return {
            "ok": True,
            "neutral_query": neutral_query,
            "comparison": comparison,
            "teaching_phase": teach_results,
        }

    return svc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8101)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--preload', action='store_true', default=True,
                    help='Load NLLB at startup (default: yes)')
    a = ap.parse_args()

    if a.preload:
        _load_nllb()
        _load_classifier()

    svc = build()

    print("\n" + "="*65)
    print("  svc-voice (Component 4 — Voice Interaction & Personalization)")
    print(f"  Port: {a.port} · http://{a.host}:{a.port}/interpret")
    print("="*65)
    print(f'  NLLB       : {"loaded" if _nllb_model else "will load on first request"}')
    print(f'  classifier : {"loaded" if _classifier else "NOT FOUND"}')
    print(f'  OpenAI     : configured' if os.getenv('OPENAI_API_KEY') else '  OpenAI     : not configured')
    print()

    import uvicorn
    uvicorn.run(svc, host=a.host, port=a.port, log_level='info')


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    main()

