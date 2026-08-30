# personalization/style_model.py
# River online learning for communication-style prediction.
#
# THREE DESIGN RULES, each fixing a specific defect found in the previous
# version (measured on 41 real logged interactions: 35 Simple, 4 Detailed,
# 0 StepByStep — the model had become a majority-class predictor):
#
# 1. EVIDENCE-ONLY LEARNING. The model learns ONLY from turns where the user
#    gave explicit evidence of their preference (an explicit SIMPLIFY /
#    ELABORATE / STEP_BY_STEP intent, an explicit personalization flag, or a
#    correction of a previous turn). It NEVER learns from its own prediction.
#    Previously every turn was learned from, including the model's own
#    guesses, so the model just reinforced whatever it already believed.
#
# 2. PER-USER MODELS. Each user_id gets its own independent pipeline. One
#    user's evidence never influences another user's predictions.
#
# 3. USER-HISTORY FEATURES. Style preference is a property of the USER, not
#    of the sentence. TF-IDF over "what is this about" carries no style
#    information at all, so text-only prediction collapses to the class
#    prior. We therefore union the TF-IDF text features with a compact
#    summary of that user's own confirmed history (recency-weighted).

import os
import pickle
from river import compose, feature_extraction, linear_model

STYLE_CLASSES = ["Simple", "Detailed", "StepByStep"]
DEFAULT_STYLE = "Detailed"

# A user's model is only trusted once it has this many confirmed examples.
# Below it, we fall back to their history summary (or the default).
MIN_CONFIRMED_FOR_MODEL = 3

# If the model's top class is less confident than this, prefer the user's
# recency-weighted historical preference over a low-confidence guess.
CONFIDENCE_THRESHOLD = 0.45

# Recency weighting: each new confirmation decays older ones by this factor,
# so a user who changes their mind is followed rather than out-voted by
# their own history.
HISTORY_DECAY = 0.85

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "style_model.pkl"
)


def _fresh_user_state():
    return {
        "model": compose.Pipeline(
            compose.TransformerUnion(
                compose.Select("text") | feature_extraction.TFIDF(on="text"),
                compose.Select("h_Simple", "h_Detailed", "h_StepByStep", "n_conf"),
            ),
            linear_model.SoftmaxRegression(),
        ),
        # Recency-weighted count of confirmed evidence per style
        "history": {cls: 0.0 for cls in STYLE_CLASSES},
        "n_confirmed": 0,
    }


def _load_all_users():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict) and all(
                isinstance(v, dict) and "model" in v and "history" in v
                for v in data.values()
            ):
                return data
        except Exception:
            pass
    return {}


def _save_all_users():
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(_users, f)


_users = _load_all_users()


def _get_user_state(user_id):
    if user_id not in _users:
        _users[user_id] = _fresh_user_state()
    return _users[user_id]


def _build_features(state, english_text):
    """Combines the utterance text with a normalized summary of this user's
    own confirmed history. The history part is what lets the model answer
    'what does THIS user usually want?' on a neutral utterance whose words
    carry no style signal."""
    hist = state["history"]
    total = sum(hist.values()) or 1.0
    return {
        "text": english_text or "",
        "h_Simple": hist["Simple"] / total,
        "h_Detailed": hist["Detailed"] / total,
        "h_StepByStep": hist["StepByStep"] / total,
        "n_conf": min(state["n_confirmed"], 20) / 20.0,
    }


def _history_majority(state):
    """The user's dominant confirmed preference, or None if no evidence."""
    hist = state["history"]
    if sum(hist.values()) == 0:
        return None
    return max(hist, key=hist.get)


def predict_style(user_id, english_text):
    """
    Predicts this user's preferred style. Returns (style, source) where
    source explains WHICH mechanism decided it — useful for the demo and
    for honestly reporting what the model contributed.

    source is one of:
      "cold_start"       — not enough confirmed evidence yet, using default
      "history_fallback" — model was unsure, used this user's history instead
      "model"            — the online model's own confident prediction
    """
    state = _get_user_state(user_id)

    if state["n_confirmed"] < MIN_CONFIRMED_FOR_MODEL:
        return (_history_majority(state) or DEFAULT_STYLE), "cold_start"

    features = _build_features(state, english_text)
    try:
        proba = state["model"].predict_proba_one(features)
    except Exception:
        proba = None

    if not proba:
        return (_history_majority(state) or DEFAULT_STYLE), "history_fallback"

    top_class = max(proba, key=proba.get)
    if proba[top_class] < CONFIDENCE_THRESHOLD:
        return (_history_majority(state) or DEFAULT_STYLE), "history_fallback"

    return top_class, "model"


def learn_style(user_id, english_text, confirmed_class):
    """
    Learns from ONE piece of real user evidence.

    Call this ONLY when the user actually signalled their preference —
    an explicit style intent, an explicit flag, or a correction. Never call
    it with a style the model itself predicted; that is the self-training
    loop this design exists to avoid.
    """
    if confirmed_class not in STYLE_CLASSES:
        raise ValueError(f"confirmed_class must be one of {STYLE_CLASSES}, got {confirmed_class}")

    state = _get_user_state(user_id)

    # Features must be built BEFORE updating history, so the model learns
    # "given the history so far, this was the right answer".
    features = _build_features(state, english_text)
    try:
        state["model"].learn_one(features, confirmed_class)
    except Exception as e:
        print(f"Warning: Failed to update model: {e}")

    for cls in STYLE_CLASSES:
        state["history"][cls] *= HISTORY_DECAY
    state["history"][confirmed_class] += 1.0
    state["n_confirmed"] += 1

    _save_all_users()


def get_user_summary(user_id):
    """Read-only snapshot of what the system has learned about a user.
    Used by the comparison table and the live display."""
    state = _get_user_state(user_id)
    hist = state["history"]
    total = sum(hist.values()) or 1.0
    return {
        "n_confirmed": state["n_confirmed"],
        "history_weights": {k: round(v / total, 3) for k, v in hist.items()},
        "dominant_preference": _history_majority(state),
    }


def reset_user(user_id):
    """Wipes one user's learned profile (used to reset a demo)."""
    _users[user_id] = _fresh_user_state()
    _save_all_users()


def reset_all():
    """Wipes every user's learned profile."""
    _users.clear()
    _save_all_users()
