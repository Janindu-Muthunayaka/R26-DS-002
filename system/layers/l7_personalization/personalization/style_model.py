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
from datetime import datetime
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

# Base splits now live in the user registry (personalization/users.py):
# 50/25/25 for the three built-ins, custom for added users. This map is only
# the fallback if the registry cannot be read.
USER_DOMINANT_STYLES = {
    "user_001": "Simple",
    "user_002": "Detailed",
    "user_003": "StepByStep",
}

# A fresh user starts with this much total history mass (1.0 / 0.5 / 0.5 for
# a 50/25/25 split), so the first few confirmations move it noticeably but do
# not swamp it.
BASE_HISTORY_MASS = 2.0

# The weight TIMELINE (18 Oct 2026): one point per change of the weights, from
# the user's creation or last reset, so the web page can draw the user's
# complete history and not only the current session. Capped so the pickle
# cannot grow without bound; the first point (the reset) is always kept.
TIMELINE_MAX = 2000


def _base_weights(user_id):
    """The user's registered base split (fractions summing to 1), or None."""
    try:
        from .users import get_base_weights
        w = get_base_weights(user_id)
        if w:
            return w
    except Exception:
        pass
    dominant = USER_DOMINANT_STYLES.get(user_id)
    if not dominant:
        return None
    return {c: (0.50 if c == dominant else 0.25) for c in STYLE_CLASSES}


def _enforce_base_limits(state, user_id):
    """A user's dominant style never drops below its base share (50% for the
    built-ins). Learning moves the rest of the mass between styles freely."""
    base = _base_weights(user_id) if user_id else None
    if not base:
        return
    dominant = max(STYLE_CLASSES, key=lambda c: base.get(c, 0))
    floor = base[dominant]
    hist = state.setdefault("history", {cls: 0.0 for cls in STYLE_CLASSES})
    others = sum(v for k, v in hist.items() if k != dominant)
    if others <= 0:
        if hist.get(dominant, 0) <= 0:
            hist[dominant] = 1.0
        return
    if floor >= 1.0:
        return
    min_dominant = others * (floor / (1.0 - floor))
    if hist.get(dominant, 0) < min_dominant:
        hist[dominant] = min_dominant


def _fresh_user_state(user_id=None):
    base = _base_weights(user_id)
    hist = {cls: 0.0 for cls in STYLE_CLASSES}
    if base:
        for cls in STYLE_CLASSES:
            hist[cls] = base.get(cls, 0.0) * BASE_HISTORY_MASS
    state = {
        "model": compose.Pipeline(
            compose.TransformerUnion(
                compose.Select("text") | feature_extraction.TFIDF(on="text"),
                compose.Select("h_Simple", "h_Detailed", "h_StepByStep", "n_conf"),
            ),
            linear_model.SoftmaxRegression(),
        ),
        "history": hist,
        "n_confirmed": 3 if base else 0,
        "timeline": [],
    }
    _record(state, user_id, "reset")
    return state


def _weights(state):
    hist = state.get("history") or {}
    total = sum(hist.values()) or 1.0
    return {cls: round(hist.get(cls, 0.0) / total, 3) for cls in STYLE_CLASSES}


def _record(state, user_id, reason, style=None):
    """Append the current weights to the user's timeline."""
    _enforce_base_limits(state, user_id)
    tl = state.setdefault("timeline", [])
    tl.append({"at": datetime.now().isoformat(timespec="seconds"), "reason": reason,
               "style": style, "n_confirmed": state.get("n_confirmed", 0),
               "weights": _weights(state)})
    if len(tl) > TIMELINE_MAX:
        del tl[1:len(tl) - TIMELINE_MAX + 1]


# The pre-registry starting split (1.10 / 0.45 / 0.45, i.e. 55/22.5/22.5).
# A user still sitting exactly on it has learned nothing, so it is moved to
# the new 50/25/25 baseline. Anyone who has learned keeps their weights.
_OLD_BASELINE = (1.10, 0.45, 0.45)


def _migrate_untouched(users):
    changed = False
    for uid, state in users.items():
        if state.get("n_confirmed") != 3:
            continue
        vals = sorted(state.get("history", {}).values(), reverse=True)
        if len(vals) == 3 and all(abs(a - b) < 1e-9 for a, b in zip(vals, _OLD_BASELINE)):
            users[uid] = _fresh_user_state(uid)
            changed = True
    return changed


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
if _migrate_untouched(_users):
    _save_all_users()


def _get_user_state(user_id):
    if user_id not in _users:
        _users[user_id] = _fresh_user_state(user_id)
    _enforce_base_limits(_users[user_id], user_id)
    return _users[user_id]


def _build_features(state, english_text, user_id=None):
    """Combines the utterance text with a normalized summary of this user's
    own confirmed history. The history part is what lets the model answer
    'what does THIS user usually want?' on a neutral utterance whose words
    carry no style signal."""
    if user_id:
        _enforce_base_limits(state, user_id)
    hist = state["history"]
    total = sum(hist.values()) or 1.0
    return {
        "text": english_text or "",
        "h_Simple": hist["Simple"] / total,
        "h_Detailed": hist["Detailed"] / total,
        "h_StepByStep": hist["StepByStep"] / total,
        "n_conf": min(state["n_confirmed"], 20) / 20.0,
    }


def _history_majority(state, user_id=None):
    """The user's dominant confirmed preference, or None if no evidence."""
    if user_id:
        _enforce_base_limits(state, user_id)
    hist = state["history"]
    if sum(hist.values()) == 0:
        return USER_DOMINANT_STYLES.get(user_id, DEFAULT_STYLE)
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
        return (_history_majority(state, user_id) or DEFAULT_STYLE), "cold_start"

    features = _build_features(state, english_text, user_id)
    try:
        proba = state["model"].predict_proba_one(features)
    except Exception:
        proba = None

    if not proba:
        return (_history_majority(state, user_id) or DEFAULT_STYLE), "history_fallback"

    top_class = max(proba, key=proba.get)
    if proba[top_class] < CONFIDENCE_THRESHOLD:
        return (_history_majority(state, user_id) or DEFAULT_STYLE), "history_fallback"

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
    _enforce_base_limits(state, user_id)
    _record(state, user_id, "learned", confirmed_class)

    _save_all_users()


def record_turn(user_id, style=None):
    """One timeline point for a turn that learned nothing, so the complete
    history grows with every turn and not only with learned preferences."""
    state = _get_user_state(user_id)
    _record(state, user_id, "turn", style)
    _save_all_users()


def get_user_summary(user_id):
    """Read-only snapshot of what the system has learned about a user.
    Used by the comparison table and the live display."""
    state = _get_user_state(user_id)
    _enforce_base_limits(state, user_id)
    hist = state["history"]
    total = sum(hist.values()) or 1.0
    return {
        "n_confirmed": state["n_confirmed"],
        "history_weights": {k: round(v / total, 3) for k, v in hist.items()},
        "dominant_preference": _history_majority(state, user_id),
    }


def get_user_timeline(user_id):
    """Every weight change since the user was created or last reset:
    {"since", "points": [{at, reason, style, n_confirmed, weights}]}.
    A user from before the timeline existed starts with one point now."""
    state = _get_user_state(user_id)
    if not state.get("timeline"):
        _record(state, user_id, "recording started")
        _save_all_users()
    tl = state["timeline"]
    return {"user_id": user_id, "since": tl[0]["at"], "reset": tl[0]["reason"] == "reset",
            "points": tl}


def reset_user(user_id):
    """Wipes one user's learned profile back to their base split."""
    _users[user_id] = _fresh_user_state(user_id)
    _save_all_users()


def delete_user_state(user_id):
    """Forget a deleted user's learned profile."""
    if _users.pop(user_id, None) is not None:
        _save_all_users()


def rename_user_state(old_id, new_id):
    """Move a user's learned profile to a new id (user renumbering)."""
    if old_id in _users:
        _users[new_id] = _users.pop(old_id)
        _save_all_users()


def reset_all():
    """Wipes every user's learned profile and restores baseline profiles."""
    _users.clear()
    for uid in USER_DOMINANT_STYLES:
        _users[uid] = _fresh_user_state(uid)
    _save_all_users()
