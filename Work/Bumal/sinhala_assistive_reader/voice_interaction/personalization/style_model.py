# personalization/style_model.py
# Step 3 — River online learning pipeline for style prediction, with persistence
#
# IMPORTANT: this holds ONE INDEPENDENT MODEL PER USER, not one shared model.
# Each user_id gets its own TFIDF+SoftmaxRegression pipeline, trained only on
# that user's own interactions. This is what makes the personalization
# genuinely per-user: user_001's corrections never influence what gets
# predicted for user_002, and each user's learned tendency can be shown to
# differ live in a demo.

import os
import pickle
from river import compose, feature_extraction, linear_model

STYLE_CLASSES = ["Simple", "Detailed", "StepByStep"]
DEFAULT_STYLE = "Detailed"  # used before a given user's model has seen any data

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "style_model.pkl"
)


def _fresh_user_state():
    return {
        "model": compose.Pipeline(
            feature_extraction.TFIDF(),
            linear_model.SoftmaxRegression()
        ),
        "seen_any_data": False,
    }


def _load_all_users():
    """Loads {user_id: {'model':..., 'seen_any_data':...}, ...} from disk.
    Returns an empty dict if no file exists yet or it can't be read."""
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)
            # Backward-compat: older single-model files aren't a per-user
            # dict. Treat them as empty rather than crashing — any user
            # will just start fresh, which is safe (just re-learns).
            if isinstance(data, dict) and all(
                isinstance(v, dict) and "model" in v for v in data.values()
            ):
                return data
        except Exception:
            pass
    return {}


def _save_all_users():
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(_users, f)


# Module-level state: a dict of per-user states, loaded once on import
_users = _load_all_users()


def _get_user_state(user_id):
    if user_id not in _users:
        _users[user_id] = _fresh_user_state()
    return _users[user_id]


def predict_style(user_id, english_text):
    """
    Predicts THIS user's preferred style class for this text, using only
    that user's own model. Falls back to DEFAULT_STYLE if this user's
    model hasn't learned anything yet.
    """
    state = _get_user_state(user_id)
    if not state["seen_any_data"]:
        return DEFAULT_STYLE
    prediction = state["model"].predict_one(english_text)
    return prediction if prediction is not None else DEFAULT_STYLE


def learn_style(user_id, english_text, confirmed_class):
    """
    Updates THIS user's model with the true (or corrected) label for this
    interaction, then immediately persists ALL users' models to disk.
    confirmed_class must be one of STYLE_CLASSES.
    """
    if confirmed_class not in STYLE_CLASSES:
        raise ValueError(f"confirmed_class must be one of {STYLE_CLASSES}, got {confirmed_class}")
    state = _get_user_state(user_id)
    state["model"].learn_one(english_text, confirmed_class)
    state["seen_any_data"] = True
    _save_all_users()
