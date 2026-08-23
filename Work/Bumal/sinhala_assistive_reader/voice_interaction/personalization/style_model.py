# personalization/style_model.py
# Step 3 — River online learning pipeline for style prediction, with persistence

import os
import pickle
from river import compose, feature_extraction, linear_model

STYLE_CLASSES = ["Simple", "Detailed", "StepByStep"]
DEFAULT_STYLE = "Detailed"  # used before the model has seen any data

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "style_model.pkl"
)


def _fresh_pipeline():
    return compose.Pipeline(
        feature_extraction.TFIDF(),
        linear_model.SoftmaxRegression()
    )


def _load_state():
    """Loads {'model': ..., 'seen_any_data': ...} from disk if it exists,
    otherwise returns a fresh, untrained state."""
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return {"model": _fresh_pipeline(), "seen_any_data": False}


def _save_state():
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(_state, f)


# Module-level state, loaded once when this module is first imported
_state = _load_state()


def predict_style(english_text):
    """
    Predicts the user's preferred style class for this text.
    Falls back to DEFAULT_STYLE if the model hasn't learned anything yet.
    """
    if not _state["seen_any_data"]:
        return DEFAULT_STYLE
    prediction = _state["model"].predict_one(english_text)
    return prediction if prediction is not None else DEFAULT_STYLE


def learn_style(english_text, confirmed_class):
    """
    Updates the model with the true (or corrected) label for this
    interaction, then immediately persists the updated model to disk.
    confirmed_class must be one of STYLE_CLASSES.
    """
    if confirmed_class not in STYLE_CLASSES:
        raise ValueError(f"confirmed_class must be one of {STYLE_CLASSES}, got {confirmed_class}")
    _state["model"].learn_one(english_text, confirmed_class)
    _state["seen_any_data"] = True
    _save_state()
