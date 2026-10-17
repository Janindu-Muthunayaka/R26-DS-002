# main.py
import os
from dotenv import load_dotenv
load_dotenv()

from vectorstore import build_vectorstore
from pipeline import run_pipeline
from contracts import SAMPLE_OCR_INPUT, SAMPLE_VOICE_INPUT

if __name__ == "__main__":
    vs = build_vectorstore()

    # Reads the user's raw age from the environment (e.g. USER_AGE=15).
    # age_to_group() in prompt.py handles mapping it to the correct band.
    # Defaults to 25 (-> "18+") if USER_AGE isn't set.
    user_age = os.getenv("USER_AGE")

    result = run_pipeline(
        vs, SAMPLE_OCR_INPUT, SAMPLE_VOICE_INPUT,
        raw_request_context={"age": int(user_age) if user_age else 25},
    )
    print(result)