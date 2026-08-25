# main.py
from dotenv import load_dotenv
load_dotenv()

from vectorstore import build_vectorstore
from pipeline import run_pipeline
from contracts import SAMPLE_OCR_INPUT, SAMPLE_VOICE_INPUT

if __name__ == "__main__":
    vs = build_vectorstore()
    result = run_pipeline(vs, SAMPLE_OCR_INPUT, SAMPLE_VOICE_INPUT)
    print(result)