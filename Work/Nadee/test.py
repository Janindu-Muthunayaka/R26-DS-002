from dotenv import load_dotenv
load_dotenv()

from vectorstore import build_vectorstore
from pipeline import run_pipeline
from contracts import SAMPLE_OCR_INPUT, SAMPLE_VOICE_INPUT
import copy

vs = build_vectorstore()

def run_case(label, voice_override=None, ocr_override=None):
    print(f"\n{'='*50}\nCASE: {label}\n{'='*50}")
    voice = copy.deepcopy(SAMPLE_VOICE_INPUT)
    ocr = copy.deepcopy(SAMPLE_OCR_INPUT)
    if voice_override:
        voice.update(voice_override)
    if ocr_override:
        ocr.update(ocr_override)
    try:
        result = run_pipeline(vs, ocr, voice)
        print("RESULT:", result)
    except Exception as e:
        print("CRASHED:", type(e).__name__, "-", e)


real_ocr = {
  "corrected_text": "ශ්‍රී ලංකාවේ ආර්ථිකය",
  "tokens": [
    {
      "original": "ශ්‍රී",
      "corrected": "ශ්‍රී",
      "label": "CORRECT",
      "confidence": 0.97,
      "was_changed": False
    },
    {
      "original": "ලංකාවේ",
      "corrected": "ලංකාවේ",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "ආරථිකය",
      "corrected": "ආර්ථිකය",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "වර්ධනය",
      "corrected": "වර්ධනය",
      "label": "CORRECT",
      "confidence": 0.91,
      "was_changed": False
    }
  ]
}
run_case("missing personalization_flags", voice_override={"personalization_flags": {}})
run_case("unknown intent", voice_override={"intent": "DO_A_BACKFLIP"})
run_case("stale chunk id", voice_override={"retrieved_chunk_id": "chunk_does_not_exist_0"})
run_case("no chunk id", voice_override={"retrieved_chunk_id": None})
run_case("non-generate route", voice_override={"route": "CLARIFY"})
run_case("real OCR shape", ocr_override=real_ocr)
run_case("empty query text", voice_override={"english_translation": ""})