# stt/google_stt_live.py
#
# Live-microphone version of the Google STT approach selected in
# stt/google_stt.py and stt/stt_all_approaches.py (SpeechRecognition +
# Google Web Speech API, language="si-LK"). That original script reads
# pre-recorded .wav files from a mounted Google Drive folder inside Colab —
# useful for the WER/CER evaluation, but not runnable as-is for a live app.
#
# This file keeps the exact same STT engine and language code, just swaps
# the audio SOURCE from "a .wav file in Drive" to "the local microphone",
# so it can be called directly from app.py.
#
# NOTE: requires the SpeechRecognition and PyAudio packages locally
# (see requirements.txt). PyAudio in particular can be awkward to install
# on Windows — see the note at the bottom of this file if `pip install
# pyaudio` fails.

import speech_recognition as sr


def listen_and_transcribe(timeout=8, phrase_time_limit=15, language="si-LK"):
    """
    Listens on the default microphone for one utterance and returns the
    transcribed Sinhala text (str), or None if nothing usable was captured.

    timeout:            seconds to wait for speech to START before giving up
    phrase_time_limit:  max seconds to record once speech has started
    language:           STT language code — si-LK is the selected approach
    """
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            print("  Calibrating for background noise...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("  Listening... speak now")
            audio_data = recognizer.listen(
                source, timeout=timeout, phrase_time_limit=phrase_time_limit
            )
    except sr.WaitTimeoutError:
        print("  No speech detected — timed out.")
        return None
    except OSError as e:
        print(f"  Microphone error: {e}")
        print("  Check that a microphone is connected and accessible.")
        return None

    print("  Recognizing (Google STT, si-LK)...")
    try:
        text = recognizer.recognize_google(audio_data, language=language)
        return text
    except sr.UnknownValueError:
        print("  Could not understand the audio — try speaking again.")
        return None
    except sr.RequestError as e:
        print(f"  Google STT request failed: {e}")
        print("  (This needs an internet connection.)")
        return None


# ─── Quick manual test ──────────────────────────────────────────────────
# Run this file directly to test the microphone + Google STT in isolation,
# without going through the rest of the pipeline.
if __name__ == "__main__":
    print("=" * 60)
    print("Google STT — Live Microphone Test (si-LK)")
    print("=" * 60)
    result = listen_and_transcribe()
    if result:
        print(f"\nTranscribed: {result}")
    else:
        print("\nNo usable transcription captured.")

# ─── If `pip install pyaudio` fails on Windows ──────────────────────────
# PyAudio doesn't always have a prebuilt wheel for the latest Python on
# Windows. If the plain pip install fails, try:
#   pip install pipwin
#   pipwin install pyaudio
# or download a matching .whl from a trusted source and:
#   pip install path\to\PyAudio‑X.X.X‑cpXX‑cpXX‑win_amd64.whl
