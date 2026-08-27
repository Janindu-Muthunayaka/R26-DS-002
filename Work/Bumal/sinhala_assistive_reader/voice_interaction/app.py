"""
app.py — Sinhala Assistive Reader: Component 4 Live Pipeline

This is the ONE entry point for Component 4. Run this file directly to see
the whole personalization pipeline work turn-by-turn, live:

    Sinhala input  -->  Translation + Intent Detection  -->
    Personalization (correction check, style prediction)  -->
    Final personalized prompt (what gets handed to Component 3 / RAG)

Every stage is printed as it completes, so you can watch the decision
being made in real time — this is what you run for your panel demo.

Usage:
    python app.py
"""

import sys
import json
import time

from colorama import init as colorama_init, Fore, Style
colorama_init(autoreset=True)

from data.test_samples import test_samples
from personalization.main_flow import handle_voice_command

DEFAULT_USER_ID = "user_001"
DEFAULT_CHUNK_ID = "chunk_article_1"

# Three fixed demo users so the panel can watch the SAME sentence get a
# DIFFERENT personalized outcome depending on who's "speaking" — this only
# works because style_model.py now keeps one online model per user_id
# instead of one shared model for everyone.
DEMO_USERS = {
    "1": ("user_001", "Student"),
    "2": ("user_002", "Elderly"),
    "3": ("user_003", "Professional"),
}

# Small pause between stage reveals so the flow reads as "live" rather than
# a wall of text dumped all at once. Set to 0 to disable.
STAGE_DELAY_SEC = 0.5


def _header(text, color=Fore.CYAN):
    width = 64
    print()
    print(color + "─" * width)
    print(color + f" {text}")
    print(color + "─" * width + Style.RESET_ALL)


def _kv(label, value, color=Fore.WHITE):
    print(f"  {Fore.LIGHTBLACK_EX}{label:<22}{Style.RESET_ALL}{color}{value}")


def _pause():
    if STAGE_DELAY_SEC:
        time.sleep(STAGE_DELAY_SEC)


def display_result(result, turn_number):
    """Prints all four stages of a single handle_voice_command() result,
    one section at a time. Pure presentation — no pipeline logic here."""

    stt = result["stt_stage"]
    intent_stage = result["intent_stage"]
    pers = result["personalization_stage"]
    final = result["final_prompt"]

    print(Fore.YELLOW + Style.BRIGHT + f"\n╔{'═'*62}╗")
    print(Fore.YELLOW + Style.BRIGHT + f"║  TURN {turn_number}" + " " * (55 - len(str(turn_number))) + "║")
    print(Fore.YELLOW + Style.BRIGHT + f"╚{'═'*62}╝")

    # Stage 1 — STT
    _header("STAGE 1 — Speech-to-Text (Sinhala)", Fore.MAGENTA)
    _kv("Sinhala input:", stt["sinhala_input"])
    _pause()

    # Stage 2 — Translation + Intent Detection
    _header("STAGE 2 — Translation + Intent Detection", Fore.BLUE)
    _kv("English translation:", intent_stage["english_translation"])
    _kv("Detected intent:", intent_stage["intent"])
    _kv("Personalization flags:", intent_stage["personalization_flags"])
    _kv("Translation time:", f"{intent_stage['translation_time_sec']}s")
    _kv("LLM intent time:", f"{intent_stage['llm_time_sec']}s")
    _pause()

    # Stage 3 — Personalization
    _header("STAGE 3 — Personalization", Fore.GREEN)
    if pers["repeat_failure"]:
        _kv("Repeat failure:", "True  → routing to TTS_REPLAY (ML bypassed)", Fore.LIGHTRED_EX)
    else:
        _kv("Repeat failure:", "False")
        _kv("Correction applied:", pers["correction_applied"])
        _kv("Predicted style:", pers["style_class"])
        _kv("Style source:", pers["style_source"])
    _pause()

    # Stage 4 — Final prompt to Component 3
    _header("STAGE 4 — Final Prompt → Component 3 (RAG)", Fore.CYAN)
    print(Fore.CYAN + json.dumps(final, indent=2, ensure_ascii=False))
    print()


def run_turn(sinhala_text, user_id, retrieved_chunk_id, turn_number):
    result = handle_voice_command(sinhala_text, user_id, retrieved_chunk_id=retrieved_chunk_id)
    display_result(result, turn_number)
    return result


def choose_mode():
    print(Fore.CYAN + Style.BRIGHT + "\nSinhala Assistive Reader — Component 4 Live Pipeline")
    print(Fore.CYAN + "=" * 64)
    print("  1) Voice input — speak into the microphone (Google STT, si-LK)")
    print("  2) Type Sinhala input live (simulates a real STT result)")
    print("  3) Step through the saved test samples (data/test_samples.py)")
    print(Fore.CYAN + "=" * 64)
    choice = input("Choose a mode [1/2/3]: ").strip()
    return choice if choice in ("1", "2", "3") else "2"


def choose_user():
    print(Fore.CYAN + Style.BRIGHT + "\nChoose a demo user")
    print(Fore.CYAN + "=" * 64)
    for key, (uid, persona) in DEMO_USERS.items():
        print(f"  {key}) {uid} — {persona} persona")
    print("  4) Custom user ID")
    print(Fore.CYAN + "=" * 64)
    choice = input("Choose a user [1-4]: ").strip()

    if choice in DEMO_USERS:
        return DEMO_USERS[choice][0]
    if choice == "4":
        return input("Enter custom user ID: ").strip() or DEFAULT_USER_ID

    return DEFAULT_USER_ID


def voice_loop(user_id, chunk_id):
    try:
        from stt.google_stt_live import listen_and_transcribe
    except ImportError as e:
        print(Fore.RED + f"\nCouldn't load the voice input module ({e}).")
        print(Fore.RED + "Make sure SpeechRecognition and PyAudio are installed "
                          "(see stt/google_stt_live.py for the PyAudio note on Windows).")
        print(Fore.YELLOW + "Falling back to typed input instead.\n")
        typed_loop(user_id, chunk_id)
        return

    print(Fore.LIGHTBLACK_EX +
          "\nPress Enter, then speak one command in Sinhala. Type 'q' + Enter to quit.\n")
    turn = 0
    while True:
        cmd = input(Fore.WHITE + Style.BRIGHT + "[Enter to speak, or 'q' to quit]: " + Style.RESET_ALL)
        if cmd.strip().lower() == "q":
            break

        print(Fore.MAGENTA + "  Listening...")
        sinhala_text = listen_and_transcribe()
        if not sinhala_text:
            continue

        turn += 1
        run_turn(sinhala_text, user_id, chunk_id, turn)


def typed_loop(user_id, chunk_id):
    print(Fore.LIGHTBLACK_EX + "Type a Sinhala sentence and press Enter. Type 'quit' to stop.\n")
    turn = 0
    while True:
        sinhala_text = input(Fore.WHITE + Style.BRIGHT + "User (Sinhala): " + Style.RESET_ALL).strip()
        if sinhala_text.lower() in ("quit", "exit"):
            break
        if not sinhala_text:
            continue
        turn += 1
        run_turn(sinhala_text, user_id, chunk_id, turn)


def test_sample_loop(user_id, chunk_id):
    turn = 0
    for sample in test_samples:
        turn += 1
        input(Fore.LIGHTBLACK_EX + f"\n[Press Enter to send test sample '{sample['id']}': "
                                    f"{sample['stt_output']}]")
        run_turn(sample["stt_output"], user_id, chunk_id, turn)
    print(Fore.YELLOW + "\nAll test samples processed.")


def main():
    mode = choose_mode()
    user_id = choose_user()
    chunk_id = input(f"\nCurrent chunk ID [{DEFAULT_CHUNK_ID}]: ").strip() or DEFAULT_CHUNK_ID

    print(Fore.LIGHTBLACK_EX +
          f"\nActive user: {user_id}"
          "\n(chunk ID represents the article chunk currently loaded — keeping it "
          "the same across turns is what lets REPEAT and correction detection work. "
          "Switch users from the start menu to compare personalization across users.)")

    if mode == "1":
        voice_loop(user_id, chunk_id)
    elif mode == "3":
        test_sample_loop(user_id, chunk_id)
    else:
        typed_loop(user_id, chunk_id)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n\nStopped.")
        sys.exit(0)
