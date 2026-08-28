"""
app.py — Sinhala Assistive Reader: Component 4 Live Pipeline

Run this to watch the whole personalization pipeline work turn-by-turn:

    Sinhala input -> Translation + Intent -> Personalization -> Prompt

Usage:
    python app.py

For the side-by-side 3-user comparison table used in the viva, run
compare_users.py instead.
"""

import sys
import json
import time

from colorama import init as colorama_init, Fore, Style
colorama_init(autoreset=True)

from data.test_samples import test_samples
from personalization.main_flow import handle_voice_command
from personalization.style_model import get_user_summary

DEMO_USERS = ["user_001", "user_002", "user_003"]
STAGE_DELAY_SEC = 0.4


def _header(text, color=Fore.CYAN):
    print()
    print(color + "─" * 64)
    print(color + f" {text}")
    print(color + "─" * 64 + Style.RESET_ALL)


def _kv(label, value, color=Fore.WHITE):
    print(f"  {Fore.LIGHTBLACK_EX}{label:<22}{Style.RESET_ALL}{color}{value}")


def _pause():
    if STAGE_DELAY_SEC:
        time.sleep(STAGE_DELAY_SEC)


def display_result(result, turn_number):
    stt = result["stt_stage"]
    intent_stage = result["intent_stage"]
    pers = result["personalization_stage"]
    final = result["final_prompt"]

    print(Fore.YELLOW + Style.BRIGHT + f"\n╔{'═'*62}╗")
    print(Fore.YELLOW + Style.BRIGHT + f"║  TURN {turn_number}" + " " * (55 - len(str(turn_number))) + "║")
    print(Fore.YELLOW + Style.BRIGHT + f"╚{'═'*62}╝")

    _header("STAGE 1 — Speech-to-Text (Sinhala)", Fore.MAGENTA)
    _kv("Sinhala input:", stt["sinhala_input"])
    _pause()

    _header("STAGE 2 — Translation + Intent Detection", Fore.BLUE)
    _kv("English translation:", intent_stage["english_translation"])
    _kv("Detected intent:", intent_stage["intent"])
    _kv("Personalization flags:", intent_stage["personalization_flags"])
    _kv("Translation time:", f"{intent_stage['translation_time_sec']}s")
    _kv("LLM intent time:", f"{intent_stage['llm_time_sec']}s")
    _pause()

    _header("STAGE 3 — Personalization", Fore.GREEN)
    if pers["is_system_command"]:
        _kv("System command:", f"{pers['command']}  → bypasses personalization",
            Fore.LIGHTCYAN_EX)
        _kv("Model trained?", "No — commands carry no style signal")
    else:
        _kv("Style class:", pers["style_class"])
        _kv("Decided by:", pers["style_source"])
        _kv("Correction applied:", pers["correction_applied"])
        _kv("Model trained?",
            "Yes — real user evidence" if pers["learned"]
            else "No — this was a prediction, not evidence",
            Fore.LIGHTGREEN_EX if pers["learned"] else Fore.LIGHTBLACK_EX)
        prof = pers["user_profile"]
        _kv("Confirmed signals:", prof["n_confirmed"])
        _kv("Learned preference:", prof["dominant_preference"] or "(none yet)")
        _kv("Profile weights:", prof["history_weights"])
    _pause()

    _header("STAGE 4 — Final Prompt → Component 3 (RAG)", Fore.CYAN)
    print(Fore.CYAN + json.dumps(final, indent=2, ensure_ascii=False))
    print()


def run_turn(sinhala_text, user_id, turn_number):
    result = handle_voice_command(sinhala_text, user_id)
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
    print(Fore.CYAN + Style.BRIGHT + "\nChoose a user")
    print(Fore.CYAN + "=" * 64)
    for i, uid in enumerate(DEMO_USERS, 1):
        summary = get_user_summary(uid)
        pref = summary["dominant_preference"] or "no history yet"
        print(f"  {i}) {uid}  —  {summary['n_confirmed']} confirmed signals, {pref}")
    print(f"  {len(DEMO_USERS) + 1}) Custom user ID")
    print(Fore.CYAN + "=" * 64)
    choice = input(f"Choose a user [1-{len(DEMO_USERS) + 1}]: ").strip()

    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(DEMO_USERS):
            return DEMO_USERS[idx - 1]
        if idx == len(DEMO_USERS) + 1:
            return input("Enter custom user ID: ").strip() or DEMO_USERS[0]
    return DEMO_USERS[0]


def voice_loop(user_id):
    try:
        from stt.google_stt_live import listen_and_transcribe
    except ImportError as e:
        print(Fore.RED + f"\nCouldn't load voice input ({e}).")
        print(Fore.RED + "Install SpeechRecognition and PyAudio, or use mode 2.")
        print(Fore.YELLOW + "Falling back to typed input.\n")
        typed_loop(user_id)
        return

    print(Fore.LIGHTBLACK_EX + "\nPress Enter, then speak one Sinhala command. Type 'q' to quit.\n")
    turn = 0
    while True:
        cmd = input(Fore.WHITE + Style.BRIGHT + "[Enter to speak, 'q' to quit]: " + Style.RESET_ALL)
        if cmd.strip().lower() == "q":
            break
        sinhala_text = listen_and_transcribe()
        if not sinhala_text:
            continue
        turn += 1
        run_turn(sinhala_text, user_id, turn)


def typed_loop(user_id):
    print(Fore.LIGHTBLACK_EX + "Type a Sinhala sentence and press Enter. Type 'quit' to stop.\n")
    turn = 0
    while True:
        sinhala_text = input(Fore.WHITE + Style.BRIGHT + "User (Sinhala): " + Style.RESET_ALL).strip()
        if sinhala_text.lower() in ("quit", "exit"):
            break
        if not sinhala_text:
            continue
        turn += 1
        run_turn(sinhala_text, user_id, turn)


def test_sample_loop(user_id):
    for turn, sample in enumerate(test_samples, 1):
        input(Fore.LIGHTBLACK_EX + f"\n[Enter to send '{sample['id']}': {sample['stt_output']}]")
        run_turn(sample["stt_output"], user_id, turn)
    print(Fore.YELLOW + "\nAll test samples processed.")


def main():
    mode = choose_mode()
    user_id = choose_user()
    print(Fore.LIGHTBLACK_EX + f"\nActive user: {user_id}"
          "\n(Each user has an independent learned profile. Switch users from "
          "the start menu to compare, or run compare_users.py for a table.)\n")

    if mode == "1":
        voice_loop(user_id)
    elif mode == "3":
        test_sample_loop(user_id)
    else:
        typed_loop(user_id)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n\nStopped.")
        sys.exit(0)
