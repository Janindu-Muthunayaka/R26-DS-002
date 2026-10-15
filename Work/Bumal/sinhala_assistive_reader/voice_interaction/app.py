"""
app.py — Sinhala Assistive Reader: Component 4 Live Pipeline

Run this to watch the whole Component 4 turn, from speech to speech:

    Sinhala input -> Translation + Intent -> Personalization -> Prompt
    -> Component 3 answer (mock) -> Adaptive delivery (plan, play, feedback)

Stages 5-6 (Phase 3, ADAPTIVE_VOICE_PLAN.md) are run by delivery/turn.py.
The answer is spoken with gTTS + ffplay (needs internet and FFmpeg); press
Enter while it speaks to stop it. Long answers pause and ask "දිගටම
කියන්නද?" (continue?) — answer yes / no, or say a new request.

Usage:
    python app.py                                  (menus)
    python app.py --mode voice --user demo_user    (live mic demo, no menus)
    python app.py --mode typed --no-audio          (no sound: prints the plan)

For the side-by-side 3-user comparison table used in the viva, run
compare_users.py instead.
"""

import argparse
import sys
import json
import time

from colorama import init as colorama_init, Fore, Style
colorama_init(autoreset=True)

from data.test_samples import test_samples
from delivery.laptop_player import play_plan, play_plan_silent
from delivery.mock_c3 import MockComponent3
from delivery.turn import TurnRunner
from personalization.main_flow import handle_voice_command
from personalization.style_model import get_user_summary

DEMO_USERS = ["user_001", "user_002", "user_003"]
STAGE_DELAY_SEC = 0.4
MODES = {"voice": "1", "typed": "2", "samples": "3"}


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


def _turn_header(turn_number):
    print(Fore.YELLOW + Style.BRIGHT + f"\n╔{'═'*62}╗")
    print(Fore.YELLOW + Style.BRIGHT + f"║  TURN {turn_number}" + " " * (55 - len(str(turn_number))) + "║")
    print(Fore.YELLOW + Style.BRIGHT + f"╚{'═'*62}╝")


def display_result(result):
    stt = result["stt_stage"]
    intent_stage = result["intent_stage"]
    pers = result["personalization_stage"]
    final = result["final_prompt"]
    frame = result["frame"]

    _header("STAGE 1 — Speech-to-Text (Sinhala)", Fore.MAGENTA)
    _kv("Sinhala input:", stt["sinhala_input"])
    _pause()

    _header("STAGE 2 — Translation + Intent Detection", Fore.BLUE)
    _kv("English translation:", intent_stage["english_translation"])
    _kv("Detected intent:", intent_stage["intent"])
    _kv("Frame (main action):", frame["intent"])
    _kv("Modifiers:", frame["modifiers"])
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


# ── Stages 5-6: shown from delivery/turn.py events ─────────────────────────

class DeliveryDisplay:
    """Prints Stage 5 (mock Component 3 answer) and Stage 6 (adaptive
    delivery) as delivery.turn.TurnRunner reports them."""

    def __init__(self):
        self.turn_feedback = None
        self.stage6_shown = False

    def new_turn(self):
        self.turn_feedback, self.stage6_shown = None, False

    def _stage6(self):
        if self.stage6_shown:
            return
        self.stage6_shown = True
        _header("STAGE 6 — Adaptive Delivery (plan, play, feedback)", Fore.LIGHTMAGENTA_EX)
        d = self.turn_feedback
        if d:
            _kv("This turn says:", "; ".join(d["reasons"]) or "no delivery signal")
            if d["changes"]:
                _kv("Settings changed:", d["changes"], Fore.LIGHTYELLOW_EX)

    def event(self, name, data):
        if name == "command_result":
            display_result(data)
        elif name == "fast_path":
            _header("STAGE 1 — Speech-to-Text (Sinhala)", Fore.MAGENTA)
            _kv("Short command:", f"repeat={data['modifiers']['repeat']}, "
                                  f"speed={data['modifiers']['speed']}")
            _kv("Stages 2-4:", "skipped — handled from the Sinhala words (fast path)",
                Fore.LIGHTBLACK_EX)
        elif name == "feedback":
            self.turn_feedback = data
        elif name == "answer":
            _header("STAGE 5 — Component 3 Answer (mock)", Fore.GREEN)
            _kv("Article:", f"{data['title']}  ({data['article_id']})")
            _kv("Style:", data["style_class"])
            print(Fore.WHITE + "  " + data["answer_si"])
            _pause()
        elif name == "plan":
            self._stage6()
            plan, start, end = data["plan"], data["start"], data["end"]
            n = len(plan["segments"])
            _kv("Action:", data["action"])
            _kv("Rate / pitch:", f"{plan['rate']} / {plan['pitch']}")
            _kv("Segments:", f"{n} (playing {start}–{(end or n) - 1})")
            _kv("Pause after each:", f"{plan['segments'][0]['pause_after_ms'] if n > 1 else 0} ms")
            _kv("Check-ins after:", plan["checkpoints"] or "none")
            if plan["preview"] and start == 0:
                _kv("Preview:", plan["preview"])
            print(Fore.LIGHTBLACK_EX + "  (press Enter to stop)")
        elif name == "segment":
            i, seg = data
            print(f"  {Fore.LIGHTBLACK_EX}[{i}]{Style.RESET_ALL} {seg['text']}")
        elif name == "playback":
            if data["completed"]:
                _kv("Playback:", "finished", Fore.LIGHTGREEN_EX)
            elif data["stopped_in"] == "checkpoint":
                _kv("Playback:", f"paused for a check-in before segment {data['segments_played']}")
            else:
                _kv("Playback:", f"stopped by you at segment {data['stopped_at']} "
                                 f"({data['stopped_in']})", Fore.LIGHTYELLOW_EX)
        elif name == "playback_feedback":
            if data["reasons"]:
                _kv("After the answer:", "; ".join(data["reasons"]))
            if data["changes"]:
                _kv("Settings changed:", data["changes"], Fore.LIGHTYELLOW_EX)
        elif name == "say":
            self._stage6()
            _kv("System says:", data, Fore.LIGHTCYAN_EX)
        elif name == "system":
            if data["article"] is None:
                _kv("System command:", f"{data['command']} — not part of this demo",
                    Fore.LIGHTBLACK_EX)
        elif name == "error":
            print(Fore.RED + f"  Playback failed: {data}")


def make_runner(user_id, audio=True):
    display = DeliveryDisplay()
    runner = TurnRunner(user_id, handle_voice_command, MockComponent3(),
                        play_plan if audio else play_plan_silent, on_event=display.event)
    runner.display = display
    return runner


def run_turn(sinhala_text, runner, turn_number):
    _turn_header(turn_number)
    runner.display.new_turn()
    rec = runner.turn(sinhala_text)
    p = rec["profile_after"]
    print()
    _kv("Route:", f"{rec['route']} / {rec['action']}", Fore.LIGHTCYAN_EX)
    _kv("Voice profile now:", f"rate {p['rate']}, pause x{p['pause_scale']}, "
                              f"chunk {p['chunk_size']}")
    if runner.pending:
        print(Fore.LIGHTCYAN_EX + "  Waiting for your answer to the question above.")
    print()
    return rec


def choose_mode(preset=None):
    if preset:
        return MODES[preset]
    print(Fore.CYAN + Style.BRIGHT + "\nSinhala Assistive Reader — Component 4 Live Pipeline")
    print(Fore.CYAN + "=" * 64)
    print("  1) Voice input — speak into the microphone (Google STT, si-LK)")
    print("  2) Type Sinhala input live (simulates a real STT result)")
    print("  3) Step through the saved test samples (data/test_samples.py)")
    print(Fore.CYAN + "=" * 64)
    choice = input("Choose a mode [1/2/3]: ").strip()
    return choice if choice in ("1", "2", "3") else "2"


def choose_user(preset=None):
    if preset:
        return preset
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


def voice_loop(runner):
    try:
        from stt.google_stt_live import listen_and_transcribe
    except ImportError as e:
        print(Fore.RED + f"\nCouldn't load voice input ({e}).")
        print(Fore.RED + "Install SpeechRecognition and PyAudio, or use mode 2.")
        print(Fore.YELLOW + "Falling back to typed input.\n")
        typed_loop(runner)
        return

    print(Fore.LIGHTBLACK_EX + "\nPress Enter, then speak one Sinhala command. Type 'q' to quit."
          "\nWhile an answer is spoken, press Enter to stop it.\n")
    turn = 0
    while True:
        cmd = input(Fore.WHITE + Style.BRIGHT + "[Enter to speak, 'q' to quit]: " + Style.RESET_ALL)
        if cmd.strip().lower() == "q":
            break
        sinhala_text = listen_and_transcribe()
        if not sinhala_text:
            continue
        turn += 1
        run_turn(sinhala_text, runner, turn)


def typed_loop(runner):
    print(Fore.LIGHTBLACK_EX + "Type a Sinhala sentence and press Enter. Type 'quit' to stop.\n")
    turn = 0
    while True:
        sinhala_text = input(Fore.WHITE + Style.BRIGHT + "User (Sinhala): " + Style.RESET_ALL).strip()
        if sinhala_text.lower() in ("quit", "exit"):
            break
        if not sinhala_text:
            continue
        turn += 1
        run_turn(sinhala_text, runner, turn)


def test_sample_loop(runner):
    for turn, sample in enumerate(test_samples, 1):
        input(Fore.LIGHTBLACK_EX + f"\n[Enter to send '{sample['id']}': {sample['stt_output']}]")
        run_turn(sample["stt_output"], runner, turn)
    print(Fore.YELLOW + "\nAll test samples processed.")


def parse_args():
    parser = argparse.ArgumentParser(description="Component 4 live pipeline (speech to speech).")
    parser.add_argument("--mode", choices=sorted(MODES), help="skip the mode menu")
    parser.add_argument("--user", help="skip the user menu (any ID; a new ID starts with defaults)")
    parser.add_argument("--no-audio", action="store_true",
                        help="don't speak: print the speech plan only (no internet / FFmpeg needed)")
    return parser.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # Sinhala in the Windows console
    args = parse_args()
    mode = choose_mode(args.mode)
    user_id = choose_user(args.user)
    runner = make_runner(user_id, audio=not args.no_audio)
    p = runner.profile
    print(Fore.LIGHTBLACK_EX + f"\nActive user: {user_id}  (voice: rate {p['rate']}, "
          f"pause x{p['pause_scale']}, chunk {p['chunk_size']})"
          "\n(Each user has an independent learned profile. Switch users from "
          "the start menu to compare, or run compare_users.py for a table.)\n")

    if mode == "1":
        voice_loop(runner)
    elif mode == "3":
        test_sample_loop(runner)
    else:
        typed_loop(runner)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n\nStopped.")
        sys.exit(0)
