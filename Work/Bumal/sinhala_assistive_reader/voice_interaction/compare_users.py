"""
compare_users.py — 3-user adaptiveness comparison for the viva.

Runs in two phases:

  PHASE 1 (teaching): each user is given a DIFFERENT short interaction
  history, using only explicit evidence — exactly the same channel a real
  user would use. This is not a pre-trained dataset; it is a scripted user
  session, and every turn is visible.

  PHASE 2 (comparison): all three users are asked the SAME neutral question
  — one with no style words in it, so the rules stay silent and only the
  learned profile can answer. The table then shows three different
  personalized prompts from one identical input.

Usage:
    python compare_users.py              # run teaching + comparison
    python compare_users.py --reset      # wipe profiles first
    python compare_users.py --html out.html
"""

import sys
import json

from colorama import init as colorama_init, Fore, Style
colorama_init(autoreset=True)

from personalization.main_flow import handle_voice_command
from personalization.style_model import get_user_summary, reset_user

# Sinhala teaching scripts — one per user, each pushing a different style.
# Every line contains an EXPLICIT style signal, so each turn is real
# evidence the model is allowed to learn from.
TEACHING = {
    "user_001": [
        "සරල කරන්න",                                   # simplify
        "කරුණාකර මෙය ඉතාමත් සරලව පහදා දෙන්න",          # explain very simply
        "කෙටියෙන් කියන්න",                              # say briefly
    ],
    "user_002": [
        "තව විස්තර කරන්න",                              # elaborate
        "මෙය තව ටිකක් විස්තර කරන්න පුළුවන්ද",           # more detail please
        "සම්පූර්ණ විස්තරය දෙන්න",                       # give full detail
    ],
    "user_003": [
        "පියවරෙන් පියවර පහදන්න",                        # explain step by step
        "එකින් එක කියන්න",                              # tell me one by one
        "පියවර වශයෙන් විස්තර කරන්න",                    # describe in steps
    ],
}

# One neutral question, asked to everyone. Deliberately contains NO style
# words, so style_from_intent() and style_from_flags() both return None and
# the answer must come from each user's learned profile.
NEUTRAL_QUESTION = "මෙහි ඇත්තේ කුමක්ද"   # "what is in this?"


def teach(user_id, lines):
    print(Fore.CYAN + f"\n── Teaching {user_id} " + "─" * (46 - len(user_id)))
    for line in lines:
        result = handle_voice_command(line, user_id)
        pers = result["personalization_stage"]
        mark = Fore.GREEN + "learned" if pers["learned"] else Fore.LIGHTBLACK_EX + "not learned"
        print(f"  {line}")
        print(f"    → {pers['style_class']}  "
              f"({pers['style_source']}, {mark}{Style.RESET_ALL})")


def compare():
    print(Fore.YELLOW + Style.BRIGHT +
          f"\n\nAsking all three users the SAME neutral question:")
    print(Fore.YELLOW + f'  "{NEUTRAL_QUESTION}"  (no style words — rules stay silent)\n')

    rows = []
    for user_id in TEACHING:
        result = handle_voice_command(NEUTRAL_QUESTION, user_id)
        pers = result["personalization_stage"]
        final = result["final_prompt"]
        prof = get_user_summary(user_id)
        rows.append({
            "user_id": user_id,
            "signals": prof["n_confirmed"],
            "learned_preference": prof["dominant_preference"] or "-",
            "style_class": final["style_class"],
            "decided_by": pers["style_source"],
            "prompt_modifier": final["prompt_modifier"],
            "final_prompt": final,
        })
    return rows


def print_table(rows):
    print(Fore.CYAN + "═" * 100)
    print(Fore.CYAN + Style.BRIGHT +
          f"{'User':<11}{'Signals':<9}{'Learned pref':<15}{'Style':<13}{'Decided by':<18}")
    print(Fore.CYAN + "═" * 100)
    for r in rows:
        print(f"{r['user_id']:<11}{r['signals']:<9}{r['learned_preference']:<15}"
              f"{r['style_class']:<13}{r['decided_by']:<18}")
    print(Fore.CYAN + "═" * 100)

    print(Fore.YELLOW + Style.BRIGHT + "\nPersonalized prompt sent to Component 3 (RAG):\n")
    for r in rows:
        print(Fore.WHITE + Style.BRIGHT + f"  {r['user_id']} → {r['style_class']}")
        print(Fore.LIGHTBLACK_EX + f"    {r['prompt_modifier']}\n")


def write_html(rows, path):
    cells = ""
    for r in rows:
        cells += (
            f"<tr><td><b>{r['user_id']}</b></td><td>{r['signals']}</td>"
            f"<td>{r['learned_preference']}</td><td><b>{r['style_class']}</b></td>"
            f"<td>{r['decided_by']}</td><td>{r['prompt_modifier']}</td></tr>"
        )
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Component 4 — Per-user personalization</title>
<style>
body{{font-family:system-ui,sans-serif;margin:32px;color:#222}}
h1{{font-size:20px}} p{{color:#555;font-size:14px}}
table{{border-collapse:collapse;width:100%;font-size:14px;margin-top:16px}}
th,td{{border:1px solid #ddd;padding:8px 10px;text-align:left;vertical-align:top}}
th{{background:#f4f4f2}}
code{{background:#f4f4f2;padding:2px 5px;border-radius:3px}}
</style></head><body>
<h1>Component 4 — Adaptive personalization across users</h1>
<p>All three users were asked the identical question <code>{NEUTRAL_QUESTION}</code>,
which contains no style words. The rule layer stays silent, so the style below
comes entirely from each user's own learned profile.</p>
<table>
<tr><th>User</th><th>Confirmed signals</th><th>Learned preference</th>
<th>Style</th><th>Decided by</th><th>Prompt modifier sent to RAG</th></tr>
{cells}
</table></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(Fore.GREEN + f"\nHTML table written to {path}")


def main():
    args = sys.argv[1:]

    if "--reset" in args:
        for user_id in TEACHING:
            reset_user(user_id)
        print(Fore.YELLOW + "Profiles reset for all three users.")

    print(Fore.YELLOW + Style.BRIGHT + "\nPHASE 1 — Teaching each user a different preference")
    print(Fore.LIGHTBLACK_EX + "(every line below is explicit user evidence, learned live)")
    for user_id, lines in TEACHING.items():
        teach(user_id, lines)

    print(Fore.YELLOW + Style.BRIGHT + "\n\nPHASE 2 — Same question, three users")
    rows = compare()
    print_table(rows)

    if "--html" in args:
        idx = args.index("--html")
        path = args[idx + 1] if len(args) > idx + 1 else "comparison.html"
        write_html(rows, path)


if __name__ == "__main__":
    main()
