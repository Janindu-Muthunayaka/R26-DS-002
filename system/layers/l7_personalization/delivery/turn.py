# delivery/turn.py
# One user turn of the standalone app, from the Sinhala STT text to what is
# spoken back (ADAPTIVE_VOICE_PLAN.md, "Order in one turn", Phase 3):
#
#   1. A "continue?" check-in is pending -> yes = play the rest, no = stop,
#      anything else = drop it and treat the reply as a new request.
#      The speed question is pending -> apply_speed_answer(), or drop it.
#   2. Short Sinhala utterance (<= 3 words) with a repeat or speed word and no
#      style word -> handled here, without the slow NLLB + Llama call.
#   3. Otherwise handle_voice_command() runs unchanged.
#   4. feedback.apply_turn() reads the frame (learning + what to do).
#   5. Route:
#        delivery - REPEAT (replay the last spoken text), slower / faster with
#                   no new content, the speed question
#        content  - a new answer from Component 3 (mock) at the current settings
#        both     - a new answer AND a speed change ("explain simply, slowly")
#        system   - navigation commands (next / previous article, ...)
#      Every answer is planned, played, saved as last_spoken, and how its
#      playback ended goes to feedback.apply_playback().
#
# Slower / faster with no repeat word (decision 15 Oct):
#   - last answer stopped mid-way -> continue from the first segment not heard,
#     at the new speed;
#   - last answer finished -> no replay; say "හරි, හෙමින් කියන්නම්" /
#     "හරි, ඉක්මනින් කියන්නම්" and use the new speed from the next answer.
#   "Again + slower" is a REPEAT: the whole answer again at the new speed.
#
# Check-in endings ("no", or a new request instead of an answer) do not go to
# apply_playback(): the user chose to stop at a point we offered, which is
# neither a delivery complaint nor a clean finish.
#
# The intent pipeline, Component 3 and the player are passed in, so this file
# loads no model and plays no sound by itself (tests use fakes).

from datetime import datetime

try:
    from delivery.feedback import apply_playback, apply_speed_answer, apply_turn, yes_no
    from delivery.planner import plan_speech
    from delivery.voice_profile import (
        count_turn, get_profile, log_delivery, repeatable, save_profile, set_last_spoken,
    )
    from intent_detection.modifiers import build_frame
except ImportError:
    from layers.l7_personalization.delivery.feedback import apply_playback, apply_speed_answer, apply_turn, yes_no
    from layers.l7_personalization.delivery.planner import plan_speech
    from layers.l7_personalization.delivery.voice_profile import (
        count_turn, get_profile, log_delivery, repeatable, save_profile, set_last_spoken,
    )
    from layers.l7_personalization.intent_detection.modifiers import build_frame

FAST_PATH_MAX_WORDS = 3
# Intents that carry no new content when a slower/faster word is present:
# "හෙමින් ලිපිය කියවන්න" (READ_ALOUD + slower) must not start a new answer.
DELIVERY_ONLY_INTENTS = {"CHANGE_SPEED", "READ_ALOUD", "UNKNOWN", "REPEAT"}

# Short Sinhala system messages: spoken, never saved as last_spoken.
SPEED_OK_SI = {"slower": "හරි, හෙමින් කියන්නම්", "faster": "හරි, ඉක්මනින් කියන්නම්"}
CONTINUE_QUESTION_SI = "දිගටම කියන්නද?"           # check-in: shall I go on?
CHECKIN_NO_SI = "හරි"
NOTHING_TO_REPEAT_SI = "ආයෙ කියන්න දෙයක් නැහැ"
NEXT_ARTICLE_SI = "ඊළඟ ලිපිය. {title}"
PREVIOUS_ARTICLE_SI = "කලින් ලිපිය. {title}"
SYSTEM_STYLE = "Simple"                            # pause length for system messages


class TurnRunner:
    """
    handle_command(sinhala, user_id) -> handle_voice_command() result
    c3                               -> object with generate(final_prompt, article_index)
                                        and article(index) (delivery/mock_c3.MockComponent3)
    play(plan, start=0, end=None, on_segment=None) -> laptop_player.play_plan() result
    on_event(name, data)             -> optional, for the app's display
    """

    def __init__(self, user_id, handle_command, c3, play, on_event=None,
                 now=datetime.now, db_path=None):
        self.user_id = user_id
        self.handle_command = handle_command
        self.c3 = c3
        self.play = play
        self.on_event = on_event or (lambda name, data: None)
        self.now = now
        self.db_path = db_path
        self.profile = get_profile(user_id, db_path)
        self.pending = None          # {"kind": "checkin", ...} | {"kind": "speed_question"}
        self.article_index = 0

    # ── One turn ───────────────────────────────────────────────────────────

    def turn(self, sinhala):
        """Runs one user turn. Returns the delivery-log record (also saved)."""
        sinhala = " ".join(str(sinhala or "").split())
        now = self.now()
        rec = self._new_record(sinhala, now)
        pending, self.pending = self.pending, None

        # 1. Replies to our own questions
        if pending and pending["kind"] == "checkin":
            answer = yes_no(sinhala)
            if answer:
                return self._checkin_reply(answer, pending, rec)
            rec["reasons"].append("check-in dropped: the reply is a new request")
        if pending and pending["kind"] == "speed_question":
            profile, decision = apply_speed_answer(self.profile, sinhala, now=now)
            if decision:
                self.profile = profile
                self._note_decision(rec, decision)
                rec["route"] = "delivery"
                self._speed_only(decision["speed"], rec, now)
                return self._finish(rec)
            rec["reasons"].append("speed question dropped: the reply is a new request")

        # 2-3. The frame: fast path for short repeat / speed utterances
        frame, result = self._fast_path_frame(sinhala), None
        if frame:
            rec["fast_path"] = True
            self.on_event("fast_path", frame)
        else:
            result = self.handle_command(sinhala, self.user_id)
            frame = result["frame"]
            self.on_event("command_result", result)
        rec["intent"], rec["modifiers"] = frame["intent"], frame["modifiers"]

        # 4. Feedback rules on this turn
        self.profile, decision = apply_turn(self.profile, frame, now)
        self._note_decision(rec, decision)
        self.on_event("feedback", decision)

        # 5. Route
        mods = frame["modifiers"]
        if decision["repeat"]:
            rec["route"] = "delivery"
            if decision["replay"]:
                rec["action"] = "replay"
                self._play_item(decision["replay"], rec, replay=True)
            else:
                rec["action"] = "nothing_to_repeat"
                self._say(NOTHING_TO_REPEAT_SI, rec)
        elif decision["ask"]:
            rec["route"], rec["action"] = "delivery", "ask_speed"
            self._say(decision["ask"]["question"], rec)
            self.pending = {"kind": "speed_question"}
        elif mods["speed"] and (result is None or
                                (frame["intent"] in DELIVERY_ONLY_INTENTS and not mods["style"])):
            rec["route"] = "delivery"
            self._speed_only(mods["speed"], rec, now)
        elif result["final_prompt"]["route"] == "SYSTEM_COMMAND":
            rec["route"] = "system"
            self._system_command(result["final_prompt"]["command"], rec)
        else:
            rec["route"] = "both" if mods["speed"] else "content"
            rec["action"] = "answer"
            answer = self.c3.generate(result["final_prompt"], self.article_index)
            rec["article_id"] = answer["article_id"]
            self.on_event("answer", answer)
            self._play_item({"text": answer["answer_si"], "kind": "answer",
                             "style": answer["style_class"]}, rec)
        return self._finish(rec)

    # ── Routes ─────────────────────────────────────────────────────────────

    def _fast_path_frame(self, sinhala):
        if len(sinhala.split()) > FAST_PATH_MAX_WORDS:
            return None
        frame = build_frame(sinhala, "", "UNKNOWN")
        mods = frame["modifiers"]
        return frame if (mods["repeat"] or mods["speed"]) and not mods["style"] else None

    def _checkin_reply(self, answer, pending, rec):
        count_turn(self.profile)
        rec["route"], rec["action"] = "delivery", f"checkin_{answer}"
        if answer == "yes":
            self._play_item(pending["item"], rec, start=pending["next"],
                            listened_from=pending["listened_from"])
        else:
            rec["reasons"].append("check-in: no - the user ended the answer (no learning)")
            self._say(CHECKIN_NO_SI, rec)
        return self._finish(rec)

    def _speed_only(self, speed, rec, now):
        """Slower / faster with no repeat word and no new content."""
        last = repeatable(self.profile, now)
        if last and not last.get("completed", True) and last.get("resume_at") is not None:
            rec["action"] = "resume"
            rec["reasons"].append(f"last answer was stopped: continue from segment {last['resume_at']}")
            self._play_item(last, rec, start=last["resume_at"])
        else:
            rec["action"] = "confirm"
            rec["reasons"].append("last answer finished (or none): confirm, new speed from the next answer")
            self._say(SPEED_OK_SI[speed], rec)

    def _system_command(self, command, rec):
        rec["action"] = command
        if command in ("GO_FORWARD", "GO_BACK"):
            self.article_index += 1 if command == "GO_FORWARD" else -1
            article = self.c3.article(self.article_index)
            rec["article_id"] = article["id"]
            template = NEXT_ARTICLE_SI if command == "GO_FORWARD" else PREVIOUS_ARTICLE_SI
            self.on_event("system", {"command": command, "article": article})
            self._say(template.format(title=article["title"]), rec)
        else:
            self.on_event("system", {"command": command, "article": None})

    # ── Speaking ───────────────────────────────────────────────────────────

    def _play_item(self, item, rec, start=0, replay=False, listened_from=None):
        """
        Plans and plays item = {text, kind, style} from segment `start` up to
        the next check-in. Saves last_spoken; when the answer ends (finished or
        stopped by the user) its playback goes to apply_playback().
        """
        plan = plan_speech(item["text"], item["style"], self.profile, replay=replay)
        end = next((c for c in plan["checkpoints"] if c > start), None)
        listened_from = start if listened_from is None else listened_from
        rec.update(style=item["style"], rate_used=plan["rate"],
                   pause_scale=self.profile["pause_scale"], chunk_size=self.profile["chunk_size"],
                   n_segments=len(plan["segments"]), start=start)
        self.on_event("plan", {"plan": plan, "start": start, "end": end, "action": rec["action"]})

        result = self._safe_play(plan, rec, start=start, end=end,
                                 on_segment=lambda i, seg: self.on_event("segment", (i, seg)))
        if result is None:
            return
        rec.update(completed=result["completed"], stopped_at=result["stopped_at"],
                   stopped_in=result["stopped_in"], segments_played=result["segments_played"])
        self.on_event("playback", result)

        # Stamped with our own clock (= when playback ended), so the 30-minute
        # expiry and the turn times always use the same clock.
        set_last_spoken(self.profile, item["text"], item.get("kind", "answer"), item["style"],
                        self.now(), completed=result["completed"],
                        resume_at=result["segments_played"])

        if result["stopped_in"] == "checkpoint":
            self.pending = {"kind": "checkin", "item": dict(item),
                            "next": result["segments_played"], "listened_from": listened_from}
            self._say(CONTINUE_QUESTION_SI, rec)
            return

        playback = {"completed": result["completed"],
                    "n_segments": len(plan["segments"]) - listened_from}
        self.profile, decision = apply_playback(self.profile, playback)
        self._note_decision(rec, decision)
        self.on_event("playback_feedback", decision)

    def _say(self, text, rec):
        """A short system message at the user's current rate."""
        rec["said"].append(text)
        self.on_event("say", text)
        plan = plan_speech(text, SYSTEM_STYLE, self.profile)
        self._safe_play(plan, rec)

    def _safe_play(self, plan, rec, **kwargs):
        try:
            return self.play(plan, **kwargs)
        except Exception as e:   # gTTS network errors, missing ffplay, ...
            rec["errors"].append(f"{type(e).__name__}: {e}")
            self.on_event("error", e)
            return None

    # ── Record ─────────────────────────────────────────────────────────────

    def _new_record(self, sinhala, now):
        return {"user_id": self.user_id, "timestamp": now.isoformat(timespec="seconds"),
                "sinhala_input": sinhala, "fast_path": False, "intent": None,
                "modifiers": None, "route": None, "action": None, "channels": [],
                "reasons": [], "changes": {}, "said": [], "errors": [],
                "article_id": None, "style": None, "rate_used": None, "pause_scale": None,
                "chunk_size": None, "n_segments": None, "start": None, "completed": None,
                "stopped_at": None, "stopped_in": None, "segments_played": None}

    @staticmethod
    def _note_decision(rec, decision):
        for c in decision["channels"]:
            if c not in rec["channels"]:
                rec["channels"].append(c)
        rec["reasons"].extend(decision["reasons"])
        for key, (old, new) in decision["changes"].items():
            first = rec["changes"].get(key, [old, new])[0]
            rec["changes"][key] = [first, new]

    def _finish(self, rec):
        rec["profile_after"] = {k: self.profile[k] for k in ("rate", "pause_scale", "chunk_size")}
        save_profile(self.user_id, self.profile, self.db_path)
        log_delivery(rec, self.db_path)
        return rec
