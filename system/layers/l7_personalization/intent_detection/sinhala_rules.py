# intent_detection/sinhala_rules.py
#
# Sinhala command rules, checked BEFORE translation + LLM.
#
# WHY (18 Oct 2026). 556 commands sent to the running /interpret and replayed
# through the turn policy (system/tools/bench/replay_turns.py): most commands
# that went wrong for the user were short CONTROL commands the translation
# broke before the LLM saw them:
#   "පස්සට යන්න" (go back)       -> "Go back to the drawing board" -> REPHRASE -> RAG
#   "ඊළඟ එකට යන්න" (go to next)  -> "Next time we go together"     -> PLAN_OUTING -> RAG
#   "පෙර පිටුව" (previous page)  -> "Other articles"               -> IDENTIFY_CONTENT -> RAG
# A control command that reaches RAG is answered "not enough information".
#
# SCOPE. Only commands whose Sinhala is unambiguous: back, next, stop,
# rephrase, summary, headlines, a bare speed change. Content requests
# (explain, elaborate, simplify, a question) still go to the LLM: there the
# label decides the answer's style, and a wrong rule would be worse than the
# LLM. A rule returns None when it is not sure.
#
# Pure functions; importing this file loads no model.

import re

try:
    from intent_detection.modifiers import (_clean, chunk_request, navigation_move_si,
                                            pause_request, repeat_in_sinhala, speed_request)
except ImportError:
    from layers.l7_personalization.intent_detection.modifiers import (
        _clean, chunk_request, navigation_move_si, pause_request, repeat_in_sinhala, speed_request)

QUESTION_SI = r"(මොකක්ද|මොනවද|මොනවාද|කවුද|කොහෙද|ඇයි|කොහොමද|කීයද|කවදාද|කුමක්ද|කාටද|කොතැනද)"

# Whole words: "පෙරලන්න" (turn the page) starts with "පෙර" (previous).
BACK_SI = r"(?<!\S)(කලින්|පෙර|පෙරට|පස්සට|ආපස්සට|ආපහු|ආපසු)(?!\S)"
NEXT_SI = r"(ඊළඟ|ඊලඟ|මීළඟ|මීලඟ|ඉස්සරහට|මේකට පස්සේ|ඊට පස්සේ|වෙන පුවතක්|වෙනත් පුවතක්)"
# "Turn the page" = next, but "දේශපාලන පිටුව පෙරලන්න" (turn to the politics
# page) is a section: only a bare page turn counts.
PAGE_TURN_SI = r"^((තව|ඉස්සරහට) )?පිටුව(ක්)? ((ඉස්සරහට|තව) )?පෙරල"
NUMBER_SI = r"(\d+|එක|දෙක|තුන|හතර|පහ|හය|හත|අට|නවය|දහය)"
PAGE_NUMBER_SI = (r"(පිටු(ව|වට)? ?අංක ?" + NUMBER_SI + r"|අංක ?" + NUMBER_SI + r"\S* පිටුව"
                  r"|\d+ ?(වන|වෙනි|වැනි) පිටුව|පිටුව \d+)")
SAY_AGAIN_VERB_SI = r"(කියන්න|අහන්න|කියවන්න)( |$)"           # say / hear / read
STOP_SI = r"(^| )(නවත්වන්න|නවතන්න|නවත්තන්න|නවත්තපන්|නතර කරන්න|ඇති ඇති)( |$)"
REPHRASE_SI = r"(වෙනත්|වෙන|වෙනස්) (විදිහ|විදිය|ආකාර|වචන|ක්‍රම)"
# නවත්තලා / නවත්වලා is "with pauses" only when repeated ("නවත්තලා නවත්තලා");
# once, it is "having stopped": "ඔතනින් නවත්වලා තියන්න" = stop it there.
STOPPED_ONCE_SI = r"නවත්(තලා|වලා)"
STOPPED_TWICE_SI = r"නවත්(තලා|වලා) නවත්(තලා|වලා)"
SUMMARY_SI = r"(සාරාංශ|සංක්ෂේප|කෙටි කරලා|කෙටි කර)"
# Headlines, or "what else / what other topics are in the paper". A plain
# "මාතෘකා මොනවද" (what are the topics) is left to the LLM: it is also a
# question about the article.
HEADLINES_SI = (r"සිරස්තල\S* (මොනවද|මොනවාද|කියවන්න|කියන්න)"
                r"|(වෙනත්|වෙන) (මාතෘකා|පුවත්|මොනවද|මොනවාද)"
                r"|පත්තරේ\S* .*(මොනවද|මොනවාද)")
SPEED_ONLY_WORDS = {"කියන්න", "කියවන්න", "කතා", "කරන්න", "ටිකක්", "පොඩ්ඩක්", "තව", "තවත්",
                    "ටික", "චුට්ටක්", "වේගය", "වේගෙ", "වේගේ", "අඩු", "වැඩි", "කරලා"}
SPEED_WORD_SI = r"^(වේගෙන්|වේගයෙන්|ඉක්මනින්|ඉක්මනට|ඉක්මණින්|ඉක්මණට|හෙමින්|සෙමින්|හෙමිහිට)$"
STYLE_OR_CONTENT_SI = r"(සරල|ලේසි|විස්තර|පැහැදිලි|පියවරෙන්|තේරුම|කෙටියෙන්)"


FILLER_SI = r"((ම්)+|(හ්ම්)+|(අම්)+|(හ්)+|අහ්|ඔහ්|ආ+|ඒ+|එ+|හා+|ඉ+|උ+)"


def _has(pattern, si):
    return bool(re.search(pattern, si))


def rule_intent(sinhala):
    """A production intent label for an unambiguous Sinhala command, or None."""
    si = _clean(sinhala)
    if not si or not re.search(r"[඀-෿]", si):
        return None
    words = si.split()

    # Hesitation sounds only ("ම්ම් අහ්"): nothing was asked. UNKNOWN makes the
    # turn policy ask the user to say it again; the LLM labelled these
    # ASK_QUESTION and RAG answered "not enough information".
    if all(re.fullmatch(FILLER_SI, w) for w in words):
        return "UNKNOWN"

    # Change the wording / summarise: new words, whatever else is said.
    if _has(REPHRASE_SI, si):
        return "REPHRASE"
    if _has(SUMMARY_SI, si):
        return "SUMMARIZE"

    # "ඊළඟ එකට යන්න නෑ නෑ කලින් එකට" (not the next one, no, the previous one):
    # the LAST direction said is the one meant.
    back = [m.start() for m in re.finditer(BACK_SI, si)]
    nxt = [m.start() for m in re.finditer(NEXT_SI, si)] + \
          [m.start() for m in re.finditer(PAGE_TURN_SI, si)]
    question = _has(QUESTION_SI, si)

    # A page number is GO_TO_PAGE, even with "back": "පිටු අංක දෙකට ආපහු යන්න".
    if _has(PAGE_NUMBER_SI, si) and not question:
        return "GO_TO_PAGE"

    if (back or nxt) and not question and not _has(STYLE_OR_CONTENT_SI, si):
        last_back = max(back) if back else -1
        last_next = max(nxt) if nxt else -1
        if last_back > last_next:
            # "කලින් කියපු එක ආයේ කියන්න" (say again what you said before) is a
            # REPEAT; going back needs a movement word or a target.
            if navigation_move_si(si) or len(words) <= 3 or not _has(SAY_AGAIN_VERB_SI, si):
                if not repeat_in_sinhala(si) or navigation_move_si(si):
                    return "PREVIOUS"
        elif last_next > last_back:
            return "NEXT"

    # "Again" with no style / content word and no move: replay the last thing
    # said ("තේරුනේ නෑ නැවත හෙමින් කියන්න" = again, slowly).
    if repeat_in_sinhala(si) and not question and not _has(STYLE_OR_CONTENT_SI, si):
        return "REPEAT"

    # Pauses / smaller parts only ("නවත්තලා නවත්තලා කියන්න", with pauses):
    # a delivery change. The LLM read නවත්තලා (stopping) as STOP.
    pauses = pause_request(si, "") and (_has(STOPPED_TWICE_SI, si) or not _has(STOPPED_ONCE_SI, si))
    if (pauses or chunk_request(si, "")) and not _has(STOP_SI, si)             and not question and not _has(STYLE_OR_CONTENT_SI, si):
        return "CHANGE_SPEED"

    # Stop: a stop verb on its own. "නවත්තලා ... කියවන්න" (with pauses) is not
    # here: it is a pause modifier, and NEXT above already won if "next" was said.
    if _has(STOP_SI, si) and not question:
        return "STOP"

    # A bare speed change ("වේගෙන් කියන්න", say it faster): a delivery command,
    # never new content.
    if speed_request(si, "") and all(w in SPEED_ONLY_WORDS or re.match(SPEED_WORD_SI, w)
                                     for w in words):
        return "CHANGE_SPEED"

    # Headlines / topics on the page ("... ඒ කොටසට යන්න", go to that section,
    # is a section request, not headlines).
    if _has(HEADLINES_SI, si) and not re.search(r"(^| )(යන්න|යමු)( |$)", si):
        return "READ_HEADLINES"

    return None
