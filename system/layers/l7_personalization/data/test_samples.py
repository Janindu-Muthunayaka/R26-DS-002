# These are the exact STT outputs from your Google STT results
# Used to test both intent detection approaches

test_samples = [
    {
        "id": "sam_1",
        "stt_output": "මෙය සාරාංශ කරන්න",
        "expected_intent": "SUMMARIZE",
        "description": "Short and clear"
    },
    {
        "id": "sam_2",
        "stt_output": "මෙය තව ටිකක් විස්තර කරන්න පුළුවන්ද",
        "expected_intent": "EXPLAIN",
        "description": "Medium with hesitation"
    },
    {
        "id": "sam_3",
        "stt_output": "කරුණාකර මෙය ඉතාමත් සරලව සහ කෙටියෙන් මට තේරෙන විදිහට පහදා දෙන්න",
        "expected_intent": "EXPLAIN",
        "description": "Long and clear"
    },
    {
        "id": "sam_4",
        "stt_output": "ඔයා ඔයා මට මෙහි ඇත්තේ කුමක්දැයි කියන්න පුළුවන්ද",
        "expected_intent": "IDENTIFY_CONTENT",
        "description": "Thinking mid sentence"
    },
    {
        "id": "sam_5",
        "stt_output": "මට ඕනේ හ්ම් ඉතාමත් දිගු විස්තර නොමැතිව කෙටියෙන් වේගයෙන් මෙය පහදන්න",
        "expected_intent": "SIMPLIFY",
        "description": "Long with thinking pauses"
    },
    {
        "id": "sam_6",
        "stt_output": "ඒක හරියට තේරෙන්නේ නෑ වෙනත් විදිහකට කියන්න",
        "expected_intent": "REPHRASE",
        "description": "Unclear intent conversational"
    },
    {
        "id": "sam_7",
        "stt_output": "ඇත්තටම මට මේ සම්පූර්ණ ලියවිල්ලම අහගෙන ඉන්න වෙලාවක් නැහැ, ඒ නිසා මේකෙ තියෙන වැදගත්ම කරුණු ටික විතරක් තෝරලා අරගෙන මට පැහැදිලිව කියවන්න පුළුවන්ද",
        "expected_intent": "SUMMARIZE",
        "description": "Very long, conversational, includes reasoning/filler before the actual command"
    },

    # ─── Samples 8–50: newly written to match the style of sam_1–sam_7 ───
    # None of these appear in data/generate_samples.py, which was the
    # training data for Approach 3 and the prompt-tuning data for
    # Approach 1 — so no approach has seen them before.
    # Each intent has short, filler/hesitant, medium and long examples.
    # Some deliberately contain words that point at another intent
    # (e.g. "again" in a REPHRASE, "enough" in a NEXT) to test confusion.

    # SUMMARIZE
    {"id": "sam_8",  "stt_output": "මේ පිටුවේ තියෙන දේ කෙටියෙන් සාරාංශයක් විදිහට දෙන්න", "expected_intent": "SUMMARIZE", "description": "Medium and clear"},
    {"id": "sam_9",  "stt_output": "අ මට ඕනෙ මේකෙ ප්‍රධාන අදහස් ටික විතරයි", "expected_intent": "SUMMARIZE", "description": "Filler at start"},
    {"id": "sam_10", "stt_output": "ඔක්කොම නෙවෙයි සාරාංශය විතරක් ඇති", "expected_intent": "SUMMARIZE", "description": "Short, contains 'enough' (ඇති)"},

    # EXPLAIN
    {"id": "sam_12", "stt_output": "මේ වචනෙ තේරුම මොකක්ද", "expected_intent": "EXPLAIN", "description": "Short question"},
    {"id": "sam_13", "stt_output": "ඔයා ඔයා කිව්ව දේ මට තේරුනේ නෑ ඒක පහදලා දෙන්න", "expected_intent": "EXPLAIN", "description": "Repeated word, thinking mid sentence"},
    {"id": "sam_14", "stt_output": "මේ ආර්ථික වචන ටිකේ අදහස පැහැදිලි කරලා දෙන්න", "expected_intent": "EXPLAIN", "description": "Medium and clear"},
    {"id": "sam_15", "stt_output": "මම මේක දෙපාරක් ඇහුවා ඒත් තාමත් ඔලුවට යන්නේ නෑ මේකෙ අදහස මට පැහැදිලි කරලා දෙන්න පුළුවන්ද", "expected_intent": "EXPLAIN", "description": "Long and conversational"},

    # SIMPLIFY
    {"id": "sam_16", "stt_output": "ලේසියෙන් කියන්න", "expected_intent": "SIMPLIFY", "description": "Short and clear"},
    {"id": "sam_17", "stt_output": "හ්ම් මේ ලොකු ලොකු වචන නැතුව සාමාන්‍ය විදිහට කියන්න", "expected_intent": "SIMPLIFY", "description": "Filler at start"},
    {"id": "sam_18", "stt_output": "මට මේ තාක්ෂණික වචන තේරෙන්නේ නෑ සරල භාෂාවෙන් කියන්න", "expected_intent": "SIMPLIFY", "description": "Medium with reason"},

    # REPHRASE
    {"id": "sam_21", "stt_output": "වෙනස් විදිහකට", "expected_intent": "REPHRASE", "description": "Short, incomplete phrase"},
    {"id": "sam_23", "stt_output": "ඔය විදිහට නෙවෙයි වෙන ක්‍රමයකට ඒක කියලා දෙන්න", "expected_intent": "REPHRASE", "description": "Medium and clear"},
    {"id": "sam_24", "stt_output": "මේ වාක්‍යය ටිකක් වෙනස් කරලා ආයෙ කියන්න", "expected_intent": "REPHRASE", "description": "Medium, contains 'again' (ආයෙ)"},
   

    # IDENTIFY_CONTENT
    {"id": "sam_26", "stt_output": "මේ මොකක්ද", "expected_intent": "IDENTIFY_CONTENT", "description": "Short question"},
    {"id": "sam_27", "stt_output": "හ්ම් මේ පිටුවෙ මොනවා ගැනද තියෙන්නෙ", "expected_intent": "IDENTIFY_CONTENT", "description": "Filler at start"},
    {"id": "sam_28", "stt_output": "මේ ලියවිල්ල මොන වගේ එකක්ද", "expected_intent": "IDENTIFY_CONTENT", "description": "Medium question"},
    {"id": "sam_29", "stt_output": "මගේ අතේ තියෙන මේ කොළේ මොනවද ලියලා තියෙන්නේ", "expected_intent": "IDENTIFY_CONTENT", "description": "Medium, real-world object"},
    {"id": "sam_30", "stt_output": "මම දැන් කැමරාව අල්ලගෙන ඉන්න මේ කොටසේ මොනවා ගැනද කතා කරන්නේ කියලා මට මුලින්ම කියන්න", "expected_intent": "IDENTIFY_CONTENT", "description": "Long and conversational"},

    # REPEAT
    {"id": "sam_31", "stt_output": "ආයෙ", "expected_intent": "REPEAT", "description": "Short, one word"},
    {"id": "sam_32", "stt_output": "අ අ නැවත කියන්න", "expected_intent": "REPEAT", "description": "Hesitant, repeated filler"},
    {"id": "sam_33", "stt_output": "කලින් කියපු කොටස තව පාරක් කියන්න", "expected_intent": "REPEAT", "description": "Medium, contains 'previous' (කලින්)"},
    {"id": "sam_34", "stt_output": "සද්දෙ නිසා මට ඇහුනෙ නෑ", "expected_intent": "REPEAT", "description": "Implicit request, no command verb"},
    {"id": "sam_35", "stt_output": "සමාවෙන්න මගේ ෆෝන් එක රින්ග් වුණා ඒ නිසා ඔයා අන්තිමට කියපු ටික මට මඟ ඇරුණා ඒක ආයෙත් කියන්න", "expected_intent": "REPEAT", "description": "Long with reasoning before the command"},

    # STOP
    {"id": "sam_36", "stt_output": "ඇති", "expected_intent": "STOP", "description": "Short, one word"},
    {"id": "sam_37", "stt_output": "හරි හරි නවත්තන්න", "expected_intent": "STOP", "description": "Filler at start"},
    {"id": "sam_38", "stt_output": "දැන් කියවන්න එපා", "expected_intent": "STOP", "description": "Negative form, contains 'read'"},
    {"id": "sam_39", "stt_output": "මට කෝල් එකක් එනවා ටිකකට නවත්තන්න", "expected_intent": "STOP", "description": "Medium with reason"},
    {"id": "sam_40", "stt_output": "අද නම් මට ඇති වෙන්න ඇහුවා ඉතුරු ටික හෙට අහන්නම් දැනට කියවීම නවත්තන්න", "expected_intent": "STOP", "description": "Long and conversational"},

    # NEXT
    {"id": "sam_41", "stt_output": "ඊළඟ", "expected_intent": "NEXT", "description": "Short, one word"},
    {"id": "sam_42", "stt_output": "අ ඊළඟ පුවතට", "expected_intent": "NEXT", "description": "Filler, incomplete phrase"},
    {"id": "sam_43", "stt_output": "අනිත් පිටුවට යන්න", "expected_intent": "NEXT", "description": "Medium, indirect wording"},
    {"id": "sam_44", "stt_output": "මේ පුවත ඇති ඊළඟ එක අහමු", "expected_intent": "NEXT", "description": "Medium, contains 'enough' (ඇති)"},
    {"id": "sam_45", "stt_output": "මේ පිටුවේ තියෙන ඔක්කොම මම අහලා ඉවරයි ඒ නිසා ඊළඟට තියෙන පිටුව පටන් ගන්න", "expected_intent": "NEXT", "description": "Long and conversational"},

    # READ_ALOUD
    {"id": "sam_46", "stt_output": "කියවන්න", "expected_intent": "READ_ALOUD", "description": "Short, one word"},
    {"id": "sam_47", "stt_output": "හ්ම් මේ ලිපිය මුල ඉඳන් අගට කියවලා දෙන්න", "expected_intent": "READ_ALOUD", "description": "Filler, contains 'from the beginning'"},
    {"id": "sam_48", "stt_output": "මේ පිටුවේ තියෙන දේ හඬ නඟලා කියවන්න", "expected_intent": "READ_ALOUD", "description": "Medium and clear"},
    {"id": "sam_49", "stt_output": "මගේ බෙහෙත් කොළේ තියෙන දේ කියවලා දෙන්න", "expected_intent": "READ_ALOUD", "description": "Medium, real-world object"},
    {"id": "sam_50", "stt_output": "මට ඇස් පේන්නෙ නැති නිසා මේ ලියුම මට කියවන්න බැහැ කරුණාකරලා ඒක මුලින්ම ඉඳන් කියවලා දෙන්න", "expected_intent": "READ_ALOUD", "description": "Long with personal reasoning"},
]
