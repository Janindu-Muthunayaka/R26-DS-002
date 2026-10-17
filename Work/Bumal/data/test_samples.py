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
        "expected_intent": "Simplify",
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
    }
]