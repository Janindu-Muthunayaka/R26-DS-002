# contracts.py

# Real OCR shape — from your OCR_INPUT.txt, the "council agreement" sentence



SAMPLE_OCR_INPUT = {
  "corrected_text": "පාස්කු දින ප්‍රහාරයේ මහ මොළකරු රාජපක්ෂ කෙනෙක් නම් එහි සත්‍යතාව රටට හෙළි කිරීමට ඊබ්‍රාහිම්ට අවස්ථාව තිබෙන බව ශ්‍රී ලංකා පොදුජන පෙරමුණේ ජාතික සංවිධායක, පාර්ලිමේන්තු මන්ත්‍රී නාමල් රාජපක්ෂ මහතා පැවසීය.\n\nපක්ෂ මූලස්ථානයේ පැවැති සාකච්ඡාවකින් අනතුරුව මාධ්‍ය වෙත අදහස් දක්වමින් නාමල් රාජපක්ෂ මහතා මේ බව කියා සිටියේය.\n\nනාමල් රාජපක්ෂ මහතා වැඩිදුරටත් මෙසේද පැවසීය.\n\n“පාස්කු දින ප්‍රහාරය තනිකරම දේශපාලනීකරණය කර තිබෙන්නේ. එක පැත්තකින් හිටපු ඇමැතිවරයකු වන පිල්ලෙයාන් අත්අඩංගුවට ගත්තා පාස්කු ප්‍රහාරයට. විෂය භාර ඇමැතිවරයා පාර්ලිමේන්තු ඇවිත් කිව්වා ඔහු අනාවරණය කර තියෙනවා තොරතුරු රාශියක් කියලා. එහෙත් ඔහු උසාවියට ඉදිරිපත් කරන්නේ වෙනත් ඝාතනයක් සම්බන්ධයෙන්. ඒ වගේම සුරේෂ් සලේ හිටපු බුද්ධි අංශ ප්‍රධානියා අත්අඩංගුවට අරගෙන ඔහුගෙන් ප්‍රශ්න කරනවා පාස්කු ඉරු දින ප්‍රහාරය ගැන. හැබැයි ඔහුගෙන් ප්‍රශ්න කරන්නේ පාස්කු ඉරිදා ප්‍රහාරයේ බුද්ධි අංශ වාර්තා ලාච්චුවේ හංගගෙන හිටපු දෙන්නා.",
  "tokens": [
    {
      "original": "පාස්කු",
      "corrected": "පාස්කු",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "දින",
      "corrected": "දින",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "ප්‍රහා­රයේ",
      "corrected": "ප්‍රහාරයේ",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "මහ",
      "corrected": "මහ",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "මොළ­කරු",
      "corrected": "මොළකරු",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "රාජ­පක්ෂ",
      "corrected": "රාජපක්ෂ",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "කෙනෙක්",
      "corrected": "කෙනෙක්",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "නම්",
      "corrected": "නම්",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "එහි",
      "corrected": "එහි",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "සත්‍ය­තාව",
      "corrected": "සත්‍යතාව",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "රටට",
      "corrected": "රටට",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "හෙළි",
      "corrected": "හෙළි",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "කිරී­මට",
      "corrected": "කිරීමට",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "ඊබ්‍රා­හිම්ට",
      "corrected": "ඊබ්‍රාහිම්ට",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "අව­ස්ථාව",
      "corrected": "අවස්ථාව",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "තිබෙන",
      "corrected": "තිබෙන",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "බව",
      "corrected": "බව",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "ශ්‍රී",
      "corrected": "ශ්‍රී",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "ලංකා",
      "corrected": "ලංකා",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "පොදු­ජන",
      "corrected": "පොදුජන",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "පෙර­මුණේ",
      "corrected": "පෙරමුණේ",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "ජාතික",
      "corrected": "ජාතික",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "සංවි­ධා­යක,",
      "corrected": "සංවිධායක,",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "පාර්ලි­මේන්තු",
      "corrected": "පාර්ලිමේන්තු",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "මන්ත්‍රී",
      "corrected": "මන්ත්‍රී",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "නාමල්",
      "corrected": "නාමල්",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "රාජ­පක්ෂ",
      "corrected": "රාජපක්ෂ",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "මහතා",
      "corrected": "මහතා",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "පැව­සීය.",
      "corrected": "පැවසීය.",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "පක්ෂ",
      "corrected": "පක්ෂ",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "මූල­ස්ථා­නයේ",
      "corrected": "මූලස්ථානයේ",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "පැවැති",
      "corrected": "පැවැති",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "සාක­ච්ඡා­ව­කින්",
      "corrected": "සාකච්ඡාවකින්",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "අන­තු­රුව",
      "corrected": "අනතුරුව",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "මාධ්‍ය",
      "corrected": "මාධ්‍ය",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "වෙත",
      "corrected": "වෙත",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "අද­හස්",
      "corrected": "අදහස්",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "දක්ව­මින්",
      "corrected": "දක්වමින්",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "නාමල්",
      "corrected": "නාමල්",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "රාජ­පක්ෂ",
      "corrected": "රාජපක්ෂ",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "මහතා",
      "corrected": "මහතා",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "මේ",
      "corrected": "මේ",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "බව",
      "corrected": "බව",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "කියා",
      "corrected": "කියා",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "සිටි­යේය.",
      "corrected": "සිටියේය.",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "නාමල්",
      "corrected": "නාමල්",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "රාජ­පක්ෂ",
      "corrected": "රාජපක්ෂ",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "මහතා",
      "corrected": "මහතා",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "වැඩි­දු­ර­ටත්",
      "corrected": "වැඩිදුරටත්",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "මෙසේද",
      "corrected": "මෙසේද",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "පැව­සීය.",
      "corrected": "පැවසීය.",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "“පාස්කු",
      "corrected": "“පාස්කු",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "දින",
      "corrected": "දින",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "ප්‍රහා­රය",
      "corrected": "ප්‍රහාරය",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "තනි­ක­රම",
      "corrected": "තනිකරම",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "දේශ­පා­ල­නී­ක­ර­ණය",
      "corrected": "දේශපාලනීකරණය",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "කර",
      "corrected": "කර",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "තිබෙන්නේ.",
      "corrected": "තිබෙන්නේ.",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "එක",
      "corrected": "එක",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "පැත්ත­කින්",
      "corrected": "පැත්තකින්",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "හිටපු",
      "corrected": "හිටපු",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "ඇමැ­ති­ව­ර­යකු",
      "corrected": "ඇමැතිවරයකු",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "වන",
      "corrected": "වන",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "පිල්ලෙ­යාන්",
      "corrected": "පිල්ලෙයාන්",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "අත්අ­ඩං­ගු­වට",
      "corrected": "අත්අඩංගුවට",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "ගත්තා",
      "corrected": "ගත්තා",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "පාස්කු",
      "corrected": "පාස්කු",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "ප්‍රහා­ර­යට.",
      "corrected": "ප්‍රහාරයට.",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "විෂය",
      "corrected": "විෂය",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "භාර",
      "corrected": "භාර",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "ඇමැ­ති­ව­රයා",
      "corrected": "ඇමැතිවරයා",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "පාර්ලි­මේන්තු",
      "corrected": "පාර්ලිමේන්තු",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "ඇවිත්",
      "corrected": "ඇවිත්",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "කිව්වා",
      "corrected": "කිව්වා",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "ඔහු",
      "corrected": "ඔහු",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "අනා­ව­ර­ණය",
      "corrected": "අනාවරණය",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "කර",
      "corrected": "කර",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "තියෙ­නවා",
      "corrected": "තියෙනවා",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "තොර­තුරු",
      "corrected": "තොරතුරු",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "රාශි­යක්",
      "corrected": "රාශියක්",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "කියලා.",
      "corrected": "කියලා.",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "එහෙත්",
      "corrected": "එහෙත්",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "ඔහු",
      "corrected": "ඔහු",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "උසා­වි­යට",
      "corrected": "උසාවියට",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "ඉදි­රි­පත්",
      "corrected": "ඉදිරිපත්",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "කරන්නේ",
      "corrected": "කරන්නේ",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "වෙනත්",
      "corrected": "වෙනත්",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "ඝාත­න­යක්",
      "corrected": "ඝාතනයක්",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "සම්බ­න්ධ­යෙන්.",
      "corrected": "සම්බන්ධයෙන්.",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "ඒ",
      "corrected": "ඒ",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "වගේම",
      "corrected": "වගේම",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "සුරේෂ්",
      "corrected": "සුරේෂ්",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "සලේ",
      "corrected": "සලේ",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "හිටපු",
      "corrected": "හිටපු",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "බුද්ධි",
      "corrected": "බුද්ධි",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "අංශ",
      "corrected": "අංශ",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "ප්‍රධා­නියා",
      "corrected": "ප්‍රධානියා",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "අත්අ­ඩං­ගු­වට",
      "corrected": "අත්අඩංගුවට",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "අර­ගෙන",
      "corrected": "අරගෙන",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "ඔහු­ගෙන්",
      "corrected": "ඔහුගෙන්",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "ප්‍රශ්න",
      "corrected": "ප්‍රශ්න",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "කර­නවා",
      "corrected": "කරනවා",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "පාස්කු",
      "corrected": "පාස්කු",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "ඉරු",
      "corrected": "ඉරු",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "දින",
      "corrected": "දින",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "ප්‍රහා­රය",
      "corrected": "ප්‍රහාරය",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "ගැන.",
      "corrected": "ගැන.",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "හැබැයි",
      "corrected": "හැබැයි",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "ඔහු­ගෙන්",
      "corrected": "ඔහුගෙන්",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "ප්‍රශ්න",
      "corrected": "ප්‍රශ්න",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "කරන්නේ",
      "corrected": "කරන්නේ",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "පාස්කු",
      "corrected": "පාස්කු",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "ඉරිදා",
      "corrected": "ඉරිදා",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "ප්‍රහා­රයේ",
      "corrected": "ප්‍රහාරයේ",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "බුද්ධි",
      "corrected": "බුද්ධි",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "අංශ",
      "corrected": "අංශ",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "වාර්තා",
      "corrected": "වාර්තා",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "ලාච්චුවේ",
      "corrected": "ලාච්චුවේ",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "හංග­ගෙන",
      "corrected": "හංගගෙන",
      "label": "ERROR",
      "confidence": 0.45,
      "was_changed": True
    },
    {
      "original": "හිටපු",
      "corrected": "හිටපු",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    },
    {
      "original": "දෙන්නා.",
      "corrected": "දෙන්නා.",
      "label": "CORRECT",
      "confidence": 0.95,
      "was_changed": False
    }
  ]
}
    
# Case 7 from sample_personalization_outputs.json — ELABORATE / Detailed / full.
# Should resolve to max_words=500 via resolve_max_words(), and anchor on chunk_article_2.
SAMPLE_VOICE_INPUT = {
    "route": "GENERATE",
    "intent": "EXPLAIN",
    "english_translation": "Break this down into numbered steps for me please.",
    "style_class": "StepByStep",
    "prompt_modifier": "Break the explanation into numbered steps.",
    "personalization_flags": {"detail_level": "structured"},
    "retrieved_chunk_id": "chunk_article_3",
    "correction_applied": None
  }

SAMPLE_REQUEST_CONTEXT = {
    "age_group": "10-13",   # sent by client every request; no storage/lookup needed
}

RAG_OUTPUT_SCHEMA = {
    "intent": None,
    "answer_si": None,
    "retrieved_sources": [],
    "speakable_text": None,
}