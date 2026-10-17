# tts_clean.py
"""
Make generated text safe to hand to a text-to-speech engine.

Chat models write markdown by habit: `**subject**`, `# heading`, `- item`,
`[link](url)`. A screen shows that as formatting; a TTS engine reads it aloud
("star star subject star star"). The prompt asks for plain spoken text, but
prompting cannot guarantee it, so every generated answer passes through
`make_speakable` before it is spoken.

Deterministic, stdlib only, no network. It removes formatting and symbols and
never changes words: Sinhala letters, vowel signs, ZWJ (needed for
ශ්‍රී-style conjuncts), digits and sentence punctuation are kept as they are.
"""
import re
import unicodedata

# Inline markup: keep the text, drop the markers.
_FENCE = re.compile(r'```[^\n]*\n?')
_INLINE_CODE = re.compile(r'`([^`]*)`')
_IMAGE = re.compile(r'!\[([^\]]*)\]\([^)]*\)')
_LINK = re.compile(r'\[([^\]]+)\]\([^)]*\)')
_URL = re.compile(r'(?:https?://|www\.)\S+')
_BOLD = re.compile(r'(\*\*|__)(?=\S)(.+?)(?<=\S)\1')
_ITALIC_STAR = re.compile(r'(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])')
_ITALIC_UND = re.compile(r'(?<![\w_])_(?=\S)([^_\n]+?)(?<=\S)_(?![\w_])')
_STRIKE = re.compile(r'~~(.+?)~~')

# Line-level markup.
_HEADING = re.compile(r'^\s{0,3}#{1,6}\s*')
_QUOTE = re.compile(r'^\s*(?:>\s*)+')
_BULLET = re.compile(r'^\s*(?:[-*+•▪◦●–—]|\d{1,3}[.)])\s+')
_RULE = re.compile(r'^\s*(?:[-*_=]\s*){3,}$')
_TABLE_SEP = re.compile(r'^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$')

# Symbols a TTS engine would pronounce by name. Anything left after the
# markup passes is dropped.
_SPOKEN_SYMBOLS = re.compile(r'[*#_~`|<>^\\{}\[\]=]+')

_END_PUNCT = '.?!:;।'
_SPACES = re.compile(r'[ \t ]+')
_SPACE_BEFORE_PUNCT = re.compile(r'\s+([.,?!:;)])')
_DOUBLE_PUNCT = re.compile(r'([.,?!:;])(?:\s*[.,])+')


def _is_dropped_char(c: str) -> bool:
    """Emoji, pictographs and variation selectors: unspeakable, drop them.
    Format characters (Cf) such as ZWJ are kept, Sinhala needs them."""
    if c in '︎️':
        return True
    return unicodedata.category(c) in ('So', 'Cs', 'Co')


def make_speakable(text: str) -> str:
    """Return `text` as plain sentences a TTS engine can read aloud."""
    if not text:
        return ''

    t = text.replace('\r\n', '\n')
    t = _FENCE.sub('', t)
    t = _INLINE_CODE.sub(r'\1', t)
    t = _IMAGE.sub(r'\1', t)
    t = _LINK.sub(r'\1', t)
    t = _URL.sub('', t)
    for _ in range(2):                      # nested ***bold italic***
        t = _BOLD.sub(r'\2', t)
        t = _ITALIC_STAR.sub(r'\1', t)
        t = _ITALIC_UND.sub(r'\1', t)
    t = _STRIKE.sub(r'\1', t)

    lines = []
    for line in t.split('\n'):
        if _RULE.match(line) or _TABLE_SEP.match(line):
            continue
        line = _HEADING.sub('', line)
        line = _QUOTE.sub('', line)
        line = _BULLET.sub('', line)
        if '|' in line:                     # table row -> comma-separated cells
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            line = ', '.join(c for c in cells if c)
        line = _SPOKEN_SYMBOLS.sub(' ', line)
        line = ''.join(c for c in line if not _is_dropped_char(c))
        line = _SPACES.sub(' ', line).strip()
        if line:
            lines.append(line)

    # Headings and list items have no full stop; without one the TTS runs
    # them into the next line with no pause.
    out = ' '.join(l if l[-1] in _END_PUNCT else l + '.' for l in lines)
    out = _SPACE_BEFORE_PUNCT.sub(r'\1', out)
    out = _DOUBLE_PUNCT.sub(r'\1', out)
    return _SPACES.sub(' ', out).strip()
