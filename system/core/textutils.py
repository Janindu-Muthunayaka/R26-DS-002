"""Text tidying shared by every OCR-consuming layer. Ported from Pipeline v9."""
import difflib, re, unicodedata


def norm(s: str) -> str:
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', s)).strip()


def collapse_repeats(line, sim=0.80):
    w = line.split(); n = len(w)
    if n < 3:
        return line
    for p in range(1, n // 2 + 1):
        if n % p == 0:
            blocks = [' '.join(w[k*p:(k+1)*p]) for k in range(n // p)]
            if all(difflib.SequenceMatcher(None, blocks[0], b).ratio() >= sim
                   for b in blocks[1:]):
                w = blocks[0].split(); n = len(w); break
    h = len(w) // 2
    if h > 0 and difflib.SequenceMatcher(
            None, ' '.join(w[:h]), ' '.join(w[h:2*h])).ratio() >= sim:
        w = w[:h] + w[2*h:]
    out = []
    for word in w:
        if out and word == out[-1]:
            continue
        out.append(word)
    return ' '.join(out)


def _pass(words, sim, mx):
    i, out = 0, []
    while i < len(words):
        hit = False
        for L in range(min(mx, (len(words) - i) // 2), 1, -1):
            a, b = words[i:i+L], words[i+L:i+2*L]
            if difflib.SequenceMatcher(None, ' '.join(a), ' '.join(b)).ratio() >= sim:
                out += a; i += 2*L; hit = True; break
        if not hit:
            out.append(words[i]); i += 1
    return out


def strong_dedup(text, sim=0.78, mx=8):
    stream = ' '.join(l for l in text.split('\n') if l.strip())
    prev = None
    while prev != stream:
        prev = stream
        stream = ' '.join(_pass(stream.split(), sim, mx))
    return collapse_repeats(stream)


SENT_END = re.compile(r'(?<=[.!?\u0964])\s+')


def sentences(text, max_chars=280):
    """Split text into units for correction.

    mT5 was trained on SENTENCES and generates with max_length=128 tokens.
    Handing it a whole article silently truncates the output — the tail is
    simply never produced, with no error. Research_Summary_Project_Knowledge
    Section 12.5 records this being found and fixed once already
    ("corrected text truncated to ~half ... fixed by correcting
    sentence-by-sentence").

    It came back because strong_dedup() joins every line into ONE string, so
    a caller that splits on newlines to get sentences gets a single item and
    makes a single call. Splitting on sentence terminators instead of on
    newlines makes correction independent of whatever dedup did to the
    layout.

    Over-long sentences (OCR often loses the full stop) are split on a space
    below max_chars, so no unit can silently exceed the generation budget.
    """
    out = []
    for part in SENT_END.split(text or ''):
        part = part.strip()
        while len(part) > max_chars:
            cut = part.rfind(' ', 0, max_chars)
            if cut <= 0:
                cut = max_chars
            head, part = part[:cut].strip(), part[cut:].strip()
            if head:
                out.append(head)
        if part:
            out.append(part)
    return out


def vote_lines(texts):
    """Medoid per line across frames — multi-frame consensus."""
    seqs = [t.split('\n') for t in texts if t.strip()]
    if not seqs:
        return ''
    out = []
    for k in range(max(len(s) for s in seqs)):
        cands = [s[k] for s in seqs if k < len(s) and s[k].strip()]
        if not cands:
            continue
        best, bs = cands[0], -1.0
        for c in cands:
            sc = sum(difflib.SequenceMatcher(None, c, d).ratio() for d in cands)
            if sc > bs:
                bs, best = sc, c
        out.append(best)
    return '\n'.join(out)
