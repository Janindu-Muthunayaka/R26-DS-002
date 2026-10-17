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


def _key(s):
    return ''.join(s.split())


def vote_lines(texts, ratio=0.60, window=4):
    """Medoid per line across frames — multi-frame consensus.

    ALIGNS THE FRAMES BY CONTENT FIRST. The previous version voted by line
    INDEX: candidates for output line k were `seq[k]` from every frame. That
    is only correct while every frame produces the same number of lines in the
    same order, and OCR does not.

    Measured on work/80654199, three frames of one static scene, same crop:
    the frames produced 105, 106 and 101 lines. From the first divergence
    onward, index k in one frame was a different physical line in another, and
    the medoid picked between unrelated candidates. Result: **15 of 100 output
    lines were near-duplicates of an earlier line** — whole passages spoken
    twice ("prices...", "the main commercial complex...") — and 235 characters
    lost. With content alignment: 0 repeats, 3729 characters against 3494.

    That is the "repeated passage strong_dedup cannot span" open item. It was
    never a dedup problem; it was the voter.

    HOW: the frame of MEDIAN length is the reference — not the longest, since
    a frame that split lines produces more lines, not better ones. For each of
    its lines, each other frame contributes its best match within +/- `window`
    lines, and only if the similarity clears `ratio`. The reference's order and
    line count are therefore preserved exactly, and the other frames can only
    correct a line, never insert or reorder one.

    A line no other frame matches is kept as-is: two frames disagreeing about
    whether a line exists is not evidence for dropping it.
    """
    seqs = [[l for l in t.split('\n') if l.strip()] for t in texts if t.strip()]
    if not seqs:
        return ''
    if len(seqs) == 1:
        return '\n'.join(seqs[0])

    # REFERENCE = the frame most like the others, not simply the median
    # length. Length alone picks a corrupted frame as often as a good one,
    # and a corrupted reference cannot be out-voted: nothing matches its bad
    # line, so it has no competition and survives. The medoid frame is the
    # one the others agree with, which is exactly the property wanted.
    keys = [_key('\n'.join(s))[:4000] for s in seqs]
    scores = [sum(difflib.SequenceMatcher(None, k, j).ratio() for j in keys)
              for k in keys]
    best = max(range(len(seqs)), key=lambda i: (scores[i], -abs(
        len(seqs[i]) - sorted(len(x) for x in seqs)[len(seqs) // 2])))
    ref = seqs[best]
    others = [s for i, s in enumerate(seqs) if i != best]

    out = []
    for i, r in enumerate(ref):
        cands = [r]
        rk = _key(r)
        for o in others:
            best, best_score = None, ratio
            for j in range(max(0, i - window), min(len(o), i + window + 1)):
                sc = difflib.SequenceMatcher(None, rk, _key(o[j])).ratio()
                if sc > best_score:
                    best_score, best = sc, o[j]
            if best is not None:
                cands.append(best)
        # medoid: the candidate closest to all the others
        pick, pick_score = cands[0], -1.0
        for c in cands:
            sc = sum(difflib.SequenceMatcher(None, c, d).ratio() for d in cands)
            if sc > pick_score:
                pick_score, pick = sc, c
        out.append(pick)
    return '\n'.join(out)
