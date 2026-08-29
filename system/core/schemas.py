"""
THE CONTRACT between layers. Change nothing here without telling the team —
every layer depends on these shapes.

Flow:
    Frame[]  -> L2 select  -> Frame[]        (best frames only)
             -> L3 segment -> Article[]      (boxes + region crops)
             -> L4A titles -> Article.title  (OTHER MEMBER)
             -> L4B body   -> Article.body   (Component 2, correction)
             -> L5 assemble-> Document
             -> L6 RAG/TTS -> audio          (OTHER MEMBER)
"""
from __future__ import annotations
from typing import Optional, List, Literal
from pydantic import BaseModel, Field


class Box(BaseModel):
    """Pixel box on a frame, top-left origin."""
    x1: float; y1: float; x2: float; y2: float

    @property
    def w(self) -> float: return self.x2 - self.x1
    @property
    def h(self) -> float: return self.y2 - self.y1


class Frame(BaseModel):
    """One captured photo plus the quality measurements from Layer 2."""
    path: str
    width: int
    height: int
    sharpness: Optional[float] = None
    # ADDITIVE 20 Aug 2026 — glyph_p75 is the capture metric Layer 2 now
    # writes. glyph_p90 is kept so nothing downstream breaks, but it is a
    # DIFFERENT measurement (OCR resize target on a region crop, not a
    # capture gate on a frame). See core/imaging.py.
    glyph_p75: Optional[float] = Field(
        None, description='p75 connected-component height on the whole '
                          'frame, px — the capture gate metric')
    glyph_p90: Optional[float] = Field(
        None, description='p90 connected-component height, px — OCR resize '
                          'target metric, not the capture gate')
    verdict: Literal['ok', 'warn', 'reject', 'unknown'] = 'unknown'
    note: str = ''


class Region(BaseModel):
    """A labelled sub-area inside an article (from layout detection)."""
    box: Box
    label: Literal['title', 'text', 'image', 'other'] = 'text'


class Article(BaseModel):
    """One article. Layers fill in different fields; none overwrite another's."""
    index: int                                  # reading order, 0-based
    box: Box
    regions: List[Region] = []

    # ---- Layer 4A writes these (OTHER MEMBER) ----
    title_raw: str = ''
    title: str = ''

    # ---- Layer 4B writes these (Component 2 — correction) ----
    body_raw: str = ''                          # OCR output, uncorrected
    body: str = ''                              # after mT5 correction

    # ---- Layer 4C writes this (optional LLM post-edit, default OFF) ----
    # SEPARATE FIELD ON PURPOSE. `body` is the research artifact — the mT5
    # output Chapter 4's CER is measured on — and nothing may overwrite it.
    # What gets SPOKEN is decided by l5_assemble.payload.article_text().
    body_polished: str = ''

    # ---- diagnostics, any layer may set ----
    glyph_p75: Optional[float] = None       # capture metric, whole frame
    glyph_p90: Optional[float] = None       # OCR-target metric, region crop
    ocr_scale: Optional[float] = None
    verdict: Literal['ok', 'warn', 'reject', 'unknown'] = 'unknown'
    note: str = ''


class Document(BaseModel):
    """Layer 5 output — the whole page, ready for RAG/TTS."""
    source_frames: List[str] = []
    articles: List[Article] = []
    timings: dict = {}
    warnings: List[str] = []


class CaptureResponse(BaseModel):
    """What the phone app receives.

    ADDITIVE 21 Aug 2026. `document` and `audio_url` are unchanged. The flat
    `title`/`body`/`warnings` fields were added because the phone speaks the
    TEXT with its own TTS rather than playing server-rendered audio:

      * l6_speech.speak() returns None — that layer is Bumal's and is a stub,
        so there is no audio to send. Waiting for it would block the demo.
      * The build record section 14 argues for this anyway: on-device TTS is
        lower latency, sends far less data, and speaks better Sinhala than an
        offline server engine.

    `audio_url` stays in the contract so that when Layer 6 lands, the phone can
    prefer it without another API change.
    """
    document: Document
    audio_url: Optional[str] = None
    error: Optional[str] = None

    # ---- flat fields the phone reads ----
    ok: bool = True
    title: str = ''          # EMPTY until Layer 4A is delivered
    body: str = ''
    warnings: List[str] = []
    n_articles: int = 0


# ==========================================================================
# THE SECOND DOOR  —  `POST /ask`
# ==========================================================================
# ADDITIVE. Nothing above this line changes shape.
#
#     Question -> L0 voice   (Bumal)  -> route / intent / style
#              -> L6 generator (Nadee) -> answer_si
#              -> Answer -> phone speaks `speakable`
#
# THE RULE FOR THE PHONE: speak `speakable` whenever it is non-empty, whatever
# `ok` says. `ok` records whether an answer was actually generated — it drives
# logging, not speech. Silence is the one unacceptable outcome.


class Question(BaseModel):
    """What the phone posts to `/ask`.

    `text` is Sinhala, already transcribed. Speech-to-text runs on the phone
    (`android.speech.SpeechRecognizer`, si-LK): lower latency, no audio over
    the network, one fewer server model. An `audio` field can be added later
    without changing anything else here.
    """
    job: str                       # the id `/capture` returned
    text: str = ''
    user_id: Optional[str] = None  # defaults to config.VOICE_USER_ID


class Answer(BaseModel):
    """What `/ask` returns. The phone reads `speakable` and nothing else."""
    ok: bool = True                       # was an answer actually generated
    job: str = ''
    route: str = ''                       # GENERATE | TTS_REPLAY | LOCAL | SYSTEM_COMMAND
    intent: str = ''
    speakable: str = ''                   # <-- the phone speaks THIS
    answer_si: str = ''                   # the generated answer alone
    english_translation: str = ''
    style_class: str = ''
    user_profile: Optional[dict] = None
    sources: List[dict] = []              # retrieval provenance, for the log
    warnings: List[str] = []
    timings: dict = {}
    error: Optional[str] = None
