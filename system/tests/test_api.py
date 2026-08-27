"""The phone/server contract, tested without models or a phone.

This guards the mismatch that would have killed the demo: ReaderApi.kt
declared the response as raw audio bytes, backend/stub_server.py returned
audio, and app/server.py returned JSON. The app worked against the stub and
would have broken against the real server, and nothing in either codebase
said so.

So the field NAMES are asserted here. MainActivity reads them by name; if one
is renamed, this fails instead of the phone failing silently at the viva.

The pipeline is faked — these tests are about the contract, not the models.
"""
import io

import numpy as np
import pytest
from PIL import Image

fastapi = pytest.importorskip('fastapi')
from fastapi.testclient import TestClient      # noqa: E402

from core.schemas import Article, Box, Document, Region   # noqa: E402
from app.server import build, _document_to_reply          # noqa: E402

# every field MainActivity reads, by name
PHONE_FIELDS = ('ok', 'job', 'title', 'body', 'warnings', 'n_articles',
                'audio_url')


class FakePipeline:
    dev = 'cpu'

    def __init__(self, doc=None, boom=False):
        self.doc, self.boom = doc, boom

    def run(self, paths, **kw):
        if self.boom:
            raise RuntimeError('pipeline exploded')
        return self.doc


def _doc(body='කුරුණෑගල නගර සභාව', title='', warnings=()):
    box = Box(x1=0, y1=0, x2=100, y2=100)
    arts = []
    if body or title:
        arts.append(Article(index=0, box=box,
                            regions=[Region(box=box, label='text')],
                            body=body, title=title, verdict='ok'))
    return Document(articles=arts, warnings=list(warnings),
                    timings={'total': 1.0})


def _client(pipe, tmp_path):
    return TestClient(build(pipe, tmp_path))


def _jpeg(w=60, h=40):
    buf = io.BytesIO()
    Image.fromarray(np.full((h, w, 3), 230, np.uint8)).save(buf, format='JPEG')
    return buf.getvalue()


def _post(client, n=1, data=None):
    files = [('frames', (f'f{i}.jpg', data or _jpeg(), 'image/jpeg'))
             for i in range(n)]
    return client.post('/capture', files=files)


# ---- the contract --------------------------------------------------------
def test_capture_returns_every_field_the_phone_reads(tmp_path):
    r = _post(_client(FakePipeline(_doc()), tmp_path))
    assert r.status_code == 200
    j = r.json()
    missing = [k for k in PHONE_FIELDS if k not in j]
    assert not missing, f'MainActivity reads these and they are absent: {missing}'
    assert j['ok'] is True
    assert j['body']
    assert isinstance(j['warnings'], list)


def test_it_is_json_not_audio(tmp_path):
    """The historical bug, asserted directly."""
    r = _post(_client(FakePipeline(_doc()), tmp_path))
    assert r.headers['content-type'].startswith('application/json'), (
        'the phone parses JSON; returning audio here is the mismatch that '
        'made the app work against stub_server.py and fail against this one')


def test_failure_keeps_the_same_shape(tmp_path):
    """A blind user is told what went wrong. The phone can only speak an
    error if the error arrives in the fields it already reads."""
    r = _post(_client(FakePipeline(boom=True), tmp_path))
    assert r.status_code == 500
    j = r.json()
    assert j['ok'] is False and j['error']
    for k in ('title', 'body', 'warnings', 'n_articles'):
        assert k in j, f'{k} missing on the error path'


def test_undecodable_frames_are_rejected_clearly(tmp_path):
    r = _post(_client(FakePipeline(_doc()), tmp_path), data=b'not a jpeg')
    assert r.status_code == 400
    assert r.json()['ok'] is False


def test_nothing_read_is_not_reported_as_success(tmp_path):
    """Zero articles must not come back ok:true with an empty body — the app
    would speak silence and the user would not know why."""
    j = _post(_client(FakePipeline(_doc(body='')), tmp_path)).json()
    assert j['ok'] is False
    assert j['error']


def test_warnings_survive_to_the_phone(tmp_path):
    j = _post(_client(FakePipeline(
        _doc(warnings=['Article 2 skipped: glyph 19px — much closer'])),
        tmp_path)).json()
    assert any('closer' in w for w in j['warnings'])


def test_title_is_empty_until_layer_4a_lands(tmp_path):
    """Documents the boundary. Title OCR is another member's layer; the
    close-up path locates the headline but does not read it."""
    j = _post(_client(FakePipeline(_doc()), tmp_path)).json()
    assert j['title'] == ''


def test_title_passes_through_when_it_exists(tmp_path):
    j = _post(_client(FakePipeline(_doc(title='ශීර්ෂය')), tmp_path)).json()
    assert j['title'] == 'ශීර්ෂය'


def test_audio_url_is_present_but_null(tmp_path):
    """Layer 6 is Bumal's stub. The key must exist so the phone's fallback is
    exercised now, not discovered when it starts returning data."""
    j = _post(_client(FakePipeline(_doc()), tmp_path)).json()
    assert 'audio_url' in j and j['audio_url'] is None


def test_multiple_articles_are_joined_in_order(tmp_path):
    box = Box(x1=0, y1=0, x2=10, y2=10)
    doc = Document(articles=[
        Article(index=0, box=box, body='පළමු'),
        Article(index=1, box=box, body='දෙවන'),
    ], warnings=[], timings={})
    j = _post(_client(FakePipeline(doc), tmp_path)).json()
    assert j['body'].index('පළමු') < j['body'].index('දෙවන')
    assert j['n_articles'] == 2


def test_body_raw_is_used_when_correction_was_skipped(tmp_path):
    box = Box(x1=0, y1=0, x2=10, y2=10)
    doc = Document(articles=[Article(index=0, box=box, body_raw='අමු පෙළ')],
                   warnings=[], timings={})
    assert 'අමු පෙළ' in _document_to_reply(doc, 'j')['body']


def test_health_and_pages(tmp_path):
    c = _client(FakePipeline(_doc()), tmp_path)
    assert c.get('/health').json()['ok'] is True
    assert c.get('/').status_code == 200
    assert c.get('/audio/nosuchjob').status_code == 404


# ---- too far to identify an article -------------------------------------
def test_a_frame_too_far_to_segment_says_move_closer(tmp_path):
    """MEASURED over 70 real captures (tools/probe_yolo.py): where the article
    detector and the layout path both answered, they DISAGREED on 69% of
    frames. Reading a confidently-wrong story to someone who cannot check it
    is worse than reading nothing — and "move a little closer" is also the
    instruction that fixes the frame, so the next capture is correct.
    """
    doc = Document(articles=[], warnings=[
        'Could not identify a single article in this frame - move a little '
        'closer and try again'], timings={'mode': 'too-far'})
    j = _post(_client(FakePipeline(doc), tmp_path)).json()
    assert j['ok'] is False
    assert 'move a little closer' in j['error']
    assert j['body'], 'the user must hear an instruction, not silence'
    from app.server import SI_MOVE_CLOSER
    assert j['body'] == SI_MOVE_CLOSER


def test_a_genuinely_unreadable_frame_still_says_so(tmp_path):
    """The two failures are different and the user's next action differs."""
    j = _post(_client(FakePipeline(_doc(body='')), tmp_path)).json()
    assert j['ok'] is False
    assert 'move a little closer' not in j['error']


def test_the_detector_is_off_in_the_phone_path_by_default():
    from core.config import SEGMENT_MODE
    assert SEGMENT_MODE.lower() == 'off', (
        'the article detector disagreed with the layout crop on 69% of real '
        'captures; it must not silently come back')
