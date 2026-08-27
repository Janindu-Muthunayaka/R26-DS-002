"""svc-rag — Component 3, tested with a fake model.

api.openai.com cannot be reached from CI or from a train, and a service that
is only ever tested against a live paid API is a service that is not tested.
So `core.llm.embed` and `core.llm.chat` are replaced with deterministic
stand-ins: the embedder is a real bag-of-words vector, so retrieval either
finds the right chunk or genuinely does not.

What this file does NOT test is answer quality. That needs the real model and
a human reading Sinhala.
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip('numpy')
_RAG = Path(__file__).resolve().parent.parent.parent / 'services' / 'rag'
if str(_RAG) not in sys.path:
    sys.path.insert(0, str(_RAG))

import answer as A          # noqa: E402
from store import VectorStore   # noqa: E402

DIM = 64
ARTICLE = ('කුරුණෑගල නගර සභාවේ විවිධ සංවර්ධන ව්‍යාපෘති සහ කඩ කාමර බෙදා දීම '
           'පිළිබඳ සාකච්ඡාව ඊයේ පැවැත්විණි.')
OTHER = ('ක්‍රිකට් තරගාවලියේ අවසන් තරගය ලබන සතියේ කොළඹදී පැවැත්වේ. '
         'ක්‍රීඩකයන් පුහුණුවීම් ආරම්භ කර ඇත.')


def _vec(text):
    v = [0.0] * DIM
    for w in (text or '').split():
        v[hash(w) % DIM] += 1.0
    return v or [0.0] * DIM


@pytest.fixture
def rag(tmp_path, monkeypatch):
    """A store plus a fake model. Returns (store, calls)."""
    calls = {'chat': [], 'embed': 0}

    def fake_embed(texts, model=None):
        calls['embed'] += 1
        return [_vec(t) for t in texts], ''

    def fake_chat(messages, **kw):
        calls['chat'].append(messages[-1]['content'])
        return 'මෙය පිළිතුරයි.', ''

    monkeypatch.setattr(A.llm, 'embed', fake_embed)
    monkeypatch.setattr(A.llm, 'chat', fake_chat)
    monkeypatch.setattr(A.llm, 'available', lambda: (True, ''))
    return VectorStore(tmp_path / 'store'), calls


def _voice(**kw):
    base = {'route': 'GENERATE', 'intent': 'SUMMARIZE',
            'english_translation': 'what is this about',
            'style_class': 'Detailed', 'prompt_modifier': '',
            'personalization_flags': {}}
    base.update(kw)
    return base


# ---- the contract --------------------------------------------------------
def test_answer_returns_component3_four_fields(rag):
    store, _ = rag
    out = A.run(store, {'corrected_text': ARTICLE}, _voice())
    for k in ('intent', 'answer_si', 'retrieved_sources', 'speakable_text'):
        assert k in out
    assert out['speakable_text'] == out['answer_si'] == 'මෙය පිළිතුරයි.'


def test_a_non_generate_route_does_not_call_the_model(rag):
    store, calls = rag
    out = A.run(store, {'corrected_text': ARTICLE},
                _voice(route='TTS_REPLAY'))
    assert calls['chat'] == []
    assert out['retrieved_sources'] == []


# ---- the corpus that builds itself --------------------------------------
def test_the_page_being_read_is_indexed_and_retrieved(rag):
    store, _ = rag
    out = A.run(store, {'corrected_text': ARTICLE}, _voice())
    kinds = [s.get('source_type') for s in out['retrieved_sources']]
    assert 'ocr_current' in kinds, 'the page being read must always be evidence'


def test_the_current_page_replaces_the_previous_one(rag):
    store, _ = rag
    A.run(store, {'corrected_text': ARTICLE}, _voice())
    A.run(store, {'corrected_text': OTHER}, _voice())
    current = [c for c in store.chunks
               if c['metadata']['source_type'] == 'ocr_current']
    assert current and all('ක්‍රිකට්' in c['text'] or 'ක්‍රීඩකයන්' in c['text']
                           for c in current)


def test_articles_that_were_read_are_remembered(rag):
    """The answer to the corpus that has never existed: index what is read."""
    store, _ = rag
    A.run(store, {'corrected_text': ARTICLE}, _voice())
    assert store.stats()['by_source_type'].get('read', 0) > 0


def test_remembering_can_be_switched_off(rag):
    store, _ = rag
    A.run(store, {'corrected_text': ARTICLE}, _voice(), remember=False)
    assert store.stats()['by_source_type'].get('read', 0) == 0


def test_the_same_article_is_not_embedded_twice(rag):
    store, calls = rag
    A.run(store, {'corrected_text': ARTICLE}, _voice())
    before = store.stats()['by_source_type'].get('read', 0)
    A.run(store, {'corrected_text': ARTICLE}, _voice())
    assert store.stats()['by_source_type'].get('read', 0) == before


def test_retrieval_prefers_the_relevant_remembered_article(rag):
    store, _ = rag
    A.index_records(store, A.records_from_text(OTHER, 'article', 'cricket'))
    A.index_records(store, A.records_from_text(ARTICLE, 'article', 'council'))
    docs, reason = A.retrieve(store, 'කුරුණෑගල නගර සභාවේ සංවර්ධන ව්‍යාපෘති',
                              top_k=2)
    assert not reason and docs
    assert any('කුරුණෑගල' in d['text'] for d in docs)


# ---- failure is a sentence, not a stack trace ---------------------------
def test_an_embedding_failure_still_returns_the_contract(rag, monkeypatch):
    store, _ = rag
    monkeypatch.setattr(A.llm, 'embed', lambda *a, **k: (None, 'HTTP 429'))
    out = A.run(store, {'corrected_text': ARTICLE}, _voice())
    assert out['speakable_text'] == A.FAILED
    assert any('429' in n for n in out['notes'])


def test_a_generation_failure_still_returns_the_contract(rag, monkeypatch):
    store, _ = rag
    monkeypatch.setattr(A.llm, 'chat', lambda *a, **k: (None, 'HTTP 500'))
    out = A.run(store, {'corrected_text': ARTICLE}, _voice())
    assert out['speakable_text'] == A.FAILED
    assert out['answer_si'] == A.FAILED


def test_no_evidence_says_so_rather_than_inventing(rag):
    store, _ = rag
    out = A.run(store, {'corrected_text': ''}, _voice())
    assert out['answer_si'] == A.NO_EVIDENCE


# ---- Nadee's guards, kept ------------------------------------------------
def test_a_code_switched_answer_triggers_the_strict_retry(rag, monkeypatch):
    store, _ = rag
    replies = ['This article is about the municipal council meeting.',
               'මෙය නගර සභා රැස්වීම පිළිබඳ ලිපියකි.']

    def chat(messages, **kw):
        return (replies.pop(0) if replies else 'x'), ''
    monkeypatch.setattr(A.llm, 'chat', chat)
    out = A.run(store, {'corrected_text': ARTICLE}, _voice())
    assert A.sinhala_purity(out['answer_si']) >= 0.85
    assert not replies, 'the retry did not happen'


def test_word_limits_follow_the_style_and_the_flags():
    assert A.resolve_max_words('Detailed', {'detail_level': 'full'}) == 500
    assert A.resolve_max_words('StepByStep', {}) == 300      # mixed case
    assert A.resolve_max_words('', {}) == 200                # the fallback


def test_the_prompt_still_forbids_going_beyond_the_evidence(rag):
    store, calls = rag
    A.run(store, {'corrected_text': ARTICLE}, _voice())
    sent = calls['chat'][0]
    assert 'සාක්ෂි' in sent and 'ප්‍රමාණවත් තොරතුරු නොමැත' in sent


# ---- the store -----------------------------------------------------------
def test_the_store_survives_a_restart(rag, tmp_path):
    store, _ = rag
    A.run(store, {'corrected_text': ARTICLE}, _voice())
    n = len(store.chunks)
    assert VectorStore(store.path).chunks and len(VectorStore(store.path).chunks) == n


def test_a_torn_store_resets_instead_of_answering_from_the_wrong_vectors(rag):
    """If the manifest and the vectors disagree, every chunk is attached to
    the wrong embedding. That looks exactly like working retrieval and is
    not, so the store refuses to load it."""
    store, _ = rag
    A.run(store, {'corrected_text': ARTICLE}, _voice())
    store.chunks.append({'chunk_id': 'x', 'text': 'x', 'hash': 'x',
                         'metadata': {}})
    store.save()
    assert VectorStore(store.path).chunks == []
