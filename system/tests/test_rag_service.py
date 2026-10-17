"""Component 3's retrieval and answering, with no network and no API key.

The OpenAI calls are the only part of `services/rag` that needs the outside
world, so they are the only part faked here. Everything else — chunking, the
dedupe key, the vector store's alignment between text and vectors, the
three-step retrieval rule, and the failure paths — is exercised for real.

Two of these tests exist because of bugs that were live and invisible:

  * `record_hash` — the page being read is indexed twice on purpose, as
    `ocr_current` (replaced every request) and as `read` (kept). With a
    text-only dedupe key the second looked like a duplicate of the first, so
    nothing was ever remembered. The store reported success every time.

  * `ok` on a NO_EVIDENCE answer — "there is not enough information in the
    text" is a perfectly good sentence to speak and a failed answer to log.
    Without the flag the two were the same thing.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_RAG = Path(__file__).resolve().parents[2] / 'services' / 'rag'
if str(_RAG) not in sys.path:
    sys.path.insert(0, str(_RAG))

import answer as A            # noqa: E402
import ingest                 # noqa: E402
from store import VectorStore, record_hash, text_hash   # noqa: E402


SI = ('කුරුණෑගල නගර සභාව අද පැවති රැස්වීමේදී නව ජල ව්‍යාපෘතියක් සඳහා '
      'අනුමැතිය ලබා දුන් බව සභාපතිවරයා පැවසීය. ')
OTHER = ('කොළඹ දුම්රිය ස්ථානයේ නවීකරණ කටයුතු ලබන මාසයේ ආරම්භ කරන බව '
         'ප්‍රවාහන අමාත්‍යාංශය දැනුම් දුන්නේය. ')


def _vec(text: str, dim: int = 16):
    """A deterministic stand-in for an embedding.

    Codepoint-bucket counts: texts sharing vocabulary land near each other, so
    "nearest" means something without a model. It is not a good embedding and
    it does not need to be — the tests here are about wiring, not accuracy.
    """
    v = np.zeros(dim, dtype=np.float32)
    for ch in text or ' ':
        v[ord(ch) % dim] += 1.0
    n = float(np.linalg.norm(v)) or 1.0
    return (v / n).tolist()


class FakeLLM:
    """Records what it was asked. Fails on demand."""

    def __init__(self, replies=('පිළිතුර සිංහලෙන්.',),
                 embed_fail='', chat_fail=''):
        self.replies = list(replies)
        self.embed_fail, self.chat_fail = embed_fail, chat_fail
        self.embed_calls, self.chat_calls = [], []

    def embed(self, texts):
        self.embed_calls.append(list(texts))
        if self.embed_fail:
            return None, self.embed_fail
        return [_vec(t) for t in texts], ''

    def chat(self, messages, **kw):
        self.chat_calls.append(messages)
        if self.chat_fail:
            return None, self.chat_fail
        return (self.replies.pop(0) if len(self.replies) > 1
                else self.replies[0]), ''


@pytest.fixture
def fake(monkeypatch):
    f = FakeLLM()
    monkeypatch.setattr(A.llm, 'embed', f.embed)
    monkeypatch.setattr(A.llm, 'chat', f.chat)
    return f


@pytest.fixture
def store(tmp_path):
    return VectorStore(tmp_path / 'vs')


VOICE = {'route': 'GENERATE', 'intent': 'SUMMARY',
         'english_translation': 'what did the council approve',
         'style_class': 'Moderate', 'prompt_modifier': '',
         'personalization_flags': {}}


# ---- the dedupe key ------------------------------------------------------
def test_the_same_text_under_two_source_types_is_not_a_duplicate():
    """THE BUG. `ocr_current` and `read` hold identical text on purpose."""
    a = {'text': SI, 'metadata': {'source_type': 'ocr_current'}}
    b = {'text': SI, 'metadata': {'source_type': 'read'}}
    assert record_hash(a) != record_hash(b), (
        'a text-only key made the kept copy look like a duplicate of the '
        'replaced one, so the corpus silently never grew')
    assert text_hash(SI) == text_hash(SI)


def test_the_same_text_and_type_still_dedupes():
    a = {'text': SI, 'metadata': {'source_type': 'read'}}
    b = {'text': SI, 'metadata': {'source_type': 'read'}}
    assert record_hash(a) == record_hash(b)


# ---- the vector store ----------------------------------------------------
def _add(store, texts, kind='read'):
    recs = [{'text': t, 'metadata': {'source_type': kind, 'chunk_id': f'c{i}'}}
            for i, t in enumerate(texts)]
    return store.add(recs, [_vec(t) for t in texts])


def test_search_returns_the_nearest_chunk_first(store):
    _add(store, [SI, OTHER])
    hits = store.search(_vec(SI), k=2)
    assert hits[0]['text'] == SI
    assert hits[0]['score'] >= hits[1]['score']


def test_vectors_are_normalised_on_add(store):
    store.add([{'text': SI, 'metadata': {}}], [[3.0, 4.0] + [0.0] * 14])
    assert np.isclose(np.linalg.norm(store.vectors[0]), 1.0)


def test_a_where_filter_restricts_the_pool(store):
    _add(store, [SI], 'ocr_current')
    _add(store, [OTHER], 'read')
    hits = store.search(_vec(OTHER), k=4, where={'source_type': 'ocr_current'})
    assert len(hits) == 1 and hits[0]['text'] == SI


def test_delete_where_keeps_text_and_vectors_aligned(store):
    _add(store, [SI], 'ocr_current')
    _add(store, [OTHER], 'read')
    assert store.delete_where('source_type', 'ocr_current') == 1
    assert len(store.chunks) == len(store.vectors) == 1
    assert store.chunks[0]['text'] == OTHER
    assert store.search(_vec(OTHER), k=1)[0]['text'] == OTHER


def test_it_survives_a_restart(tmp_path):
    p = tmp_path / 'vs'
    a = VectorStore(p)
    _add(a, [SI, OTHER])
    a.save()
    b = VectorStore(p)
    assert len(b.chunks) == 2
    assert b.search(_vec(SI), k=1)[0]['text'] == SI


def test_a_store_whose_two_files_disagree_is_dropped_not_used(tmp_path):
    """Text and vectors out of step means answering with evidence attached to
    the wrong vectors — which looks exactly like working retrieval."""
    p = tmp_path / 'vs'
    a = VectorStore(p)
    _add(a, [SI, OTHER])
    a.save()
    np.save(p / 'vectors.npy', np.asarray([_vec(SI)], dtype=np.float32))
    b = VectorStore(p)
    assert b.chunks == [] and b.vectors is None


def test_a_corrupt_manifest_does_not_stop_the_service_starting(tmp_path):
    p = tmp_path / 'vs'
    p.mkdir(parents=True)
    (p / 'manifest.json').write_text('{not json', encoding='utf-8')
    assert VectorStore(p).chunks == []


# ---- chunking ------------------------------------------------------------
def test_chunks_stay_under_the_size_limit():
    chunks = ingest.split_chunks(SI * 20, size=400, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 400 + 60 for c in chunks), \
        [len(c) for c in chunks]


def test_short_text_is_one_chunk_not_zero():
    assert len(ingest.split_chunks('කෙටි පාඨයකි.')) == 1


def test_records_carry_the_metadata_retrieval_filters_on():
    recs = ingest.records_from_text(SI * 4, 'read', 'base1',
                                    {'article_id': 'base1'})
    assert recs
    for r in recs:
        assert r['text'].strip()
        assert r['metadata']['source_type'] == 'read'
        assert r['metadata']['chunk_id']
        assert r['metadata']['article_id'] == 'base1'
    ids = [r['metadata']['chunk_id'] for r in recs]
    assert len(set(ids)) == len(ids), 'chunk_ids must be unique'


def test_indexing_skips_text_already_present(store, fake):
    recs = ingest.records_from_text(SI * 3, 'read', 'b')
    n1, _ = A.index_records(store, recs)
    n2, _ = A.index_records(store, ingest.records_from_text(SI * 3, 'read', 'b'))
    assert n1 > 0 and n2 == 0


def test_an_embedding_failure_stores_nothing_and_says_why(store, monkeypatch):
    f = FakeLLM(embed_fail='HTTP 429')
    monkeypatch.setattr(A.llm, 'embed', f.embed)
    n, reason = A.index_records(store, ingest.records_from_text(SI, 'read', 'b'))
    assert n == 0 and '429' in reason
    assert store.chunks == []


# ---- word limits and the purity guard ------------------------------------
def test_detail_level_overrides_the_style_class():
    assert A.resolve_max_words('Detailed', {'detail_level': 'brief'}) == 80
    assert A.resolve_max_words('Detailed', {}) == 400


def test_an_unknown_style_class_falls_back_rather_than_raising():
    assert A.resolve_max_words('not-a-style', {}) > 0


def test_sinhala_purity_scores_what_it_says():
    assert A.sinhala_purity('නගර සභාව') == 1.0
    assert A.sinhala_purity('the council') == 0.0
    assert A.sinhala_purity('') == 1.0        # nothing to be impure


def test_a_code_switched_answer_is_retried_in_sinhala(store, monkeypatch):
    """Prompting alone cannot guarantee no English. The guard is a retry."""
    f = FakeLLM(replies=['The council approved the water project.',
                         'නගර සභාව ජල ව්‍යාපෘතියට අනුමැතිය දුන්නේය.'])
    monkeypatch.setattr(A.llm, 'embed', f.embed)
    monkeypatch.setattr(A.llm, 'chat', f.chat)
    out = A.run(store, {'corrected_text': SI}, VOICE)
    assert len(f.chat_calls) == 2, 'the impure answer was not retried'
    assert A.STRICT_SUFFIX.strip()[:20] in f.chat_calls[1][0]['content']
    assert A.sinhala_purity(out['answer_si']) > 0.85


# ---- run(): the /answer contract -----------------------------------------
def test_a_normal_question_is_answered_from_the_page(store, fake):
    out = A.run(store, {'corrected_text': SI}, VOICE)
    assert out['ok'] is True
    assert out['answer_si'].strip()
    assert out['speakable_text'] == out['answer_si']
    assert out['retrieved_sources'], 'the answer cites nothing'
    assert all('score' in s for s in out['retrieved_sources'])


def test_the_page_being_read_is_both_replaced_and_remembered(store, fake):
    """Both halves of the fix: `ocr_current` is swapped for the new page, and
    a `read` copy of the old one stays behind. Before `record_hash` the second
    never happened and nothing said so."""
    A.run(store, {'corrected_text': SI}, VOICE)
    kinds = [c['metadata']['source_type'] for c in store.chunks]
    assert 'ocr_current' in kinds and 'read' in kinds

    A.run(store, {'corrected_text': OTHER}, VOICE)
    current = [c['text'] for c in store.chunks
               if c['metadata']['source_type'] == 'ocr_current']
    remembered = [c['text'] for c in store.chunks
                  if c['metadata']['source_type'] == 'read']
    assert current and all(OTHER[:20] in t for t in current), \
        'the previous page was not replaced'
    assert any(SI[:20] in t for t in remembered), \
        'the previous page was not remembered'


def test_remember_false_indexes_only_the_current_page(store, fake):
    A.run(store, {'corrected_text': SI}, VOICE, remember=False)
    kinds = {c['metadata']['source_type'] for c in store.chunks}
    assert kinds == {'ocr_current'}


def test_retrieval_always_includes_the_page_being_read(store, fake):
    _add(store, [OTHER] * 6, 'read')
    out = A.run(store, {'corrected_text': SI}, VOICE)
    kinds = [s.get('source_type') for s in out['retrieved_sources']]
    assert 'ocr_current' in kinds, (
        'a question about the page in front of the user must see that page, '
        'however the semantic search ranks it')


def test_an_anchor_chunk_from_the_voice_module_is_honoured(store, fake):
    _add(store, [OTHER], 'read')
    anchor = store.chunks[-1]['metadata']['chunk_id']
    v = dict(VOICE, retrieved_chunk_id=anchor)
    out = A.run(store, {'corrected_text': SI}, v)
    assert any(s.get('chunk_id') == anchor for s in out['retrieved_sources'])


def test_no_evidence_is_a_failure_even_though_it_is_speakable(store, fake):
    """THE OTHER BUG. A sentence the user should hear, and an answer that was
    not produced, are not the same thing."""
    out = A.run(store, {'corrected_text': ''}, VOICE)
    assert out['speakable_text'] == A.NO_EVIDENCE
    assert out['ok'] is False


def test_a_retrieval_failure_still_returns_something_speakable(store,
                                                              monkeypatch):
    f = FakeLLM(embed_fail='unreachable: connection refused')
    monkeypatch.setattr(A.llm, 'embed', f.embed)
    monkeypatch.setattr(A.llm, 'chat', f.chat)
    out = A.run(store, {'corrected_text': SI}, VOICE)
    assert out['ok'] is False
    assert out['speakable_text'] == A.FAILED
    assert any('retrieval failed' in n for n in out['notes'])


def test_a_generation_failure_still_returns_something_speakable(store,
                                                               monkeypatch):
    f = FakeLLM(chat_fail='HTTP 500')
    monkeypatch.setattr(A.llm, 'embed', f.embed)
    monkeypatch.setattr(A.llm, 'chat', f.chat)
    out = A.run(store, {'corrected_text': SI}, VOICE)
    assert out['ok'] is False
    assert out['speakable_text'] == A.FAILED
    assert any('generation failed' in n for n in out['notes'])


def test_a_non_generate_route_never_calls_the_model(store, fake):
    out = A.run(store, {'corrected_text': SI},
                dict(VOICE, route='LOCAL', intent='REPEAT'))
    assert out['ok'] is True
    assert out['retrieved_sources'] == []
    assert fake.chat_calls == [], 'a local command paid for a generation'


def test_the_evidence_reaches_the_prompt(store, fake):
    A.run(store, {'corrected_text': SI}, VOICE)
    sent = fake.chat_calls[-1][0]['content']
    assert SI[:25] in sent, 'the model was asked without the article'
    assert 'what did the council approve' in sent


def test_embeddings_are_batched_without_losing_order(store, fake):
    """`embed()` sorts on the index the API returns — a reordered batch would
    attach chunks to the wrong text, and every answer would still look fine."""
    recs = ingest.records_from_text(SI * 30, 'read', 'b')
    assert len(recs) > 1
    A.index_records(store, recs)
    assert [c['text'] for c in store.chunks] == [r['text'] for r in recs]
