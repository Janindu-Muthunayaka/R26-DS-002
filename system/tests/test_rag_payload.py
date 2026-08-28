"""The Layer 5 -> Component 3 payload.

The point of this file is one assertion repeated three ways: **the payload
carries no per-token label and no per-token confidence.** The deployed
corrector is plain full-sequence mT5; there is no classifier to produce them.
The model that would have is the SinBERT-gated corrector, which underperformed
and is the project's reported negative result.

If someone later adds those fields to satisfy `TempFormatPleaseRead.txt`,
these tests fail, and that is the whole intent.
"""
from core.schemas import Article, Box, Document
from layers.l5_assemble.payload import article_text, diff_tokens, rag_payload

BOX = Box(x1=0, y1=0, x2=10, y2=10)


def _doc(*pairs, warnings=()):
    arts = [Article(index=i, box=BOX, body_raw=raw, body=corr, verdict='ok')
            for i, (raw, corr) in enumerate(pairs)]
    return Document(articles=arts, warnings=list(warnings))


def test_no_confidence_and_no_label_anywhere():
    p = rag_payload(_doc(('ආරථිකය වර්ධනය', 'ආර්ථිකය වර්ධනය')))
    assert p['tokens'], 'expected some tokens to compare'
    for t in p['tokens']:
        assert 'confidence' not in t, (
            'mT5 is full-sequence — there is no per-token confidence to '
            'report, and inventing one would fabricate a number')
        assert 'label' not in t, (
            'a per-token ERROR/CORRECT label needs a classifier; the gated '
            'model that had one is the negative result and is not deployed')


def test_token_source_says_what_the_array_is():
    assert rag_payload(_doc(('a b', 'a c')))['token_source'] == 'diff'


def test_unchanged_words_are_marked_unchanged():
    toks = diff_tokens('එක දෙක තුන', 'එක දෙක තුන')
    assert len(toks) == 3
    assert all(t['was_changed'] is False for t in toks)


def test_a_substitution_is_paired():
    toks = diff_tokens('ආරථිකය වර්ධනය', 'ආර්ථිකය වර්ධනය')
    changed = [t for t in toks if t['was_changed']]
    assert len(changed) == 1
    assert changed[0]['original'] == 'ආරථිකය'
    assert changed[0]['corrected'] == 'ආර්ථිකය'


def test_a_deletion_has_an_empty_corrected():
    d = [t for t in diff_tokens('a b c', 'a c') if t['was_changed']]
    assert d and d[0]['original'] == 'b' and d[0]['corrected'] == ''


def test_an_insertion_has_an_empty_original():
    d = [t for t in diff_tokens('a c', 'a b c') if t['was_changed']]
    assert d and d[0]['original'] == '' and d[0]['corrected'] == 'b'


def test_empty_inputs_do_not_raise():
    assert diff_tokens('', '') == []
    assert all(t['was_changed'] is False for t in diff_tokens('a b', ''))
    assert all(t['was_changed'] is True for t in diff_tokens('', 'a b'))


def test_the_diff_is_capped():
    assert len(diff_tokens(' '.join(['w'] * 50), ' '.join(['w'] * 50),
                           max_words=10)) == 10


def test_corrected_text_prefers_body_and_falls_back_to_raw():
    assert 'නිවැරදි' in rag_payload(_doc(('අමු', 'නිවැරදි')))['corrected_text']
    assert 'අමු' in rag_payload(_doc(('අමු', '')))['corrected_text']


def test_the_polished_text_wins_when_there_is_one():
    """`body` is the research artifact; `body_polished` is what was spoken."""
    a = Article(index=0, box=BOX, body_raw='raw', body='mt5')
    assert article_text(a) == 'mt5'
    a.body_polished = 'polished'
    assert article_text(a) == 'polished'
    assert a.body == 'mt5', 'body must never be overwritten'


def test_articles_are_kept_in_order_with_diagnostics():
    p = rag_payload(_doc(('a', 'පළමු'), ('b', 'දෙවන')))
    assert [x['index'] for x in p['articles']] == [0, 1]
    assert p['corrected_text'].index('පළමු') < p['corrected_text'].index('දෙවන')
    assert 'verdict' in p['articles'][0] and 'polished' in p['articles'][0]


def test_warnings_travel_with_the_payload():
    p = rag_payload(_doc(('a', 'b'), warnings=['Article 2 skipped: too far']))
    assert any('skipped' in w for w in p['warnings'])


def test_empty_document_is_safe():
    p = rag_payload(Document())
    assert p['corrected_text'] == '' and p['tokens'] == []


def test_tokens_can_be_switched_off():
    assert rag_payload(_doc(('a', 'b')), with_tokens=False)['tokens'] == []
