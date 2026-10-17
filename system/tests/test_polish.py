"""Layer 4C — the guards that stop a language model inventing the news.

Handing corrupt Sinhala to an LLM and asking it to "fix" it is the most
dangerous thing in this repository. A model given a shattered sentence returns
a fluent one, with names and numbers that were never on the page, and the
phone reads it to a blind user in the same voice it uses for real news.

`check()` is the entire defence. It is deliberately separated from the network
call so it can be tested exhaustively without one, and this file is that test.
"""
from pathlib import Path

import pytest

from core import quality
from core.config import POLISH_MODE
from layers.l4c_polish import polish as P

GOOD = ('ඉංග්‍රීසි ජාතික ආක්‍රමණිකයන් හා ලක්දිව එකල පැවති එකම නිදහස් රට වූ '
        'සිංහලේ රදළ වරුන් අතර ඇති කර ගත් අවබෝධතා ගිවිසුමකි. කුරුණෑගල නගර '
        'සභාවේ විවිධ සංවර්ධන ව්‍යාපෘති සාකච්ඡා විය.')
SLIGHTLY_BROKEN = GOOD.replace('ආක්‍රමණිකයන්', 'ආක්‍රමණීකයන්').replace('රදළ', 'රදල')


def test_it_is_off_by_default():
    """The only setting under which any CER may be quoted."""
    assert POLISH_MODE.lower() == 'off'
    r = P.polish(GOOD)
    assert r['applied'] is False and r['text'] == GOOD.strip()


def test_a_faithful_repair_is_accepted():
    ok, why, sim = P.check(SLIGHTLY_BROKEN, GOOD)
    assert ok, why
    assert sim > 0.9


def test_a_wholesale_rewrite_is_rejected():
    """The core defence: fluent, plausible, and nothing to do with the page."""
    invented = ('ජනාධිපතිවරයා ඊයේ පාර්ලිමේන්තුවේදී නව බදු ප්‍රතිපත්තියක් '
                'ප්‍රකාශයට පත් කළ අතර එය ලබන මාසයේ සිට ක්‍රියාත්මක වේ.')
    ok, why, _ = P.check(GOOD, invented)
    assert not ok and 'rewrote too much' in why


def test_a_summary_is_rejected_by_length():
    assert not P.check(GOOD, 'කුරුණෑගල නගර සභාව ගැන ලිපියකි.')[0]


def test_padding_is_rejected():
    assert not P.check(GOOD, GOOD + ' ' + GOOD)[0]


def test_an_english_answer_is_rejected():
    assert not P.check(
        GOOD, 'This article is about the Kurunegala municipal council and '
              'various development projects discussed at a recent meeting of')[0]


def test_an_empty_reply_is_rejected():
    ok, why, _ = P.check(GOOD, '')
    assert not ok and 'empty' in why


def test_word_inflation_is_rejected():
    assert not P.check(GOOD, GOOD[:len(GOOD) // 2] + ' ' +
                       ' '.join(['නව'] * 60))[0]


def _fake_chat(reply, calls=None):
    def chat(messages, **kw):
        if calls is not None:
            calls.append(messages)
        return reply, ''
    return chat


def test_a_hallucinating_model_never_reaches_the_caller(monkeypatch):
    monkeypatch.setattr(P.llm, 'available', lambda: (True, ''))
    monkeypatch.setattr(P.llm, 'chat', _fake_chat(
        'ජනාධිපතිවරයා ඊයේ නව බදු ප්‍රතිපත්තියක් ප්‍රකාශයට පත් කළේය.'))
    r = P.polish(GOOD, mode='on')
    assert r['applied'] is False
    assert r['text'] == GOOD.strip(), 'the ORIGINAL must come back'
    assert 'REJECTED' in r['reason']


def test_an_accepted_repair_is_returned_and_reported(monkeypatch):
    monkeypatch.setattr(P.llm, 'available', lambda: (True, ''))
    monkeypatch.setattr(P.llm, 'chat', _fake_chat(GOOD))
    r = P.polish(SLIGHTLY_BROKEN, mode='on')
    assert r['applied'] is True and r['text'] == GOOD
    assert 'applied' in r['reason'] and '%' in r['reason']


def test_a_failed_call_returns_the_original(monkeypatch):
    monkeypatch.setattr(P.llm, 'available', lambda: (True, ''))
    monkeypatch.setattr(P.llm, 'chat', lambda *a, **k: (None, 'HTTP 429'))
    r = P.polish(GOOD, mode='on')
    assert r['applied'] is False and r['text'] == GOOD.strip()
    assert '429' in r['reason']


def test_no_key_is_a_reason_not_a_crash(monkeypatch):
    monkeypatch.setattr(P.llm, 'available', lambda: (False, 'no key'))
    r = P.polish(GOOD, mode='on')
    assert r['applied'] is False and 'unavailable' in r['reason']


def test_auto_leaves_good_text_alone(monkeypatch):
    monkeypatch.setattr(P.llm, 'available',
                        lambda: pytest.fail('must not call the model'))
    r = P.polish(GOOD, mode='auto')
    assert r['applied'] is False and 'not needed' in r['reason']


def test_auto_refuses_unreadable_text(monkeypatch):
    """THE DELIBERATE REFUSAL. Unreadable text is where a repair would be most
    welcome and where invention is most likely: with little real signal left,
    fluency is all the model has to go on."""
    monkeypatch.setattr(P.llm, 'available',
                        lambda: pytest.fail('must not call the model'))
    shattered = 'ක ය ම න ද ව ග ර ත ප ල ස හ ක ය ම න ද ව ග ර'
    assert quality.score(shattered)['verdict'] == 'unreadable'
    r = P.polish(shattered, mode='auto')
    assert r['applied'] is True and r['text'] == '[DISCARD]'


def test_the_prompt_forbids_adding_information(monkeypatch):
    calls = []
    monkeypatch.setattr(P.llm, 'available', lambda: (True, ''))
    monkeypatch.setattr(P.llm, 'chat', _fake_chat(GOOD, calls))
    P.polish(SLIGHTLY_BROKEN, mode='on')
    system = calls[0][0]['content'].lower()
    assert 'never add' in system and 'never remove' in system
    assert 'only the repaired text' in system


EVAL_TOOLS = ('eval_articles.py', 'verify_model.py', 'compare_framing.py',
              'reproduce_diagnostics.py')


def test_no_evaluation_tool_can_switch_this_on():
    """Chapter 4's CER is mT5's. If an evaluation path could enable a
    general-purpose language model, the number being measured would no longer
    be the model the thesis is about."""
    tools = Path(__file__).resolve().parent.parent / 'tools'
    for name in EVAL_TOOLS:
        p = tools / name
        if p.exists():
            assert 'polish' not in p.read_text(
                encoding='utf-8', errors='replace').lower(), \
                f'{name} references the post-editor'


def test_layer_4b_does_not_import_the_post_editor():
    """`body.correct()` is the measured path and must stay untouched."""
    src = (Path(__file__).resolve().parent.parent /
           'layers' / 'l4b_body' / 'body.py').read_text(encoding='utf-8')
    assert 'polish' not in src.lower()


def test_the_polished_text_lives_in_its_own_field():
    from core.schemas import Article, Box
    from layers.l5_assemble.payload import article_text
    a = Article(index=0, box=Box(x1=0, y1=0, x2=1, y2=1),
                body_raw='raw', body='mt5')
    a.body_polished = 'polished'
    assert a.body == 'mt5' and a.body_raw == 'raw'
    assert article_text(a) == 'polished'
    a.body_polished = ''
    assert article_text(a) == 'mt5'


def test_polish_discards_nonsense_title_when_regeneration_fails(monkeypatch):
    from core.schemas import Article, Box
    monkeypatch.setattr(P.llm, 'available', lambda: (True, ''))
    # Simulate LLM returning a garbage title or empty title even after regeneration
    garbage_reply = '[{"id": "0", "title": "...", "body": "මෙම ලිපිය කියවිය හැකි නමුත් ශීර්ෂය නිවැරදි නැත."}]'
    monkeypatch.setattr(P.llm, 'chat', lambda *a, **k: (garbage_reply, ''))
    
    art = Article(index=0, box=Box(x1=0, y1=0, x2=1, y2=1),
                  body="මෙම ලිපිය කියවිය හැකි නමුත් ශීර්ෂය නිවැරදි නැත.",
                  title="...")
    res = P.polish_articles([art], mode='on')[0]
    # Because title returned is nonsense ("..."), it should be discarded
    assert res['body'] == '[DISCARD]'
    assert res['title'] == '[DISCARD]'


def test_polish_regenerates_nonsense_title(monkeypatch):
    from core.schemas import Article, Box
    monkeypatch.setattr(P.llm, 'available', lambda: (True, ''))
    # Simulate LLM successfully generating a proper title
    good_reply = '[{"id": "0", "title": "නව රැස්වීම", "body": "මෙම ලිපිය කියවිය හැකි නමුත් ශීර්ෂය නිවැරදි නැත."}]'
    monkeypatch.setattr(P.llm, 'chat', lambda *a, **k: (good_reply, ''))
    
    art = Article(index=0, box=Box(x1=0, y1=0, x2=1, y2=1),
                  body="මෙම ලිපිය කියවිය හැකි නමුත් ශීර්ෂය නිවැරදි නැත.",
                  title="...")
    res = P.polish_articles([art], mode='on')[0]
    assert res['body'] == "මෙම ලිපිය කියවිය හැකි නමුත් ශීර්ෂය නිවැරදි නැත."
    assert res['title'] == "නව රැස්වීම"
