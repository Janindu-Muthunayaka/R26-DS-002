"""Segmentation fixes: L4C reports, segfix checks, only the obvious is applied.

One story cut into two consecutive articles is merged; two stories in one
article are split. The LLM's word is never enough on its own: a hint that
fails any check is ignored and the segments stay exactly as they were.
"""
import json

from core.schemas import Article, Box
from layers.l4c_polish import polish as P
from layers.l5_assemble import segfix
from layers.l5_assemble.payload import article_text

B = Box(x1=0, y1=0, x2=10, y2=10)
STORY_A1 = 'කුරුණෑගල නගර සභාවේ අද පැවති රැස්වීමේදී නව පාලම ඉදිකිරීමට තීරණය කළ බව සභාපතිවරයා පැවසීය. එම පාලම'
STORY_A2 = 'ලබන වසරේ විවෘත කිරීමට නියමිත බවත් ඒ සඳහා රුපියල් මිලියන පනහක් වැය වන බවත් ඔහු වැඩිදුරටත් කීය.'
STORY_B = 'කොළඹ ජාතික රෝහලේ විකිරණ යන්ත්‍රය ක්‍රියා විරහිත වී ඇති බව රෝහල් බලධාරීන් අද නිවේදනය කළේය. රෝගීන්ට වෙනත් රෝහල් වෙත යාමට සිදුවී ඇත.'


def _art(i, body, title=''):
    return Article(index=i, box=Box(x1=i * 10, y1=0, x2=i * 10 + 10, y2=10),
                   body_raw=body, body=body, title=title, verdict='ok')


def _res(hint=None, seen=True):
    r = {'body': '', 'title': '', 'applied': False, 'reason': ''}
    if seen:
        r['segment'] = hint or {}
    return r


# ── merge ──────────────────────────────────────────────────────────────────
def test_one_story_cut_in_two_is_merged_and_the_originals_are_untouched():
    a, b = _art(0, STORY_A1, 'පාලම'), _art(1, STORY_A2)
    out, notes = segfix.apply([a, b], [_res({'merge_with': '1'}), _res()])
    assert len(out) == 1
    assert STORY_A1 in article_text(out[0]) and STORY_A2 in article_text(out[0])
    assert out[0].title == 'පාලම' and out[0].box.x2 == 20
    assert any('merged' in n for n in notes)
    assert a.body == STORY_A1 and b.body == STORY_A2, 'body (mT5 output) must never be overwritten'


def test_a_merge_with_anything_but_the_next_article_is_ignored():
    arts = [_art(0, STORY_A1), _art(1, STORY_B), _art(2, STORY_A2)]
    out, notes = segfix.apply(arts, [_res({'merge_with': '2'}), _res(), _res()])
    assert len(out) == 3 and any('ignored merge' in n for n in notes)


def test_a_merge_with_an_article_the_llm_never_saw_is_ignored():
    out, _ = segfix.apply([_art(0, STORY_A1), _art(1, STORY_A2)],
                          [_res({'merge_with': '1'}), _res(seen=False)])
    assert len(out) == 2


def test_a_chain_of_three_merges_into_one():
    arts = [_art(0, STORY_A1), _art(1, STORY_A2), _art(2, 'ඒ පිළිබඳ විස්තර පසුව දැනුම් දෙන බව ද ඔහු කීය.')]
    out, _ = segfix.apply(arts, [_res({'merge_with': '1'}), _res({'merge_with': '2'}), _res()])
    assert len(out) == 1 and 'පසුව' in article_text(out[0])


# ── split ──────────────────────────────────────────────────────────────────
def test_two_stories_in_one_article_are_split_at_a_sentence_start():
    text = f'{STORY_A1} {STORY_A2} {STORY_B}'
    out, notes = segfix.apply([_art(0, text, 'පාලම')],
                              [_res({'split_at': 'කොළඹ ජාතික රෝහලේ විකිරණ'})])
    assert len(out) == 2
    assert article_text(out[0]).endswith('කීය.') and article_text(out[1]).startswith('කොළඹ')
    assert out[0].title == 'පාලම' and out[1].title == '', 'no invented headline'
    assert [a.index for a in out] == [0, 1]
    assert any('split' in n for n in notes)


def test_a_split_title_is_used_only_if_it_is_in_the_text():
    text = f'{STORY_A1} {STORY_A2}\nවිකිරණ යන්ත්‍රය නවතී\n{STORY_B}'
    out, _ = segfix.apply([_art(0, text)], [_res({'split_at': 'විකිරණ යන්ත්‍රය නවතී',
                                                  'split_title': 'විකිරණ යන්ත්‍රය නවතී'})])
    assert out[1].title == 'විකිරණ යන්ත්‍රය නවතී'
    out, _ = segfix.apply([_art(0, text)], [_res({'split_at': 'විකිරණ යන්ත්‍රය නවතී',
                                                  'split_title': 'රෝහලේ අර්බුදය'})])
    assert out[1].title == '', 'a headline that is not in the text is never used'


def test_a_split_mid_sentence_is_ignored():
    text = f'{STORY_A1} {STORY_A2} {STORY_B}'
    out, notes = segfix.apply([_art(0, text)], [_res({'split_at': 'මිලියන පනහක් වැය වන'})])
    assert len(out) == 1 and any('ignored split' in n for n in notes)


def test_a_split_phrase_that_is_not_in_the_text_is_ignored():
    out, _ = segfix.apply([_art(0, f'{STORY_A1} {STORY_B}')], [_res({'split_at': 'නොපවතින වාක්‍යයක්'})])
    assert len(out) == 1


def test_a_split_leaving_a_tiny_part_is_ignored():
    text = f'{STORY_B} කෙටි.'
    out, _ = segfix.apply([_art(0, text)], [_res({'split_at': 'කෙටි.'})])
    assert len(out) == 1


def test_merge_and_split_on_the_same_article_means_neither():
    text = f'{STORY_A1} {STORY_A2} {STORY_B}'
    out, _ = segfix.apply([_art(0, text), _art(1, STORY_B)],
                          [_res({'merge_with': '1', 'split_at': 'කොළඹ ජාතික රෝහලේ'}), _res()])
    assert len(out) == 2 and article_text(out[0]) == text


def test_no_hints_change_nothing():
    arts = [_art(0, STORY_A1), _art(1, STORY_B)]
    out, notes = segfix.apply(arts, [_res(), _res()])
    assert out == arts and notes == []


# ── L4C passes the hints through ───────────────────────────────────────────
def test_l4c_reports_hints_for_the_articles_it_saw(monkeypatch):
    arts = [_art(0, STORY_A1, 'පාලම'), _art(1, STORY_A2, 'පාලම')]
    reply = json.dumps([{'id': '0', 'title': 'පාලම', 'body': STORY_A1, 'merge_with': '1'},
                        {'id': '1', 'title': 'පාලම', 'body': STORY_A2}], ensure_ascii=False)
    monkeypatch.setattr(P.llm, 'available', lambda: (True, ''))
    monkeypatch.setattr(P.llm, 'chat', lambda *a, **k: (reply, ''))
    res = P.polish_articles(arts, mode='on')
    assert res[0]['segment'] == {'merge_with': '1'} and res[1]['segment'] == {}


def test_the_prompt_says_only_when_obvious_and_never_guess():
    p = P.SYSTEM_PROMPT
    assert 'SEGMENTATION CHECK' in p and 'OBVIOUS' in p and 'never guess' in p
    assert 'do NOT add these fields' in p
