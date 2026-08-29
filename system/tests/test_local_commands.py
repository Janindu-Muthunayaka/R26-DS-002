"""The commands that work with every service down.

These are not stubs. Each one is the correct implementation, needs no network,
no API key and no teammate's component, and is therefore the part of the
conversation that cannot break during a viva.
"""
from core.schemas import Article, Box, Document
from layers.l0_voice.voice import _stub
from layers.l6_generator import generate as l6gen

BOX = Box(x1=0, y1=0, x2=10, y2=10)


def _doc(text='', title='', warnings=()):
    return Document(
        articles=[Article(index=0, box=BOX, body_raw=text, body=text,
                          title=title, verdict='ok')],
        warnings=list(warnings))


def _long(n=12):
    return _doc(' '.join(f'මෙය {i} වන වාක්‍යයයි එය තරමක් දිගු වේ.'
                         for i in range(n)))


def _ask(doc, text, cursor=0):
    return l6gen.answer(doc, _stub(text), mode='off', cursor=cursor)


def test_every_command_word_routes_locally():
    cases = {
        'නවත්වන්න': 'STOP', 'ඊළඟ': 'NEXT', 'කලින් එක': 'PREVIOUS',
        'මුල සිට කියවන්න': 'FIRST', 'වචන කීයද': 'LENGTH',
        'ශීර්ෂය මොකක්ද': 'TITLE', 'මොනවද මඟ හැරුණේ': 'WARNINGS',
        'නැවත කියවන්න': 'REPEAT', 'කියවන්න': 'READ_ALOUD',
        'next': 'NEXT', 'how long': 'LENGTH', 'headline': 'TITLE',
    }
    for text, intent in cases.items():
        v = _stub(text)
        assert v['intent'] == intent, f'{text!r} -> {v["intent"]}, want {intent}'
        assert v['route'] == 'LOCAL'


def test_a_real_question_is_still_routed_to_the_generator():
    v = _stub('මේක සාරාංශ කරන්න')
    assert v['route'] == 'GENERATE' and v['intent'] == 'ASK'


def test_tell_me_about_this_is_a_question_not_a_read_command():
    """'කියන්න' ("say / tell me") was a READ keyword and swallowed
    "මේ ලිපිය ගැන කියන්න" — tell me about this article. A word that appears in
    ordinary questions is not a command word."""
    assert _stub('මේ ලිපිය ගැන කියන්න')['route'] == 'GENERATE'


def test_from_the_start_is_not_swallowed_by_read():
    """'මුල සිට කියවන්න' CONTAINS 'කියවන්න'. If READ_ALOUD were tested first
    every navigation command would collapse into "read it all again"."""
    assert _stub('මුල සිට කියවන්න')['intent'] == 'FIRST'


def test_next_walks_forward_and_does_not_repeat_itself():
    d = _long(40)
    n = len(l6gen._parts(d))
    assert n >= 3, 'the fixture must be long enough to actually walk'
    seen, cursor = [], 0
    for _ in range(n):
        r = _ask(d, 'ඊළඟ', cursor)
        seen.append(r['speakable'])
        cursor = r['cursor']
    assert len(set(seen)) == n, 'a part was delivered twice'
    assert cursor == n
    assert ' '.join(seen) == ' '.join(l6gen._parts(d))


def test_next_past_the_end_says_so_and_does_not_crash():
    r = _ask(_long(2), 'next', cursor=99)
    assert r['ok'] is True and r['speakable'] == l6gen.SI_END
    assert r['cursor'] == 99, 'the cursor must not run away past the end'


def test_previous_goes_back_to_what_was_just_read():
    d = _long()
    first = _ask(d, 'next', 0)
    second = _ask(d, 'next', first['cursor'])
    back = _ask(d, 'back', second['cursor'])
    assert back['speakable'] == first['speakable']
    assert back['cursor'] == first['cursor']


def test_previous_at_the_start_says_so():
    r = _ask(_long(), 'back', cursor=1)
    assert r['speakable'] == l6gen.SI_START and r['cursor'] == 0


def test_first_returns_to_part_one():
    d = _long()
    r = _ask(d, 'මුල සිට කියවන්න', cursor=5)
    assert r['cursor'] == 1
    assert r['speakable'] == _ask(d, 'next', 0)['speakable']


def test_repeat_reads_everything_and_resets_the_walk():
    d = _long()
    r = _ask(d, 'නැවත කියවන්න', cursor=6)
    assert r['cursor'] == 0
    assert len(r['speakable']) > len(_ask(d, 'next', 0)['speakable'])


def test_parts_never_begin_mid_sentence():
    for part in l6gen._parts(_long(20)):
        assert part == part.strip() and part
        assert not part.startswith(('.', '।'))


def test_length_counts_words():
    r = _ask(_doc('එක දෙක තුන හතර'), 'වචන කීයද')
    assert r['ok'] is True and '4' in r['speakable']


def test_title_is_honest_while_layer_4a_is_off():
    r = _ask(_doc('පෙළ'), 'ශීර්ෂය මොකක්ද')
    assert r['ok'] is False and r['speakable'] == l6gen.SI_NO_TITLE


def test_title_is_spoken_once_layer_4a_lands():
    r = _ask(_doc('පෙළ', title='ප්‍රධාන ශීර්ෂය'), 'headline')
    assert r['ok'] is True and 'ප්‍රධාන ශීර්ෂය' in r['speakable']


def test_warnings_tell_the_listener_what_was_skipped():
    d = _doc('පෙළ', warnings=['part of this article is off the right edge'])
    assert 'off the right edge' in _ask(d, 'මොනවද මඟ හැරුණේ')['speakable']


def test_nothing_skipped_is_also_an_answer():
    assert _ask(_doc('පෙළ'), 'did i miss')['speakable'] == l6gen.SI_NOTHING_MISSED


def test_an_empty_article_never_replays_silence():
    for q in ('නැවත කියවන්න', 'next', 'මුල සිට කියවන්න'):
        r = _ask(_doc(''), q)
        assert r['speakable'] == l6gen.SI_NOTHING_TO_REPEAT and r['ok'] is False


def test_stop_still_asks_the_phone_to_act():
    r = _ask(_long(), 'නවත්වන්න', cursor=3)
    assert r['intent'] == 'STOP' and r['speakable'] == ''
    assert r['cursor'] == 3, "stopping must not lose the listener's place"
