"""The generated answer is spoken aloud: no markdown symbol may reach the TTS."""
from layers.l6_generator.tts_clean import make_speakable


def test_bold_is_unwrapped_not_spoken():
    # The bug: TTS read "**subject**" as "star star subject star star".
    assert make_speakable('**විෂය** ගැන කතා කරමු.') == 'විෂය ගැන කතා කරමු.'
    assert make_speakable('this is **subject** here') == 'this is subject here.'


def test_italic_strike_and_code_are_unwrapped():
    assert make_speakable('*අවධාරණය* සහ _මෙය_ ~~මකන~~ `කේතය`') == 'අවධාරණය සහ මෙය මකන කේතය.'


def test_headings_and_bullets_become_sentences():
    text = '## සාරාංශය\n- පළමු කරුණ\n* දෙවන කරුණ\n1. තුන්වන කරුණ.'
    assert make_speakable(text) == 'සාරාංශය. පළමු කරුණ. දෙවන කරුණ. තුන්වන කරුණ.'


def test_links_urls_tables_and_emoji_are_removed():
    assert make_speakable('[මූලාශ්‍රය](https://x.lk) බලන්න https://y.lk 👍') == 'මූලාශ්‍රය බලන්න.'
    table = '| නම | වයස |\n|---|---|\n| නිමල් | 30 |'
    assert make_speakable(table) == 'නම, වයස. නිමල්, 30.'


def test_sinhala_conjuncts_numbers_and_punctuation_survive():
    # ZWJ inside ශ්‍රී must be kept, as must decimals, %, dates and hyphens.
    text = 'ශ්‍රී ලංකාවේ වර්ධනය 3.5% කි. 2024-05-01 දින, ඒ-9 මාර්ගයේ?'
    assert make_speakable(text) == text


def test_stray_symbols_are_dropped_and_empty_is_safe():
    assert make_speakable('පිළිතුර # = > ඇත') == 'පිළිතුර ඇත.'
    assert make_speakable('') == ''
    assert make_speakable('***') == ''
