"""Speed changes are clearly heard, then fade back to normal (18 Oct 2026).

One "faster" / "slower" moves the rate x1.15 / x0.88 (was x1.10 / x0.90), a
second one reaches the limit, and every turn without a delivery request moves
rate and pause 40% of the way back to 1.0 (delivery/feedback.py, RELAX_*).
Before, the rate stayed wherever the user last put it, and a user who had
said "slower" twice was still slow in every later session.
"""
from datetime import datetime

from layers.l7_personalization.delivery import feedback as fb
from layers.l7_personalization.delivery.voice_profile import new_profile
from layers.l7_personalization.intent_detection.modifiers import build_frame

NOW = datetime(2026, 10, 18, 12, 0)


def run(*texts, profile=None):
    p, rates = profile or new_profile(), []
    for t in texts:
        p, _ = fb.apply_turn(p, build_frame(t, '', 'UNKNOWN'), NOW)
        rates.append(p['rate'])
    return p, rates


def test_one_request_is_a_big_step_but_not_the_limit():
    _, (fast,) = run('ඉක්මනින් කියන්න')
    _, (slow,) = run('හෙමින් කියන්න')
    assert 1.12 <= fast < fb.RATE_MAX
    assert fb.RATE_MIN < slow <= 0.9


def test_two_requests_reach_the_limit():
    _, rates = run('ඉක්මනින් කියන්න', 'ඉක්මනින් කියන්න')
    assert rates[-1] == fb.RATE_MAX
    _, rates = run('හෙමින් කියන්න', 'හෙමින් කියන්න')
    assert rates[-1] == fb.RATE_MIN


def test_the_speed_fades_back_to_normal_within_a_few_turns():
    p, rates = run('ඉක්මනින් කියන්න', 'ඊළඟ කොටස', 'කවුද ආවේ', 'සරලව කියන්න', 'ඊළඟ කොටස')
    assert rates[0] > rates[1] > rates[2] > rates[3] > 1.0, 'should come down gradually'
    assert rates[-1] == 1.0 and p['pause_scale'] == 1.0
    p, rates = run('හෙමින් කියන්න', 'ඊළඟ කොටස', 'ඊළඟ කොටස', 'ඊළඟ කොටස')
    assert rates[-1] == 1.0 and p['pause_scale'] == 1.0


def test_asking_again_keeps_it_fast():
    _, rates = run('ඉක්මනින් කියන්න', 'ඊළඟ කොටස', 'ඉක්මනින් කියන්න')
    assert rates[-1] > rates[0]


def test_a_delivery_request_does_not_relax_on_the_same_turn():
    p, _ = run('නවත්තලා නවත්තලා කියන්න')         # more pauses
    assert p['pause_scale'] > 1.0 and p['rate'] == 1.0
