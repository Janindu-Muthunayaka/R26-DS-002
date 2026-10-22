"""The complete weight history behind the second History Weights graph.

Until 18 Oct 2026 only the CURRENT weights were stored, so the web page could
draw the current session only. Every change is now appended to a per-user
timeline that starts at the user's creation / last reset.
"""
import pytest

sm = pytest.importorskip('layers.l7_personalization.personalization.style_model')


@pytest.fixture
def model(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, 'MODEL_PATH', str(tmp_path / 'style_model.pkl'))
    monkeypatch.setattr(sm, '_users', {})
    return sm


def test_a_reset_starts_the_history_at_the_base_weights(model):
    model.reset_user('user_001')
    h = model.get_user_timeline('user_001')
    assert h['reset'] is True and len(h['points']) == 1
    first = h['points'][0]
    assert first['reason'] == 'reset' and h['since'] == first['at']
    assert abs(sum(first['weights'].values()) - 1.0) < 0.01


def test_every_learned_preference_adds_a_point(model):
    model.reset_user('user_002')
    for style in ('Simple', 'Simple', 'StepByStep'):
        model.learn_style('user_002', 'say it simply', style)
    pts = model.get_user_timeline('user_002')['points']
    assert [p['reason'] for p in pts] == ['reset', 'learned', 'learned', 'learned']
    assert [p['style'] for p in pts[1:]] == ['Simple', 'Simple', 'StepByStep']
    assert pts[2]['weights']['Simple'] > pts[0]['weights']['Simple']
    # the last point is what /profile reports right now
    assert pts[-1]['weights'] == model.get_user_summary('user_002')['history_weights']


def test_a_turn_that_learns_nothing_still_adds_a_point(model):
    """The complete graph grows with every turn, not only with learning."""
    model.reset_user('user_001')
    model.record_turn('user_001', 'Detailed')
    model.learn_style('user_001', 'x', 'Simple')
    model.record_turn('user_001')
    pts = model.get_user_timeline('user_001')['points']
    assert [p['reason'] for p in pts] == ['reset', 'turn', 'learned', 'turn']
    assert pts[1]['weights'] == pts[0]['weights']      # nothing was learned
    assert pts[3]['weights'] == pts[2]['weights']


def test_a_reset_wipes_the_old_history(model):
    model.reset_user('user_001')
    model.learn_style('user_001', 'x', 'Detailed')
    model.reset_user('user_001')
    pts = model.get_user_timeline('user_001')['points']
    assert len(pts) == 1 and pts[0]['reason'] == 'reset'


def test_a_user_from_before_the_timeline_starts_recording_now(model):
    model.reset_user('user_003')
    model._users['user_003'].pop('timeline')          # as saved by the old code
    h = model.get_user_timeline('user_003')
    assert h['reset'] is False and h['points'][0]['reason'] == 'recording started'


def test_the_history_is_capped_but_keeps_its_start(model, monkeypatch):
    monkeypatch.setattr(model, 'TIMELINE_MAX', 5)
    model.reset_user('user_001')
    for _ in range(12):
        model.learn_style('user_001', 'x', 'Detailed')
    pts = model.get_user_timeline('user_001')['points']
    assert len(pts) == 5 and pts[0]['reason'] == 'reset'


def test_the_server_reports_unavailable_instead_of_failing(monkeypatch):
    from layers.l0_voice import voice
    monkeypatch.setattr(voice, 'VOICE_MODE', 'stub')
    out = voice.get_profile_history('user_001')
    assert out['ok'] is False and out['points'] == []
