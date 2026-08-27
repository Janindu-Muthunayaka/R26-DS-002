"""The system's memory, tested without sleeping.

`SessionStore` takes an injectable clock precisely so expiry can be asserted
rather than waited for. A test that sleeps for thirty minutes is a test that
gets deleted.
"""
from core.schemas import Article, Box, Document
from core.session import SessionStore


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def _doc(body='පෙළ'):
    return Document(articles=[Article(index=0, body=body,
                                      box=Box(x1=0, y1=0, x2=1, y2=1))])


def test_put_then_get():
    s = SessionStore()
    s.put('job1', _doc('එක'))
    assert s.get('job1').articles[0].body == 'එක'


def test_missing_job_is_none_not_an_error():
    """A miss must be answerable, not raisable — `/ask` turns it into a
    sentence the user hears."""
    assert SessionStore().get('nosuchjob') is None
    assert SessionStore().get('') is None


def test_entries_expire():
    c = Clock()
    s = SessionStore(ttl_s=60, clock=c)
    s.put('j', _doc())
    c.advance(59)
    assert s.get('j') is not None
    c.advance(2)
    assert s.get('j') is None, 'expired entry was served'


def test_expiry_and_miss_are_indistinguishable():
    """Deliberate: the user's next action is the same either way."""
    c = Clock()
    s = SessionStore(ttl_s=10, clock=c)
    s.put('j', _doc())
    c.advance(11)
    assert s.get('j') == s.get('never-existed') == None


def test_oldest_is_evicted_at_the_cap():
    s = SessionStore(max_items=3)
    for i in range(5):
        s.put(f'j{i}', _doc(str(i)))
    assert len(s) == 3
    assert s.get('j0') is None and s.get('j1') is None
    assert s.get('j4') is not None


def test_reput_refreshes_position_and_time():
    c = Clock()
    s = SessionStore(ttl_s=100, max_items=2, clock=c)
    s.put('a', _doc('A'))
    c.advance(50)
    s.put('b', _doc('B'))
    s.put('a', _doc('A2'))          # a is now the newest
    s.put('c', _doc('C'))           # evicts b, not a
    assert s.get('b') is None
    assert s.get('a').articles[0].body == 'A2'
    assert s.get('c') is not None


def test_purge_reports_what_it_dropped():
    """Every screening step must say what it removed. Silent screening is
    how the `g0` null marker survived as a measurement of zero."""
    c = Clock()
    s = SessionStore(ttl_s=10, clock=c)
    s.put('a', _doc()); s.put('b', _doc())
    c.advance(11)
    assert s.purge() == 2


def test_ttl_zero_disables_expiry():
    c = Clock()
    s = SessionStore(ttl_s=0, clock=c)
    s.put('j', _doc())
    c.advance(10_000_000)
    assert s.get('j') is not None


def test_drop_and_contains():
    s = SessionStore()
    s.put('j', _doc())
    assert 'j' in s
    assert s.drop('j') is True
    assert s.drop('j') is False
    assert 'j' not in s
