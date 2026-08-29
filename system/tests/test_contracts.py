"""Contract tests. If these fail, layers will not fit together."""
from core.schemas import Box, Frame, Region, Article, Document


def test_box_dims():
    b = Box(x1=10, y1=20, x2=110, y2=220)
    assert b.w == 100 and b.h == 200


def test_article_fields_independent():
    """Layer 4A and 4B must write DIFFERENT fields — neither may clobber
    the other. This is the whole reason the split exists."""
    a = Article(index=0, box=Box(x1=0, y1=0, x2=10, y2=10))
    a.title = 'T'
    a.body = 'B'
    assert a.title == 'T' and a.body == 'B'


def test_document_roundtrip():
    d = Document(articles=[Article(index=0,
                                   box=Box(x1=0, y1=0, x2=1, y2=1))])
    assert Document(**d.model_dump()).articles[0].index == 0


def test_assemble_drops_rejected():
    from layers.l5_assemble.assemble import assemble
    good = Article(index=0, box=Box(x1=0, y1=0, x2=1, y2=1),
                   body='text', verdict='ok')
    bad = Article(index=1, box=Box(x1=0, y1=0, x2=1, y2=1),
                  verdict='reject', note='too far')
    doc = assemble([good, bad], ['f.jpg'])
    assert len(doc.articles) == 1
    assert any('too far' in w for w in doc.warnings)


def test_stubs_do_not_break_flow():
    from layers.l4a_title import title as l4a
    from layers.l6_speech import speech as l6
    a = Article(index=0, box=Box(x1=0, y1=0, x2=1, y2=1))
    assert l4a.extract(None, a) is a
    assert l6.speak(Document()) is None
