"""Static guard against the D1 bug class returning.

The p90/p75 conflation was not one mistake in one file. It appeared three
times, in three layers, written months apart:

    l2_select/select.py   glyph_p90 on a whole frame -> capture_verdict
    l4b_body/body.py      glyph_p90 on a region crop -> rescale_to_optimum
    l3_segment/segment.py glyph_p90 on an article crop -> capture_verdict

Two of those were wrong. Each was locally plausible, which is exactly why it
survived review — the names were similar enough that nothing looked odd.

Unit tests cannot catch this: passing a p90 float where p75 is expected is
type-correct and returns a perfectly sensible-looking verdict. So this test
reads the source instead.

The rule is a POSITIVE requirement, not a blacklist. A first attempt banned
arguments containing "p90" and it did NOT catch the real bug, because the
original line was `capture_verdict(p)` — the variable was named `p`, so a
blacklist saw nothing wrong. A blacklist only catches the version of the
mistake obvious enough to spot by eye.

So: the argument to capture_verdict must be a call to glyph_p75() or a name
containing p75. Anything else fails, including innocent names like `p` or `g`.
Being forced to name the variable after the percentile it holds is the entire
defence.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SEARCH_DIRS = ('core', 'layers', 'app')
# function -> (substring the argument must contain, the producing function)
REQUIRED = {
    'capture_verdict':  ('p75', 'glyph_p75'),
    'scale_for_target': ('p90', 'glyph_p90'),
}


def _python_files():
    for d in SEARCH_DIRS:
        yield from (ROOT / d).rglob('*.py')


def _arg_source(node, src_lines):
    """Readable text for the first positional argument of a call."""
    try:
        return ast.get_source_segment('\n'.join(src_lines), node) or ast.dump(node)
    except Exception:
        return ast.dump(node)


def _offences(fn_name, needle, producer):
    out = []
    for f in _python_files():
        src = f.read_text(encoding='utf-8')
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            pytest.fail(f'{f} does not parse: {e}')
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, 'attr', '')
            if name != fn_name or not node.args:
                continue
            arg = ast.get_source_segment(src, node.args[0]) or ''
            if needle not in arg.lower():
                out.append(f'{f.relative_to(ROOT)}:{node.lineno}  '
                           f'{fn_name}({arg}) — the argument must be a '
                           f'{producer}() result and must be named so')
    return out


def test_capture_verdict_is_always_given_a_p75():
    o = _offences('capture_verdict', *REQUIRED['capture_verdict'])
    assert not o, ('capture_verdict takes glyph_p75. See core/config.py:\n  '
                   + '\n  '.join(o))


def test_scale_for_target_is_always_given_a_p90():
    """The mirror image: the OCR resize target is calibrated in p90."""
    o = _offences('scale_for_target', *REQUIRED['scale_for_target'])
    assert not o, ('scale_for_target takes glyph_p90:\n  ' + '\n  '.join(o))


def _would_flag(source: str, fn_name='capture_verdict', needle='p75') -> int:
    tree = ast.parse(source)
    n = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else getattr(fn, 'attr', '')
        if name != fn_name or not node.args:
            continue
        arg = ast.get_source_segment(source, node.args[0]) or ''
        if needle not in arg.lower():
            n += 1
    return n


def test_the_guard_catches_the_bug_as_it_was_actually_written():
    """A guard that cannot fail is decoration. This is the original line from
    l3_segment/segment.py, verbatim in shape — a blacklist on 'p90' missed it
    because the variable was called `p`."""
    historical = 'p = glyph_p90(crop) if crop.size else None\nv, note = capture_verdict(p)\n'
    assert _would_flag(historical) == 1, 'the guard would have missed the real bug'

    obvious = 'p90 = glyph_p90(crop)\nv, n = capture_verdict(p90)\n'
    assert _would_flag(obvious) == 1

    correct = 'p75 = glyph_p75(crop)\nv, n = capture_verdict(p75)\n'
    assert _would_flag(correct) == 0, 'the guard fires on correct code'

    inline = 'v, n = capture_verdict(glyph_p75(crop))\n'
    assert _would_flag(inline) == 0
