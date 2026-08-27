"""Point the PHONE path at guidance_verdict. Run from E:\\RP\\R26-DS-002\\system.

    python fix_callers.py

Two production call sites, two different questions:

  layers/l2_select/select.py    picks the best of the captured frames and
                                produces the note the LISTENER hears.
                                Question: "should the user move?"  -> 20
                                CHANGED HERE.

  layers/l3_segment/segment.py  gates a whole page before segmenting it.
                                Question: "is this page good enough to OCR
                                at all?"  -> 25, and this is the call
                                test_corpus_verdict reproduces.
                                LEFT ALONE, deliberately.

Refuses to write unless it finds exactly what it expects. Backs up to .bak5.
"""
import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
SEL = ROOT / 'layers' / 'l2_select' / 'select.py'


def main():
    if not SEL.exists():
        sys.exit(f'NOT FOUND: {SEL}\nRun this from the system folder.')

    txt = SEL.read_text(encoding='utf-8')
    n = txt.count('capture_verdict')
    if n != 2:
        sys.exit(f'REFUSING: select.py mentions capture_verdict {n} times, '
                 f'expected 2 (the import and the call). Nothing written - '
                 f'send me the file.')
    if 'guidance_verdict' in txt:
        sys.exit('Already patched. Nothing to do.')

    out = txt.replace('capture_verdict', 'guidance_verdict')

    SEL.with_suffix('.py.bak5').write_text(txt, encoding='utf-8')
    SEL.write_text(out, encoding='utf-8')
    print('patched layers/l2_select/select.py  (backup select.py.bak5)')
    print('  import  -> guidance_verdict')
    print('  call    -> guidance_verdict(p75)')
    print('\nleft alone on purpose: layers/l3_segment/segment.py')
    print('  that one is the 168-page page gate at 25 that Chapter 4 cites')
    print('\nNow run:  python -m pytest tests -q')


if __name__ == '__main__':
    main()
