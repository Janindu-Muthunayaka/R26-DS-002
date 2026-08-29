"""
Health check — verify all services are running.

    python tools/check_services.py

Checks:
  1. Main server (port 8000)    /health
  2. Voice service (port 8101)  /health
  3. RAG service (port 8102)    /health
"""
import json
import sys
import urllib.error
import urllib.request

SERVICES = [
    ('Main Pipeline', 'http://127.0.0.1:8000/health', 8000),
    ('Voice (Component 4)', 'http://127.0.0.1:8101/health', 8101),
    ('RAG (Component 3)', 'http://127.0.0.1:8102/health', 8102),
]


def check(name, url, port):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read().decode('utf-8'))
            print(f'  ✅ {name:25s} port {port}  {json.dumps(data, indent=None)[:120]}')
            return True
    except urllib.error.URLError:
        print(f'  ❌ {name:25s} port {port}  NOT RUNNING')
        return False
    except Exception as e:
        print(f'  ⚠  {name:25s} port {port}  {type(e).__name__}: {e}')
        return False


def main():
    print()
    print('Sinhala Reader — Service Health Check')
    print('=' * 55)

    results = [check(n, u, p) for n, u, p in SERVICES]

    print()
    up = sum(results)
    total = len(results)
    if up == total:
        print(f'All {total} services are UP ✅')
        print()
        print('System is ready. To connect the phone:')
        print('  adb reverse tcp:8000 tcp:8000')
    else:
        print(f'{up}/{total} services are UP')
        print()
        print('Start missing services:')
        if not results[0]:
            print('  cd system && python -m app.server --root E:/RP/corpus/Sinhala_OCR_Correction_v2')
        if not results[1]:
            print('  python services/voice/app.py --port 8101')
        if not results[2]:
            print('  python services/rag/app.py --port 8102')
    print()
    sys.exit(0 if up == total else 1)


if __name__ == '__main__':
    main()
