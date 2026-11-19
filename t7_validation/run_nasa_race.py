#!/usr/bin/env python3
"""Run NASA T7 Race through Jack's unmodified checker."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = Path(os.environ.get('T7_HARNESS_ROOT', str(ROOT))).resolve()
sys.path.insert(0, str(HARNESS_ROOT))
from harness.checks import CheckResult, contract_ok  # noqa: E402
from harness.semantic import evaluate  # noqa: E402

ADAPTER = ROOT / 't7_validation/adapters/nasa/race.py'
DEFAULT_CSV = ROOT / 'data/raw/nasa93dem.csv'
REQUESTS = ('NASA-19', 'NASA-20')


def execute(csv_path):
    p = subprocess.run([sys.executable, str(ADAPTER), str(csv_path)], cwd=ROOT,
                       capture_output=True, text=True, timeout=90)
    success = p.returncode == 0
    return CheckResult(runs=success, contract=success and contract_ok(p.stdout, REQUESTS),
                       timed_out=False, seconds=0., stdout=p.stdout, stderr=p.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path, default=DEFAULT_CSV)
    parser.add_argument('--evidence', type=Path,
                        default=ROOT/'t7_validation/results/nasa_race.json')
    args = parser.parse_args()
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    base = execute(args.csv)
    print('Execution:', 'PASS' if base.runs else 'FAIL')
    print('Format contract:', 'PASS' if base.contract else 'FAIL')
    if not base.runs:
        print('Adapter error:', base.stderr)
    verdict = evaluate(REQUESTS, args.csv, base, execute, evidence=args.evidence)
    print('Semantic skill verdict:', verdict.status)
    for req in verdict.requests:
        print(f'{req.request_id}: {req.status} ({req.passed}/{req.total} checks)')
        for failure in req.failed:
            print('  FAILURE:', failure)
        for note in req.notes:
            print('  NOTE:', note)
    print('Prudence runs:', json.dumps(verdict.runs, sort_keys=True))
    print('Evidence:', args.evidence)
    if not base.runs or not base.contract or verdict.status != 'pass':
        sys.exit(1)


if __name__ == '__main__':
    main()
