#!/usr/bin/env python3
"""Evaluate NASA T7 Remember against Jack's unchanged common checker."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = Path(os.environ.get("T7_HARNESS_ROOT", str(ROOT))).resolve()
sys.path.insert(0, str(HARNESS_ROOT))
from harness.checks import CheckResult, contract_ok  # noqa: E402
from harness.semantic import evaluate  # noqa: E402

ADAPTER = ROOT / "t7_validation" / "adapters" / "nasa" / "remember.py"
DEFAULT_CSV = ROOT / "data" / "raw" / "nasa93dem.csv"
REQUESTS = ("NASA-01", "NASA-02")


def execute(csv_path):
    proc = subprocess.run(
        [sys.executable, str(ADAPTER), str(csv_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    success = proc.returncode == 0
    return CheckResult(
        runs=success,
        contract=success and contract_ok(proc.stdout, REQUESTS),
        timed_out=False,
        seconds=0.0,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--evidence", type=Path, default=ROOT / "t7_validation/results/nasa_remember.json")
    args = parser.parse_args()
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    base = execute(args.csv)
    print(f"Execution: {'PASS' if base.runs else 'FAIL'}")
    print(f"Format contract: {'PASS' if base.contract else 'FAIL'}")
    if not base.runs:
        print("Adapter error:", base.stderr)
    result = evaluate(REQUESTS, args.csv, base, execute, evidence=args.evidence)
    print(f"Semantic skill verdict: {result.status}")
    for item in result.requests:
        print(f"{item.request_id}: {item.status} ({item.passed}/{item.total} checks)")
        for failure in item.failed:
            print("  FAILURE:", failure)
        for note in item.notes:
            print("  NOTE:", note)
    print("Prudence runs:", json.dumps(result.runs, sort_keys=True))
    print("Evidence:", args.evidence)
    if not base.runs or not base.contract or result.status != "pass":
        sys.exit(1)


if __name__ == "__main__":
    main()
