#!/usr/bin/env python3
"""Apache T7 Flow adapter (APACHE-07/08) for the common prudence harness.

Uses the original human-directed StreamMonitor; conforms to the common
last-20%-by-row-position arrival protocol, not the original random split.
Never imports checker/oracle code to construct its answers.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "human_t7" / "apache"))

from remember import GOAL, load_configurations  # noqa: E402
from flow import StreamMonitor  # noqa: E402


def checkpoint(monitor, after):
    summary = monitor.snapshot()
    return {
        "after": after,
        "summary": {
            GOAL: {
                "n": summary["count"],
                "mean": summary["mean"],
                "min": summary["minimum"],
                "max": summary["maximum"],
            }
        },
        "shortlist": [r["row_number"] for r in monitor.top_configurations()],
    }


def answers(csv_path):
    rows = load_configurations(csv_path)
    # Match the shared harness protocol: round(20% of the row count),
    # with .5 rounding upward. The last k *file positions* are arrivals.
    k = int(0.2 * len(rows) + 0.5)
    if not 0 < k < len(rows):
        raise ValueError("Need history and arrivals")
    history, arrivals = rows[:-k], rows[-k:]
    monitor = StreamMonitor(history)
    half = k // 2
    checkpoints = [checkpoint(monitor, 0)]
    assessments = []
    for i, row in enumerate(arrivals, start=1):
        event = monitor.process_arrival(row)
        assessments.append({"row": event["row_number"], "alert": bool(event["alert"])})
        if i == half or i == k:
            checkpoints.append(checkpoint(monitor, i))
    return {"assessments": assessments}, {"checkpoints": checkpoints}


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: flow.py PATH_TO_APACHE_CSV")
    alerts, summary = answers(Path(sys.argv[1]))
    print("== APACHE-07 ==")
    print("Evaluate each new benchmark against its previously observed history")
    print(f"Arrivals assessed: {len(alerts['assessments'])}")
    print("ANSWER APACHE-07 " + json.dumps(alerts, sort_keys=True))
    print("== APACHE-08 ==")
    print("Maintain summary and best-five shortlist at 0, halfway, all arrivals")
    print(f"Checkpoints produced: {len(summary['checkpoints'])}")
    print("ANSWER APACHE-08 " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
