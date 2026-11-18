#!/usr/bin/env python3
"""T7 Apache Certify adapter for the unchanged shared prudence evaluator.

Uses human_t7/apache/certify.py to compute all group statistics and the
statistical assessment. Does not import the oracle or test expected values.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "human_t7" / "apache"))

from remember import load_configurations  # noqa: E402
from certify import compare_groups  # noqa: E402


def comparison_answer(rows, setting):
    result = compare_groups(rows, setting)
    # A conservative observational decision rule: require both the T7
    # permutation test and a large signed dominance effect. A non-different
    # verdict means not enough evidence under this modeling rule, NOT proof
    # of equivalence or of no causal effect.
    distinguishable = (
        result["permutation_p"] < 0.05
        and abs(result["cliffs_delta"]) >= 0.474
    )
    return {
        "groups": {
            "0": {
                "n": result["n0"],
                "median": result["median0"],
                "mean": result["mean0"],
            },
            "1": {
                "n": result["n1"],
                "median": result["median1"],
                "mean": result["mean1"],
            },
        },
        "comparisons": [{
            "a": "0",
            "b": "1",
            "verdict": "different" if distinguishable else "same",
        }],
    }


def answers(csv_path):
    rows = load_configurations(csv_path)
    return (
        comparison_answer(rows, "keepAlive"),
        comparison_answer(rows, "accessLog"),
    )


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: certify.py PATH_TO_APACHE_CSV")
    keepalive, accesslog = answers(Path(sys.argv[1]))
    for rid, label, result in (
        ("APACHE-05", "keepAlive: enabled versus disabled", keepalive),
        ("APACHE-06", "accessLog: enabled versus disabled", accesslog),
    ):
        print(f"== {rid} ==")
        print(label)
        print("Descriptive group statistics and an exploratory verdict")
        print("Observational evidence does not establish causation")
        print(f"ANSWER {rid} " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
