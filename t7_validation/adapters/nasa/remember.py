#!/usr/bin/env python3
"""NASA T7 Remember adapter: original T7 algorithms -> common ANSWER contract.

The expected-answer oracle is not imported by this adapter.
Usage: python3 t7_validation/adapters/nasa/remember.py PATH_TO_NASA_CSV
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NASA_REMEMBER = ROOT / "human_t7" / "nasa" / "remember.py"
# Load this module under a unique name. Apache also has a remember.py, and
# unittest discovery loads Apache's module first in the same Python process.
_spec = importlib.util.spec_from_file_location("human_t7_nasa_remember", NASA_REMEMBER)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load NASA implementation: {NASA_REMEMBER}")
_nasa_remember = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_nasa_remember)
GOALS = _nasa_remember.GOALS
load_projects = _nasa_remember.load_projects
rank_projects = _nasa_remember.rank_projects
summarize = _nasa_remember.summarize


def answers(csv_path):
    projects = load_projects(csv_path)
    stats = summarize(projects)
    summary = {}
    for goal, direction in GOALS.items():
        s = stats[goal]
        summary[goal] = {
            "mean": s["mean"],
            "median": s["median"],
            "min": s["minimum"],
            "max": s["maximum"],
            "best": s["maximum"] if direction > 0 else s["minimum"],
            "worst": s["minimum"] if direction > 0 else s["maximum"],
        }
    top = [project["row_number"] for project in rank_projects(projects)[:10]]
    return summary, {"top": top}


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: remember.py PATH_TO_NASA_CSV")
    summary, favorable = answers(Path(sys.argv[1]))
    print("== NASA-01 ==")
    print("Historical project size, effort and defects summary")
    print("Goal columns: 3 (Kloc+, Effort-, Defects-)")
    print("ANSWER NASA-01 " + json.dumps(summary, sort_keys=True))
    print("== NASA-02 ==")
    print("Ten favorable projects ranked by original T7 distance-to-heaven")
    print("Top projects reported: 10")
    print("ANSWER NASA-02 " + json.dumps(favorable, sort_keys=True))


if __name__ == "__main__":
    main()
