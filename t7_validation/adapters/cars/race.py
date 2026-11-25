#!/usr/bin/env python3
"""Cars T7 Race -> the common CAR-20 ANSWER contract (prediction race only).

Calls the shared library's prediction_race(training, test, "Mpg+") exactly as
human_t7/cars/race.py does, but on the common every-fifth-row split instead of the
original seeded 80/20 split. prediction_race fits its 5-NN on the training rows and
uses the training-mean baseline; the held-out targets are only used to score errors.
CAR-19 (selection race) is not part of this adapter.

The verdict is a reporting label taken from the library's own paired comparison:
"no clear difference" when its compare() finds the error samples the same (or cannot
decide), otherwise "better"/"worse" by the median error.
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TARGET, common_split, load  # noqa: E402
from shrinking_code.baselines import prediction_race  # noqa: E402


def answers(csv_path):
    training, heldout = common_split(load(csv_path))
    race = prediction_race(training, heldout, TARGET)
    ev = race.evaluation
    knn = statistics.median(p.knn_error for p in race.pairs)
    mean = statistics.median(p.mean_error for p in race.pairs)
    if race.comparison.same is not False or knn == mean:
        verdict = "no clear difference"
    else:
        verdict = "better" if knn < mean else "worse"
    return {
        "predictions": {str(row.number): p.value for row, p in ev.predictions},
        "baseline": ev.baseline,
        "mae_model": ev.mae,
        "mae_baseline": ev.baseline_mae,
        "verdict": verdict,
    }


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: race.py PATH_TO_CARS_CSV")
    answer = answers(Path(sys.argv[1]))
    print("== CAR-20 ==")
    print("Shared-library 5-NN versus the training-mean estimate; every fifth row held out")
    print(f"kNN MAE {answer['mae_model']:.3f}; historical-mean MAE {answer['mae_baseline']:.3f}; "
          f"verdict: {answer['verdict']}")
    print("ANSWER CAR-20 " + json.dumps(answer, sort_keys=True))


if __name__ == "__main__":
    main()
