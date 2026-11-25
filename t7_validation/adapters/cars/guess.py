#!/usr/bin/env python3
"""Cars T7 Guess -> the common CAR-03 ANSWER contract (fuel-efficiency prediction only).

Prediction is the shared library's 5-NN regressor exactly as human_t7/cars/guess.py
uses it (KNNRegressor.fit(training, k=5), model.predict(row, "Mpg+")), fitted on the
common training rows only. Held-out rows are queried with every goal cell masked, and
their actual values are read only after all predictions are made, to report the MAE.
CAR-04 (weight interval) is not part of this adapter.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import K, TARGET, common_split, load, masked  # noqa: E402
from shrinking_code.evaluation import mae  # noqa: E402
from shrinking_code.prediction import KNNRegressor  # noqa: E402


def answers(csv_path):
    table = load(csv_path)
    training, heldout = common_split(table)
    model = KNNRegressor.fit(training, k=K)
    predictions = {r.number: model.predict(masked(r, table.columns), TARGET).value for r in heldout.rows}
    column = model.target_column(TARGET)
    error = mae([r.cells[column.index] for r in heldout.rows], [predictions[r.number] for r in heldout.rows])
    return {"predictions": {str(n): v for n, v in predictions.items()}, "mae": error}


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: guess.py PATH_TO_CARS_CSV")
    answer = answers(Path(sys.argv[1]))
    print("== CAR-03 ==")
    print("Shared-library 5-NN fuel-efficiency estimates; every fifth row held out")
    print(f"Held-out predictions: {len(answer['predictions'])}; MAE: {answer['mae']:.3f}")
    print("ANSWER CAR-03 " + json.dumps(answer, sort_keys=True))


if __name__ == "__main__":
    main()
