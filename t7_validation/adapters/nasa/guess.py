#!/usr/bin/env python3
"""NASA T7 Guess -> Jack's common NASA-03/04 ANSWER contract.

Uses the original NASA 5-NN model, NOT harness oracle or hand-coded answers.
Every fifth row is held out per the common test protocol.
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NASA_DIR = ROOT / "human_t7" / "nasa"


def _load_module(alias, path):
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import NASA original module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Original guess.py imports `from remember import load_projects`. Both Apache and
# NASA have remember.py, so isolate the dependency during original module import.
_original_remember = _load_module("t7_nasa_guess_dependency_remember", NASA_DIR / "remember.py")
_previous = sys.modules.get("remember")
try:
    sys.modules["remember"] = _original_remember
    _original_guess = _load_module("t7_nasa_original_guess", NASA_DIR / "guess.py")
finally:
    if _previous is None:
        sys.modules.pop("remember", None)
    else:
        sys.modules["remember"] = _previous


def answers(csv_path):
    projects = _original_remember.load_projects(csv_path)
    training = [r for r in projects if r["row_number"] % 5]
    heldout = [r for r in projects if r["row_number"] % 5 == 0]
    if not training or not heldout:
        raise ValueError("Common Guess evaluation requires training and held-out rows")
    features = _original_guess.get_features(projects)
    if not features:
        raise ValueError("Cannot estimate from an empty feature set")

    effort_predictions = {}
    defect_predictions = {}
    effort_abs_errors = []
    for proposal in heldout:
        rid = str(proposal["row_number"])
        effort = _original_guess.predict(proposal, training, features, "Effort-")
        defects = _original_guess.predict(proposal, training, features, "Defects-")
        effort_predictions[rid] = effort["estimate"]
        defect_predictions[rid] = {
            "estimate": defects["estimate"],
            "low": defects["low"],
            "high": defects["high"],
        }
        # Only score the model *after* making predictions. Held-out actual
        # values never enter the training set or the prediction functions.
        effort_abs_errors.append(abs(proposal["Effort-"] - effort["estimate"]))

    return (
        {"predictions": effort_predictions,
         "mae": sum(effort_abs_errors) / len(effort_abs_errors)},
        {"predictions": defect_predictions},
    )


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: guess.py PATH_TO_NASA_CSV")
    effort, defects = answers(Path(sys.argv[1]))
    print("== NASA-03 ==")
    print("5-NN effort prediction; every fifth row held out")
    print(f"Held-out predictions: {len(effort['predictions'])}; MAE: {effort['mae']:.2f}")
    print("ANSWER NASA-03 " + json.dumps(effort, sort_keys=True))
    print("== NASA-04 ==")
    print("5-NN defects predictions with descriptive neighbor outcome ranges")
    print(f"Held-out estimates and ranges: {len(defects['predictions'])}")
    print("ANSWER NASA-04 " + json.dumps(defects, sort_keys=True))


if __name__ == "__main__":
    main()
