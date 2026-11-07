"""Fixed holdout evaluation for Guess, not a treatment-comparison framework."""

from dataclasses import dataclass
from random import Random
from statistics import mean
from .data import Table
from .stats import summarize


def train_test_split(table, seed=42, train_fraction=0.8):
    """Shuffle row identities, take floor(fraction*n) for training, retain IDs."""
    if not 0 < train_fraction < 1:
        raise ValueError('Training fraction must be between 0 and 1')
    if len({r.number for r in table.rows}) != len(table.rows):
        raise ValueError('Row identities must be unique')
    rows = sorted(table.rows, key=lambda r: r.number)
    Random(seed).shuffle(rows)
    cut = int(len(rows) * train_fraction)
    if not 0 < cut < len(rows):
        raise ValueError('Split requires nonempty training and test sets')
    return tuple(Table(table.columns, tuple(sorted(part, key=lambda r: r.number)))
                 for part in (rows[:cut], rows[cut:]))


def mae(actual, predicted):
    actual, predicted = tuple(actual), tuple(predicted)
    if not actual or len(actual) != len(predicted):
        raise ValueError('MAE requires equal nonempty sequences')
    return mean(abs(a - b) for a, b in zip(actual, predicted))


@dataclass(frozen=True)
class Evaluation:
    training_count: int
    test_count: int
    mae: float
    baseline: float
    baseline_mae: float
    predictions: tuple


def evaluate(model, test, target):
    """Evaluate both predictors on identical observed-target test rows; fail loudly
    if a scored row cannot be predicted rather than silently changing coverage.
    """
    if test.columns != model.training.columns:
        raise ValueError('Test and training schemas must match')
    if {r.number for r in test.rows} & {r.number for r in model.training.rows}:
        raise ValueError('Test row identities overlap training data')
    column = model.target_column(target)
    baseline = summarize(model.training, column).mean
    if baseline is None:
        raise ValueError(f'No observed training target: {target}')
    predictions = tuple((row, model.predict(row, target)) for row in test.rows
                        if row.cells[column.index] is not None)
    actual = [row.cells[column.index] for row, _ in predictions]
    return Evaluation(len(model.training.rows), len(actual),
                      mae(actual, [p.value for _, p in predictions]), baseline,
                      mae(actual, [baseline] * len(actual)), predictions)
