"""Descriptive X isolation and held-out residual review, with fixed top-n flags."""

from dataclasses import dataclass, replace
from math import isfinite
from .data import Row
from .distance import XDistance
from .prediction import KNNRegressor
from .stats import NumericSummary


@dataclass(frozen=True)
class Isolation:
    row: Row
    score: object
    neighbors: tuple  # (original row ID, unrounded X distance)
    flagged: bool = False
    reason: str = ''


@dataclass(frozen=True)
class Residual:
    row: Row
    target: str
    actual: object
    prediction: object
    signed: object
    score: object
    flagged: bool = False
    reason: str = ''


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def _rank(findings, top, reason):
    _positive(top, 'top')
    ordered = sorted(findings, key=lambda r: (r.score is None,
                                             -r.score if r.score is not None else 0,
                                             r.row.number))
    return tuple(replace(r, flagged=i < top and r.score is not None,
                         reason=reason if i < top and r.score is not None else r.reason)
                 for i, r in enumerate(ordered))


def input_isolation(reference, k=5, top=5):
    """Mean distance to up to k other comparable records; all bounds use X only.

    Neighbor ordering follows existing kNN: rounded distance (12 decimals), ID.
    Means use unrounded distances; score ties use ID. No neighbors => unscored.
    """
    _positive(k, 'k')
    if len({r.number for r in reference.rows}) != len(reference.rows):
        raise ValueError('Reference identities must be unique')
    distance = XDistance.fit(reference)
    findings = []
    for row in reference.rows:
        neighbors = []
        for other in reference.rows:
            if other.number != row.number:
                value = distance(row, other)
                if isfinite(value):
                    neighbors.append((other.number, value))
        neighbors = tuple(sorted(neighbors, key=lambda item: (round(item[1], 12), item[0]))[:k])
        score = NumericSummary.from_values(value for _, value in neighbors).mean
        findings.append(Isolation(row, score, neighbors, reason=(
            'No comparable other records; unscored' if score is None else 'Outside top review ranks')))
    return _rank(findings, top, f'Top {top} mean nearest-neighbor X distances; review only')


def outcome_residuals(training, test, target, k=5, top=5):
    """One target per ranking; same kNN semantics as Guess, with masked queries.

    Missing test targets are unscored. As in Guess, an observed target without
    eligible training neighbors raises instead of inventing a prediction.
    """
    _positive(top, 'top')
    if training.columns != test.columns:
        raise ValueError('Training and test schemas must match')
    if len({r.number for r in test.rows}) != len(test.rows):
        raise ValueError('Test identities must be unique')
    if {r.number for r in training.rows} & {r.number for r in test.rows}:
        raise ValueError('Training and test identities overlap')
    model = KNNRegressor.fit(training, k=k)
    column = model.target_column(target)
    if not any(r.cells[column.index] is not None for r in training.rows):
        raise ValueError(f'No observed training target: {target}')
    findings = []
    for row in test.rows:
        if row.cells[column.index] is None:
            findings.append(Residual(row, target, None, None, None, None,
                                     reason='Missing observed target; unscored'))
            continue
        query = Row(row.number, tuple(row.cells[c.index] if c.role == 'x' else None
                                      for c in test.columns))
        prediction = model.predict(query, target)
        actual = row.cells[column.index]
        signed = actual - prediction.value
        findings.append(Residual(row, target, actual, prediction, signed, abs(signed),
                                 reason='Outside top review ranks'))
    return _rank(findings, top, f'Top {top} absolute held-out {target} residuals; review only')
