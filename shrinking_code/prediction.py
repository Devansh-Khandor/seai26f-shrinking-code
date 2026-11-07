"""Training-only k-nearest-neighbors regression for any numerical goal."""

from dataclasses import dataclass
from math import isfinite
from statistics import pstdev
from .distance import XDistance
from .stats import NumericSummary


@dataclass(frozen=True)
class Prediction:
    value: float
    minimum: float
    maximum: float
    stddev: float
    neighbors: tuple


@dataclass(frozen=True)
class KNNRegressor:
    training: object
    distance: XDistance
    k: int

    @classmethod
    def fit(cls, training, k=5, p=2):
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError('k must be a positive integer')
        if not training.rows:
            raise ValueError('Training table is empty')
        if len({r.number for r in training.rows}) != len(training.rows):
            raise ValueError('Training row identities must be unique')
        return cls(training, XDistance.fit(training, p), k)

    def target_column(self, target):
        for column in self.training.role('y'):
            if column.name == target and column.numeric:
                return column
        raise ValueError(f'Unavailable numerical goal: {target}')

    def predict(self, query, target):
        """Reject training identities; return unweighted mean and neighbor dispersion.

        Row numbers identify records within one source dataset, not features.
        Queries must share its schema and use an identity absent from training.
        """
        column = self.target_column(target)
        if len(query.cells) != len(self.training.columns):
            raise ValueError('Query must use the training schema')
        if any(r.number == query.number for r in self.training.rows):
            raise ValueError('Query row identity overlaps training data')
        candidates = []
        for row in self.training.rows:
            if row.cells[column.index] is not None:
                distance = self.distance(query, row)
                if isfinite(distance):
                    # Stabilize mathematical ties against normalization roundoff.
                    candidates.append((round(distance, 12), row.number, row))
        neighbors = tuple(item[2] for item in sorted(candidates)[:self.k])
        if not neighbors:
            raise ValueError(f'No eligible neighbors with observed {target} and comparable X')
        values = [r.cells[column.index] for r in neighbors]
        summary = NumericSummary.from_values(values)
        return Prediction(summary.mean, summary.minimum, summary.maximum,
                          pstdev(values), neighbors)
