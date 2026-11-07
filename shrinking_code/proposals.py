"""Assess X-only proposals, then accept separately observed completions."""

from dataclasses import dataclass
from math import ceil
from .prediction import KNNRegressor
from .streaming import Inventory


@dataclass(frozen=True)
class Risk:
    target: str
    prediction: object
    threshold: float
    high: bool


@dataclass(frozen=True)
class Proposal:
    number: int
    reference_count: int
    risks: tuple


class ProposalStream:
    """High predicted minimizing goals exceed current nearest-rank upper quartile.

    The fixed policy is Q75 = sorted observed values[ceil(.75*n)-1], strict >.
    No target labels or ignored cells may appear in assessment queries.
    """
    def __init__(self, initial, targets, k=5):
        self.inventory = Inventory(initial)
        self.targets = tuple(targets)
        self.k = k
        model = KNNRegressor.fit(self.inventory.table, k=k)
        if not self.targets or len(set(self.targets)) != len(self.targets):
            raise ValueError('Targets must be nonempty and unique')
        for target in self.targets:
            if model.target_column(target).ideal != 0:
                raise ValueError('High-risk targets must be minimizing numerical goals')

    def assess(self, query):
        reference = self.inventory.table
        if len(query.cells) != len(reference.columns) or any(
                query.cells[c.index] is not None for c in reference.columns if c.role != 'x'):
            raise ValueError('Proposal must expose X only; mask goals and ignored fields')
        model = KNNRegressor.fit(reference, k=self.k)
        risks = []
        for target in self.targets:
            c = model.target_column(target)
            values = sorted(r.cells[c.index] for r in reference.rows if r.cells[c.index] is not None)
            if not values:
                raise ValueError(f'No observed reference values for {target}')
            threshold = values[ceil(.75*len(values))-1]
            prediction = model.predict(query, target)
            risks.append(Risk(target, prediction, threshold, prediction.value > threshold))
        return Proposal(query.number, len(reference.rows), tuple(risks))

    def complete(self, row):
        """Only call when outcomes are observed; reuse validated Inventory updates."""
        return self.inventory.accept(row.cells, number=row.number)
