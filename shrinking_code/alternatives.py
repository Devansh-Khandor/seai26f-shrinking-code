"""Observed historical alternatives; no synthetic repairs or causal claims."""

from dataclasses import dataclass
from math import isfinite
from .explanation import Explainer


@dataclass(frozen=True)
class TargetStatus:
    satisfied: tuple
    failed: tuple
    unknown: tuple

    @property
    def count(self):
        return len(self.satisfied)


def target_status(row, columns, targets):
    """Inclusive direction-aware thresholds; missing outcomes remain unknown.

    Targets map numerical goal names to finite thresholds; subsets are allowed.
    """
    goals = {c.name: c for c in columns if c.role == 'y' and c.numeric}
    if not targets or any(name not in goals for name in targets):
        raise ValueError('Targets must name one or more numerical goals')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v)
           for v in targets.values()):
        raise ValueError('Targets must be finite numbers')
    satisfied, failed, unknown = [], [], []
    for name, c in goals.items():
        if name not in targets:
            continue
        value = row.cells[c.index]
        if value is None:
            unknown.append(name)
        elif (value <= targets[name] if c.ideal == 0 else value >= targets[name]):
            satisfied.append(name)
        else:
            failed.append(name)
    return TargetStatus(tuple(satisfied), tuple(failed), tuple(unknown))


@dataclass(frozen=True)
class CharacteristicChange:
    name: str
    before: object
    after: object


@dataclass(frozen=True)
class Alternative:
    comparison: object  # Existing PairExplanation: a=source, b=alternative.
    changed: tuple
    unknown_x: tuple
    distance: float
    source_targets: object
    alternative_targets: object
    ranking_key: tuple
    rationale: str

    @property
    def improvement(self):
        return self.comparison.a.score - self.comparison.b.score

    @property
    def pareto_dominates(self):
        return bool(self.comparison.favors_b) and not (
            self.comparison.favors_a or self.comparison.unknown)


class Alternatives:
    def __init__(self, reference):
        self.explainer = Explainer(reference)

    def worst(self):
        """Worst complete-goal score; ties choose smallest original ID."""
        ranked = self.explainer.ranked
        return min(ranked, key=lambda item: (-item[1], item[0].number))[0] if ranked else None

    def _candidates(self, number, targets=None):
        e = self.explainer
        source = e.explain(number)
        if source.score is None:
            raise ValueError('Source requires complete observed goals')
        before = target_status(source.row, e.reference.columns, targets) if targets is not None else None
        for row, score in e.ranked:
            if row.number == number:
                continue
            distance = e.distance(source.row, row)
            if not isfinite(distance):
                continue
            changed, unknown = [], []
            for c in e.reference.role('x'):
                a, b = source.row.cells[c.index], row.cells[c.index]
                if a is None or b is None:
                    unknown.append(c.name)
                elif a != b:
                    changed.append(CharacteristicChange(c.name, a, b))
            after = target_status(row, e.reference.columns, targets) if targets is not None else None
            yield e.compare(number, row.number), tuple(changed), tuple(unknown), distance, before, after

    def better(self, number, limit=5):
        """Strictly better d2h, ordered by raw X distance, d2h, then ID."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError('Limit must be a positive integer')
        results = []
        for pair, changed, unknown, distance, before, after in self._candidates(number):
            if pair.b.score < pair.a.score:
                key = (distance, pair.b.score, pair.b.row.number)
                results.append(Alternative(pair, changed, unknown, distance, before, after, key,
                                           'X distance, d2h, original row ID'))
        return tuple(sorted(results, key=lambda r: r.ranking_key)[:limit])

    def meet_more(self, number, targets):
        """Fewest known X changes first; unknown X pairs excluded. None=no solution.

        More satisfied targets need not imply lower d2h or Pareto dominance.
        """
        results = []
        for pair, changed, unknown, distance, before, after in self._candidates(number, targets):
            if not unknown and after.count > before.count:
                key = (len(changed), distance, -after.count, pair.b.score, pair.b.row.number)
                results.append(Alternative(pair, changed, unknown, distance, before, after, key,
                                           'changed X count, X distance, descending target count, d2h, original row ID'))
        return min(results, key=lambda r: r.ranking_key) if results else None
