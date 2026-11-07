"""Explain historical scores and tradeoffs against one explicit reference table."""

from dataclasses import dataclass
from math import isfinite
from .data import Row
from .distance import XDistance
from .ranking import rank_rows
from .stats import summarize


@dataclass(frozen=True)
class GoalContribution:
    name: str
    raw: object
    direction: str
    minimum: object
    maximum: object
    normalized: object
    ideal: int
    squared: object


@dataclass(frozen=True)
class ScoreExplanation:
    row: Row
    rank: object
    score: object
    goals: tuple


@dataclass(frozen=True)
class PairExplanation:
    a: ScoreExplanation
    b: ScoreExplanation
    favors_a: tuple
    favors_b: tuple
    equal: tuple
    unknown: tuple
    winner: object


@dataclass(frozen=True)
class Neighbor:
    explanation: ScoreExplanation
    distance: float


class Explainer:
    """Historical rows only; no held-out prediction explanation or unseen labels.

    Ranking and X-distance both use the supplied reference table. Missing goals
    remain unranked, though known goal contributions can still be displayed.
    """

    def __init__(self, reference):
        self.reference = reference
        self.rows = {r.number: r for r in reference.rows}
        if len(self.rows) != len(reference.rows):
            raise ValueError('Reference row identities must be unique')
        self.ranked = tuple(rank_rows(reference))
        self.ranks = {r.number: (i, score) for i, (r, score) in enumerate(self.ranked, 1)}
        self.summaries = {c.index: summarize(reference, c) for c in reference.role('y')}
        self.distance = XDistance.fit(reference)

    def explain(self, number):
        if number not in self.rows:
            raise ValueError('Row must belong to the historical reference table')
        row = self.rows[number]
        goals = []
        for c in self.reference.role('y'):
            summary, raw = self.summaries[c.index], row.cells[c.index]
            normalized = None if raw is None else summary.normalize(raw)
            squared = None if normalized is None else (normalized - c.ideal) ** 2
            goals.append(GoalContribution(c.name, raw, 'minimize' if c.ideal == 0 else 'maximize',
                                          summary.minimum, summary.maximum, normalized, c.ideal, squared))
        rank, score = self.ranks.get(number, (None, None))
        return ScoreExplanation(row, rank, score, tuple(goals))

    def neighbors(self, number, k=3):
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError('k must be a positive integer')
        selected = self.explain(number).row
        candidates = []
        for row in self.reference.rows:
            if row.number != number:
                distance = self.distance(selected, row)
                if isfinite(distance):
                    candidates.append((round(distance, 12), row.number, distance))
        return tuple(Neighbor(self.explain(n), distance)
                     for _, n, distance in sorted(candidates)[:k])

    def compare(self, a, b):
        if a == b:
            raise ValueError('Choose two distinct historical records')
        left, right = self.explain(a), self.explain(b)
        favors_a, favors_b, equal, unknown = [], [], [], []
        for x, y in zip(left.goals, right.goals):
            if x.raw is None or y.raw is None:
                unknown.append(x.name)
            elif x.raw == y.raw:
                equal.append(x.name)
            elif (x.raw > y.raw) == (x.ideal == 1):
                favors_a.append(x.name)
            else:
                favors_b.append(x.name)
        winner = None
        if left.score is not None and right.score is not None and left.score != right.score:
            winner = a if left.score < right.score else b
        return PairExplanation(left, right, tuple(favors_a), tuple(favors_b), tuple(equal),
                               tuple(unknown), winner)

    def tradeoff_pair(self):
        """First pair in nested ranking order with strictly better A and wins each way.

        Equal-score pairs and incomplete rows cannot qualify. None means no pair.
        """
        for i, (a, score_a) in enumerate(self.ranked):
            for b, score_b in self.ranked[i + 1:]:
                if score_a < score_b:
                    pair = self.compare(a.number, b.number)
                    if pair.favors_a and pair.favors_b:
                        return pair
        return None
