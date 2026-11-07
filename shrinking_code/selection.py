"""Sequential greedy inspection with an explicit hidden-goal boundary."""

from dataclasses import dataclass
from .data import Row, Table
from .prediction import KNNRegressor
from .ranking import distance_to_heaven
from .stats import summarize


def check_budget(budget):
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError('Budget must be a nonnegative integer')


class HistoricalOracle:
    """Own hidden Y; reveal one record per paid, non-repeatable inspection.

    This is an API boundary for an offline simulation, not a Python security
    sandbox. The policy receives only visible and the inspect callback.
    """

    def __init__(self, hidden, budget):
        check_budget(budget)
        if len({r.number for r in hidden.rows}) != len(hidden.rows):
            raise ValueError('Candidate identities must be unique')
        self.__goals = {r.number: tuple(r.cells[c.index] for c in hidden.role('y'))
                        for r in hidden.rows}
        self.visible = Table(hidden.columns, tuple(
            Row(r.number, tuple(r.cells[c.index] if c.role == 'x' else None
                                for c in hidden.columns)) for r in hidden.rows))
        self.__budget = budget
        self.__inspected = []

    @property
    def inspected(self):
        return tuple(self.__inspected)

    def inspect(self, number):
        if number not in self.__goals:
            raise ValueError('Unknown candidate identity')
        if number in self.__inspected:
            raise ValueError('Candidate already inspected')
        if len(self.__inspected) >= self.__budget:
            raise ValueError('Inspection budget exhausted')
        self.__inspected.append(number)
        return self.__goals[number]


@dataclass(frozen=True)
class Inspection:
    row: Row
    predicted_goals: tuple
    predicted_score: float
    observed_score: object
    best_row: object
    best_score: object
    training_count: int


@dataclass(frozen=True)
class SelectionResult:
    inspections: tuple
    prediction_count: int

    def checkpoint(self, cost):
        """Return state after exactly cost inspections, without rerunning."""
        check_budget(cost)
        if cost > len(self.inspections):
            raise ValueError('Checkpoint exceeds completed inspections')
        return self.inspections[cost - 1] if cost else None


def select_candidates(training, visible, inspect, budget=40, k=5):
    """Greedy predicted d2h, then original ID; spend at most min(budget, pool).

    Fixed initial-training goal summaries preserve existing extrapolation and
    constant-midpoint semantics. X bounds refit on observations each round.
    Missing revealed goals cost one, train available targets, but cannot win.
    Unpredictable goals raise before that round's inspection (no fake scores).
    """
    check_budget(budget)
    if training.columns != visible.columns:
        raise ValueError('Training and candidates must share a schema')
    ids = [r.number for r in visible.rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Candidate identities must be unique')
    if set(ids) & {r.number for r in training.rows}:
        raise ValueError('Training and candidate identities overlap')
    for row in visible.rows:
        if len(row.cells) != len(visible.columns):
            raise ValueError('Candidate width does not match schema')
        if any(row.cells[c.index] is not None for c in visible.columns if c.role != 'x'):
            raise ValueError('Candidates must expose only X; mask goals and ignored fields')
    goals = training.role('y')
    summaries = {c.index: summarize(training, c) for c in goals}
    if not goals or any(not s.count for s in summaries.values()):
        raise ValueError('Need at least one observed training value for every goal')
    # Validate training/k even for zero budget or an empty pool.
    model = KNNRegressor.fit(training, k=k)
    remaining = {r.number: r for r in visible.rows}
    history, prediction_count = [], 0
    best_row = best_score = None

    def with_goals(row, values):
        cells = list(row.cells)
        for c, value in zip(goals, values):
            cells[c.index] = value
        return Row(row.number, tuple(cells))

    for _ in range(min(budget, len(remaining))):
        offers = []
        for row in remaining.values():
            values = tuple(model.predict(row, c.name).value for c in goals)
            prediction_count += len(goals)
            score = distance_to_heaven(with_goals(row, values), goals, summaries)
            offers.append((score, row.number, values))
        predicted_score, number, predicted_goals = min(offers)
        candidate = remaining[number]
        # Decision is complete before the only access to this candidate's Y.
        revealed = tuple(inspect(number))
        if len(revealed) != len(goals):
            raise ValueError('Oracle must return one value per goal')
        values = tuple(c.parse('?' if v is None else str(v))
                       for c, v in zip(goals, revealed))
        observed = with_goals(candidate, values)
        score = distance_to_heaven(observed, goals, summaries)
        if score is not None and (best_score is None or
                                  (score, number) < (best_score, best_row.number)):
            best_row, best_score = observed, score
        history.append(Inspection(observed, predicted_goals, predicted_score, score,
                                  best_row, best_score, len(training.rows)))
        del remaining[number]
        training = Table(training.columns, training.rows + (observed,))
        model = KNNRegressor.fit(training, k=k)
    return SelectionResult(tuple(history), prediction_count)
