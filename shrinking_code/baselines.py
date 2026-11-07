"""Customer-facing baseline comparisons, not software-development treatments."""

from dataclasses import dataclass
from random import Random
from .data import Row
from .selection import HistoricalOracle, select_candidates, check_budget
from .ranking import distance_to_heaven
from .stats import NumericSummary, summarize
from .prediction import KNNRegressor
from .evaluation import evaluate, train_test_split
from .certification import compare


@dataclass(frozen=True)
class RandomInspection:
    row: Row
    observed_score: object
    best_row: object
    best_score: object


@dataclass(frozen=True)
class RandomTrial:
    seed: int
    inspections: tuple
    method: str = 'random'
    prediction_count: int = 0


@dataclass(frozen=True)
class BudgetComparison:
    budget: int
    model_score: object
    model_predictions: int
    random_summary: NumericSummary
    match_or_beat_fraction: object
    valid_trials: int
    total_trials: int


@dataclass(frozen=True)
class SelectionRace:
    training_ids: tuple
    candidate_ids: tuple
    model: object
    random_trials: tuple
    checkpoints: tuple


def random_selection(training, visible, inspect, budget, seed):
    """Shuffle sorted visible IDs with local random.Random(seed); never peek at Y."""
    check_budget(budget)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('Seed must be an integer')
    if training.columns != visible.columns:
        raise ValueError('Schemas must match')
    ids = [r.number for r in visible.rows]
    train_ids = [r.number for r in training.rows]
    if len(set(ids)) != len(ids) or len(set(train_ids)) != len(train_ids):
        raise ValueError('Row identities must be unique')
    if set(ids) & set(train_ids):
        raise ValueError('Training/candidate identities overlap')
    for row in visible.rows:
        if len(row.cells) != len(visible.columns) or any(
                row.cells[c.index] is not None for c in visible.columns if c.role != 'x'):
            raise ValueError('Candidates must expose only X in the shared schema')
    goals = training.role('y')
    summaries = {c.index: summarize(training, c) for c in goals}
    if not goals or any(not s.count for s in summaries.values()):
        raise ValueError('Need observed training bounds for every goal')
    order = sorted(visible.rows, key=lambda r: r.number)
    Random(seed).shuffle(order)
    history = []
    best_row = best_score = None
    for row in order[:budget]:
        values = tuple(inspect(row.number))
        if len(values) != len(goals):
            raise ValueError('Oracle must return one value per goal')
        cells = list(row.cells)
        for c, value in zip(goals, values):
            cells[c.index] = c.parse('?' if value is None else str(value))
        observed = Row(row.number, tuple(cells))
        score = distance_to_heaven(observed, goals, summaries)
        if score is not None and (best_score is None or
                                  (score, row.number) < (best_score, best_row.number)):
            best_row, best_score = observed, score
        history.append(RandomInspection(observed, score, best_row, best_score))
    return RandomTrial(seed, tuple(history))


def summarize_budget(model_score, scores):
    """Exclude undefined best scores, disclose valid n; no fabricated wins."""
    scores = tuple(scores)
    summary = NumericSummary.from_values(scores)
    fraction = (sum(s <= model_score for s in scores if s is not None) / summary.count
                if summary.count and model_score is not None else None)
    return summary, fraction


def selection_race(table, budgets=(10, 20, 40), seeds=tuple(range(30))):
    budgets, seeds = tuple(budgets), tuple(seeds)
    if not budgets or len(set(budgets)) != len(budgets):
        raise ValueError('Budgets must be nonempty and unique')
    for budget in budgets:
        check_budget(budget)
        if budget == 0:
            raise ValueError('Race checkpoints require positive budgets')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be nonempty and unique')
    training, hidden = train_test_split(table)
    maximum = max(budgets)
    if maximum > len(hidden.rows):
        raise ValueError('Race budget exceeds candidate pool')
    oracle = HistoricalOracle(hidden, maximum)
    model = select_candidates(training, oracle.visible, oracle.inspect, maximum)
    trials = []
    for seed in seeds:
        oracle = HistoricalOracle(hidden, maximum)
        trials.append(random_selection(training, oracle.visible, oracle.inspect, maximum, seed))
    checkpoints = []
    for budget in sorted(budgets):
        score = model.checkpoint(budget).best_score
        summary, fraction = summarize_budget(score, (r.inspections[budget-1].best_score for r in trials))
        # Greedy predicts each goal for every remaining candidate each round.
        predictions = len(training.role('y')) * sum(len(hidden.rows)-i for i in range(budget))
        checkpoints.append(BudgetComparison(budget, score, predictions, summary, fraction,
                                            summary.count, len(trials)))
    return SelectionRace(tuple(r.number for r in training.rows), tuple(r.number for r in hidden.rows),
                         model, tuple(trials), tuple(checkpoints))


@dataclass(frozen=True)
class PairedError:
    number: int
    knn_error: float
    mean_error: float
    difference: float  # kNN minus mean; negative favors kNN.


@dataclass(frozen=True)
class PredictionRace:
    target: str
    evaluation: object
    pairs: tuple
    improvement: float  # baseline MAE minus kNN MAE; positive favors kNN.
    percentage_improvement: object
    wins: int
    losses: int
    ties: int
    comparison: object


def prediction_race(training, test, target):
    model = KNNRegressor.fit(training)
    evaluation = evaluate(model, test, target)
    column = model.target_column(target)
    pairs = []
    for row, prediction in evaluation.predictions:
        knn = abs(row.cells[column.index] - prediction.value)
        baseline = abs(row.cells[column.index] - evaluation.baseline)
        pairs.append(PairedError(row.number, knn, baseline, knn-baseline))
    improvement = evaluation.baseline_mae - evaluation.mae
    percentage = 100 * improvement / evaluation.baseline_mae if evaluation.baseline_mae else None
    return PredictionRace(target, evaluation, tuple(pairs), improvement, percentage,
                          sum(p.difference < 0 for p in pairs), sum(p.difference > 0 for p in pairs),
                          sum(p.difference == 0 for p in pairs),
                          compare([p.knn_error for p in pairs], [p.mean_error for p in pairs]))
