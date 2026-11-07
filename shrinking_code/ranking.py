"""Equal-weight distance to heaven for historical observed outcomes."""

from math import sqrt
from .stats import summarize


def distance_to_heaven(row, goals, summaries):
    """Return None for any missing goal, never impute a favorable outcome.

    summaries maps goal indices to ranges from the entire evaluation table,
    including observed values in otherwise incomplete rows.
    """
    if not goals:
        raise ValueError('Ranking requires at least one numerical goal')
    if any(row.cells[c.index] is None for c in goals):
        return None
    return sqrt(sum((summaries[c.index].normalize(row.cells[c.index]) - c.ideal) ** 2
                    for c in goals) / len(goals))


def rank_rows(table):
    """Return (row, score) pairs, omitting incomplete rows; ties use row number."""
    goals = table.role('y')
    if not goals:
        raise ValueError('Ranking requires at least one numerical goal')
    summaries = {c.index: summarize(table, c) for c in goals}
    ranked = []
    for row in table.rows:
        score = distance_to_heaven(row, goals, summaries)
        if score is not None:
            ranked.append((row, score))
    return sorted(ranked, key=lambda pair: (pair[1], pair[0].number))
