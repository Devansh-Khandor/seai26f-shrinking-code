"""Descriptive X-only diagnosis of fixed outcome groups, never causal inference."""

from dataclasses import dataclass
from math import isfinite
from .data import Column, Table
from .stats import NumericSummary, summarize
from .ranking import rank_rows
from .certification import compare


@dataclass(frozen=True)
class Groups:
    poor: Table
    favorable: Table
    poor_outcome: NumericSummary
    favorable_outcome: NumericSummary
    eligible_count: int


def select_quartiles(table, scores, higher_better=True):
    """Use first/last floor(n/4) of (observed score, original ID) ascending.

    Missing scores are omitted. Upper-bound ties select higher row IDs. The
    middle records are unused, groups are disjoint, and n<4 is insufficient.
    """
    if len({r.number for r in table.rows}) != len(table.rows):
        raise ValueError('Row identities must be unique')
    pairs = [(r, scores.get(r.number)) for r in table.rows if scores.get(r.number) is not None]
    if any(not isfinite(score) for _, score in pairs):
        raise ValueError('Outcome scores must be finite')
    pairs.sort(key=lambda pair: (pair[1], pair[0].number))
    q = len(pairs) // 4
    if q == 0:
        raise ValueError('Quartiles require at least four observed outcomes')
    low, high = pairs[:q], pairs[-q:]
    poor, favorable = (low, high) if higher_better else (high, low)
    return Groups(Table(table.columns, tuple(r for r, _ in poor)),
                  Table(table.columns, tuple(r for r, _ in favorable)),
                  NumericSummary.from_values(v for _, v in poor),
                  NumericSummary.from_values(v for _, v in favorable), len(pairs))


def outcome_groups(table, outcome=None, higher_better=True):
    """Select by a numerical column, or full-table Step 1 d2h when outcome=None."""
    if outcome is None:
        return select_quartiles(table, {r.number: s for r, s in rank_rows(table)}, False)
    column = next((c for c in table.columns if c.name == outcome and c.numeric), None)
    if column is None:
        raise ValueError('Outcome must name a numerical column')
    return select_quartiles(table, {r.number: r.cells[column.index] for r in table.rows}, higher_better)


@dataclass(frozen=True)
class CategoryEvidence:
    value: str
    poor_count: int
    favorable_count: int
    poor_proportion: object
    favorable_proportion: object
    difference: object


@dataclass(frozen=True)
class Finding:
    column: Column
    poor: object
    favorable: object
    poor_total: int
    favorable_total: int
    separation: object
    direction: str
    certification: object
    categories: tuple
    limitations: tuple


def compare_groups(poor, favorable):
    """Rank within type: numeric absolute Cliff's delta; categorical TV distance.

    Scores require >=2 observed values in each group. Missing X values are
    omitted per feature; insufficient findings sort last, ties by column order.
    """
    if poor.columns != favorable.columns:
        raise ValueError('Group schemas must match')
    ids = [r.number for r in poor.rows + favorable.rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Groups must have unique, disjoint row identities')
    findings = []
    for c in poor.role('x'):
        a, b = summarize(poor, c), summarize(favorable, c)
        enough = min(a.count, b.count) >= 2
        notes = []
        if a.count < len(poor.rows) or b.count < len(favorable.rows):
            notes.append('Missing X values omitted; observed denominators differ from group sizes')
        if not enough:
            notes.append('Insufficient evidence: need two observed values in each group')
        certification, categories, score = None, (), None
        if c.numeric:
            certification = compare([r.cells[c.index] for r in poor.rows],
                                    [r.cells[c.index] for r in favorable.rows])
            score = certification.cliffs
            direction = ('unknown' if not a.count or not b.count else
                         'higher mean in poor' if a.mean > b.mean else
                         'lower mean in poor' if a.mean < b.mean else 'equal means')
            if a.count and b.count and a.minimum == a.maximum == b.minimum == b.maximum:
                notes.append('Constant across both groups')
        else:
            evidence = []
            for value in sorted(set(a.frequencies) | set(b.frequencies)):
                na, nb = a.frequencies.get(value, 0), b.frequencies.get(value, 0)
                pa, pb = na / a.count if a.count else None, nb / b.count if b.count else None
                evidence.append(CategoryEvidence(value, na, nb, pa, pb,
                                                 pa - pb if pa is not None and pb is not None else None))
            categories = tuple(evidence)
            score = sum(abs(e.difference) for e in categories) / 2 if enough else None
            direction = 'positive category differences indicate overrepresentation in poor'
            if any(0 < n < 5 for e in categories for n in (e.poor_count, e.favorable_count)):
                notes.append('Some nonzero category counts are below five; proportions may be unstable')
            if len(categories) == 1:
                notes.append('Only one observed category')
        findings.append(Finding(c, a, b, len(poor.rows), len(favorable.rows), score,
                                direction, certification, categories, tuple(notes)))
    return tuple(sorted(findings, key=lambda f: (not f.column.numeric, f.separation is None,
                                               -(f.separation or 0), f.column.index)))
