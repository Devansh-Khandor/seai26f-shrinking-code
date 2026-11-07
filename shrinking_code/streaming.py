"""Sequential inventory updates using existing batch summaries and d2h ranking."""

from dataclasses import dataclass
from .data import Column, Row, Table
from .ranking import rank_rows
from .stats import summarize


@dataclass(frozen=True)
class Arrival:
    row: Row
    previous_count: int
    rank: object
    score: object
    top_five: bool


class Inventory:
    """Retain accepted records; validate and stage each update before committing.

    Bounds include the current inventory and arriving row, never future rows.
    Incomplete-goal rows are accepted but unranked; observed values still update
    bounds. Exact summaries and all ranks are recomputed after every arrival.
    """

    def __init__(self, initial):
        if not isinstance(initial, Table):
            raise ValueError('Initial state must be a Table')
        columns = initial.columns
        if any(not isinstance(c, Column) or not isinstance(c.name, str) for c in columns):
            raise ValueError('Initial table requires valid columns')
        names = [c.name for c in columns]
        if (not names or len(set(names)) != len(names)
                or any(not n or n != n.strip() for n in names)
                or any(c.index != i for i, c in enumerate(columns))):
            raise ValueError('Initial table requires a valid ordered schema')
        if not initial.role('y'):
            raise ValueError('Inventory ranking requires numerical goals')
        rows = tuple(self._row(columns, r.cells, r.number) for r in initial.rows)
        if len({r.number for r in rows}) != len(rows):
            raise ValueError('Initial row identities must be unique')
        self.table = Table(columns, tuple(sorted(rows, key=lambda r: r.number)))
        self.summaries = tuple(summarize(self.table, c) for c in columns)
        self.shortlist = tuple(rank_rows(self.table)[:5])
        self.accepted_count = 0

    @staticmethod
    def _row(columns, cells, number):
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
            raise ValueError('Row identity must be a positive integer')
        if isinstance(cells, (str, bytes)):
            raise ValueError('Provide cells in schema order, not a CSV string')
        try:
            cells = tuple(cells)
        except TypeError as error:
            raise ValueError('Provide an iterable of cells') from error
        if any(isinstance(v, bool) or (v is not None and not isinstance(v, (str, int, float)))
               for v in cells):
            raise ValueError('Cells must be strings, numbers, or None')
        if len(cells) != len(columns):
            raise ValueError(f'Expected {len(columns)} cells')
        return Row(number, tuple(c.parse('?' if v is None else str(v))
                                 for c, v in zip(columns, cells)))

    def accept(self, cells, number=None):
        """Assess before state mutation, then accept; invalid arrivals change nothing.

        Explicit identities preserve source row numbers; default is max(ID)+1.
        rank is 1-based among complete-goal candidates, or None if incomplete.
        """
        ids = {r.number for r in self.table.rows}
        if number is None:
            number = max(ids, default=0) + 1
        row = self._row(self.table.columns, cells, number)
        if number in ids:
            raise ValueError('Row identity already exists')
        candidate = Table(self.table.columns, tuple(sorted(self.table.rows + (row,),
                                                          key=lambda r: r.number)))
        ranked = rank_rows(candidate)
        rank, score = next(((i, score) for i, (r, score) in enumerate(ranked, 1)
                            if r.number == number), (None, None))
        decision = Arrival(row, len(self.table.rows), rank, score,
                           rank is not None and rank <= 5)
        summaries = tuple(summarize(candidate, c) for c in candidate.columns)
        self.table, self.summaries, self.shortlist = candidate, summaries, tuple(ranked[:5])
        self.accepted_count += 1
        return decision
