"""Mixed-type X distance fitted only on a training table."""

from dataclasses import dataclass
from math import inf, isfinite
from .stats import summarize


@dataclass(frozen=True)
class XDistance:
    columns: tuple
    summaries: tuple
    p: float = 2

    @classmethod
    def fit(cls, training, p=2):
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not isfinite(p) or p < 1:
            raise ValueError('p must be a finite number >= 1')
        columns = training.role('x')
        return cls(columns, tuple(summarize(training, c) if c.numeric else None
                                  for c in columns), p)

    def __call__(self, left, right):
        """Mean Minkowski distance; skip unavailable inputs, return inf if none.

        Constant numeric inputs contribute zero using the existing normalization.
        Out-of-training-range values are not clipped and can yield distances > 1.
        """
        differences = []
        for c, summary in zip(self.columns, self.summaries):
            a, b = left.cells[c.index], right.cells[c.index]
            if a is None or b is None or (c.numeric and not summary.count):
                continue
            difference = (abs(summary.normalize(a) - summary.normalize(b))
                          if c.numeric else float(a != b))
            differences.append(difference ** self.p)
        return (sum(differences) / len(differences)) ** (1 / self.p) if differences else inf
