"""Batch summaries; missing values never contribute to statistics."""

from collections import Counter
from dataclasses import dataclass
from statistics import mean, median


@dataclass(frozen=True)
class NumericSummary:
    count: int
    minimum: object
    maximum: object
    mean: object
    median: object

    @classmethod
    def from_values(cls, values):
        valid = [v for v in values if v is not None]
        return (cls(len(valid), min(valid), max(valid), mean(valid), median(valid))
                if valid else cls(0, None, None, None, None))

    def normalize(self, value):
        """Map an observed value to [0,1]; constants use neutral midpoint 0.5."""
        if value is None or not self.count:
            raise ValueError('Normalization requires a known value and observed range')
        if self.maximum == self.minimum:
            return 0.5
        return (value - self.minimum) / (self.maximum - self.minimum)


@dataclass(frozen=True)
class SymbolicSummary:
    count: int
    frequencies: dict
    mode: object

    @classmethod
    def from_values(cls, values):
        counts = Counter(v for v in values if v is not None)
        # Counter preserves encounter order, resolving mode ties deterministically.
        mode = counts.most_common(1)[0][0] if counts else None
        return cls(sum(counts.values()), dict(counts), mode)


def summarize(table, column):
    kind = NumericSummary if column.numeric else SymbolicSummary
    return kind.from_values(row.cells[column.index] for row in table.rows)
