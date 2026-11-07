"""MOOT tables with ordered columns and original data-row numbers."""

import csv
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Column:
    index: int
    name: str

    @property
    def role(self):
        return 'ignored' if self.name.endswith('X') else (
            'y' if self.name.endswith(('+', '-')) else 'x')

    @property
    def numeric(self):
        return self.name[0].isupper() or self.role == 'y'

    @property
    def ideal(self):
        return 0 if self.name.endswith('-') else 1

    def parse(self, text):
        text = text.strip()
        if text == '?':
            return None
        if not text:
            raise ValueError(f'{self.name}: empty cell; use ? for missing values')
        if not self.numeric:
            return text
        value = float(text)
        if not isfinite(value):
            raise ValueError(f'{self.name}: expected a finite number')
        return value


@dataclass(frozen=True)
class Row:
    number: int
    cells: tuple


@dataclass(frozen=True)
class Table:
    columns: tuple
    rows: tuple

    def role(self, name):
        return tuple(c for c in self.columns if c.role == name)

    @classmethod
    def from_csv(cls, source):
        """Read a text stream; retain ignored fields but exclude them from X/Y."""
        reader = csv.reader(source)
        names = next(reader, [])
        if not names or any(not n or n != n.strip() for n in names):
            raise ValueError('Expected nonempty column names without outer whitespace')
        if len(set(names)) != len(names):
            raise ValueError('Duplicate column names')
        columns = tuple(Column(i, name) for i, name in enumerate(names))
        rows = []
        for number, cells in enumerate(reader, 1):
            if len(cells) != len(columns):
                raise ValueError(f'Data row {number}: expected {len(columns)} cells')
            try:
                values = tuple(c.parse(v) for c, v in zip(columns, cells))
            except ValueError as error:
                raise ValueError(f'Data row {number}: {error}') from error
            rows.append(Row(number, values))
        return cls(columns, tuple(rows))

    @classmethod
    def load(cls, path):
        with open(path, newline='', encoding='utf-8-sig') as source:
            return cls.from_csv(source)
