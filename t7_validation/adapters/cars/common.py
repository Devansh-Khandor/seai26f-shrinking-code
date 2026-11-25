"""Shared orchestration for the Cars T7 prediction adapters (no modelling logic).

Cars T7 (`human_t7/cars/`) calls the shared `shrinking_code/` library, which was
developed with LLM assistance (see the repository README, "Disclosure"). These
adapters only re-split the data to the common protocol and print the common
ANSWER format; every prediction comes from that library, unchanged.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shrinking_code.data import Row, Table  # noqa: E402

TARGET = "Mpg+"
K = 5            # as in human_t7/cars/guess.py: KNNRegressor.fit(training, k=5)


def load(csv_path):
    """The CSV the common evaluator passes (real data or a controlled copy)."""
    return Table.load(csv_path)


def common_split(table):
    """The common protocol: rows whose 1-based number is divisible by 5 are held out.

    This replaces T7's original seeded 80/20 split (train_test_split, seed 42)."""
    training = Table(table.columns, tuple(r for r in table.rows if r.number % 5))
    heldout = Table(table.columns, tuple(r for r in table.rows if r.number % 5 == 0))
    if not training.rows or not heldout.rows:
        raise ValueError("Common evaluation requires training and held-out rows")
    return training, heldout


def masked(row, columns):
    """The row as the model may see it: every goal cell hidden."""
    return Row(row.number, tuple(None if c.role == "y" else v for c, v in zip(columns, row.cells)))
