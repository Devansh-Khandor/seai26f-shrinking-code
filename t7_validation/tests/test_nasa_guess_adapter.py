"""Tests for NASA Guess common-contract bridge and held-out leakage control."""
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "t7_validation/adapters/nasa/guess.py"
spec = importlib.util.spec_from_file_location("nasa_guess_adapter", SRC)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class TestNASAGuessAdapter(unittest.TestCase):
    def setUp(self):
        self.dataset = ROOT / "data/raw/nasa93dem.csv"

    def test_full_contract_heldout_rows_and_ranges(self):
        effort, defects = adapter.answers(self.dataset)
        heldout = {str(i) for i in range(5, 94, 5)}
        self.assertEqual(set(effort["predictions"]), heldout)
        self.assertEqual(set(defects["predictions"]), heldout)
        self.assertGreater(effort["mae"], 0)
        for p in defects["predictions"].values():
            self.assertLessEqual(p["low"], p["estimate"])
            self.assertLessEqual(p["estimate"], p["high"])

    def test_masked_effort_and_defects_cannot_change_predictions(self):
        orig_effort, orig_defects = adapter.answers(self.dataset)
        with self.dataset.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        for target in ("Effort-", "Defects-"):
            altered = [r[:] for r in rows]
            col = altered[0].index(target)
            for rid in range(5, 94, 5):
                altered[rid][col] = str(float(altered[rid][col]) + 1000000)
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "masked.csv"
                with path.open("w", newline="", encoding="utf-8") as stream:
                    csv.writer(stream).writerows(altered)
                new_effort, new_defects = adapter.answers(path)
            self.assertEqual(new_effort["predictions"], orig_effort["predictions"])
            self.assertEqual(new_defects["predictions"], orig_defects["predictions"])

    def test_shuffled_rows_rebuild_train_and_test(self):
        orig_effort, _ = adapter.answers(self.dataset)
        with self.dataset.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "shuffled.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerows([rows[0]] + list(reversed(rows[1:])))
            new_effort, new_defects = adapter.answers(path)
        self.assertEqual(len(new_effort["predictions"]), 18)
        self.assertEqual(len(new_defects["predictions"]), 18)
        self.assertNotEqual(orig_effort["predictions"], new_effort["predictions"])


if __name__ == "__main__":
    unittest.main()
