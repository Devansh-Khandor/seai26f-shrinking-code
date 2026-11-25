from pathlib import Path
import tempfile
import unittest

from harness.results import FIELDS, append_row, completed_keys, load_rows


def row(skill, status):
    return {"model": "m", "domain": "cars", "treatment": "T1", "trial": 1,
            "skill_index": skill, "status": status}


class TestResults(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "sub" / "runs.csv"

    def tearDown(self):
        self._tmp.cleanup()

    def test_header_once(self):
        append_row(self.path, row(1, "ok"))
        append_row(self.path, row(2, "ok"))
        lines = self.path.read_text().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0], ",".join(FIELDS))
        self.assertEqual(load_rows(self.path)[1]["skill_index"], "2")

    def test_completed_excludes_backend_error(self):
        append_row(self.path, row(1, "ok"))
        append_row(self.path, row(2, "backend_error"))
        self.assertEqual(completed_keys(self.path), {("m", "cars", "T1", "1", "1")})

    def test_latest_wins(self):
        append_row(self.path, row(2, "backend_error"))
        append_row(self.path, row(2, "ok"))
        self.assertIn(("m", "cars", "T1", "1", "2"), completed_keys(self.path))

    def test_missing_file(self):
        self.assertEqual(load_rows(self.path), [])


if __name__ == "__main__":
    unittest.main()
