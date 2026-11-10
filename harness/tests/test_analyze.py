from pathlib import Path
import shutil
import tempfile
import unittest

from harness.analyze import (for_model, incomplete_trials, pass_rates, pooled, same, sizing, trial_totals, verdict,
                             write_reports, histogram)

HAS_LUA = bool(shutil.which("lua") or shutil.which("luajit"))


def r(treatment, trial, skill, new_loc, status="ok", runs="True", model="m", pct="10.0", est="100"):
    return {"model": model, "domain": "cars", "treatment": treatment, "trial": str(trial),
            "skill_index": str(skill), "status": status, "runs": runs,
            "contract": runs, "new_loc": new_loc, "pct_new": pct, "prompt_tokens": "",
            "prompt_tokens_est": est}


class TestAnalyze(unittest.TestCase):
    @unittest.skipUnless(HAS_LUA, "Lua not installed")
    def test_same_identical(self):
        self.assertTrue(same([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]))

    @unittest.skipUnless(HAS_LUA, "Lua not installed")
    def test_same_different(self):
        self.assertFalse(same([1, 2, 3, 4, 5], [50, 60, 70, 80, 90]))

    def test_same_empty(self):
        with self.assertRaises(ValueError):
            same([], [1.0])

    def ladder(self, t, trial, loc="1", blank_at=None):
        return [r(t, trial, s, "" if s == blank_at else loc,
                  "context_overflow" if s == blank_at else "ok") for s in range(1, 11)]

    def test_trial_totals_complete_only(self):
        rows = self.ladder("T1", 1, "2") + self.ladder("T1", 2, "2", blank_at=5) + self.ladder("T1", 3, "3")[:6]
        self.assertEqual(trial_totals(rows, "cars", "T1"), [18.0])
        self.assertEqual(incomplete_trials(rows, "cars", "T1"), 2)

    def test_trial_totals_sorted_by_trial(self):
        rows = self.ladder("T1", 10, "1") + self.ladder("T1", 2, "2")
        self.assertEqual(trial_totals(rows, "cars", "T1"), [18.0, 9.0])

    def test_all_failed_arm_not_certified_lower(self):
        rows = [r("T2", k, s, "", "parse_failed", "False") for k in (1, 2, 3) for s in range(1, 11)]
        rows += [x for k in (1, 2, 3) for x in self.ladder("T1", k, "40")]
        self.assertEqual(trial_totals(rows, "cars", "T2"), [])

    def test_pass_rates_ignore_unresolved_backend_errors(self):
        rows = [r("T1", 1, 1, "1"), r("T1", 1, 2, "", "backend_error", "False")]
        self.assertEqual(pass_rates(rows, "cars", "T1")["n"], 1)

    def test_pooled_skips_blank(self):
        rows = [r("T1", 1, 1, "99"), r("T1", 1, 2, "10"), r("T1", 1, 3, ""), r("T1", 1, 4, "5")]
        self.assertEqual(pooled(rows, "cars", "T1"), [10.0, 5.0])

    def test_for_model_excludes_human(self):
        rows = [r("T1", 1, 1, "1"), r("T7", 1, 1, "1", model="human")]
        self.assertEqual([x["model"] for x in for_model(rows, "m")], ["m"])

    def test_pass_rates(self):
        rows = [r("T1", 1, i, "1") for i in (1, 2, 3)] + [r("T1", 1, 4, "", "context_overflow", "False")]
        p = pass_rates(rows, "cars", "T1")
        self.assertEqual((p["n"], p["runs_rate"], p["overflow"]), (4, 0.75, 1))

    def test_histogram_median(self):
        rows = [r("T1", t, 2, "1", pct=str(p)) for t, p in ((1, 10), (2, 30), (3, 20))]
        self.assertEqual(histogram(rows)[("cars", "T1", 2)], 20.0)

    def test_sizing_falls_back_to_estimate(self):
        rows = [r("T5", 1, 4, "1", est="9000"), r("T5", 2, 4, "1", est="9500")]
        self.assertEqual(sizing(rows)[("cars", "T5", 4)], 9500)

    @unittest.skipUnless(HAS_LUA, "Lua not installed")
    def test_verdict_lower(self):
        self.assertEqual(verdict([1, 2, 3, 2, 1], [50, 60, 70, 80, 90]), "reuse lower (certified)")
        self.assertEqual(verdict([1, 2, 3], [1, 2, 3]), "no certified difference")

    @unittest.skipUnless(HAS_LUA, "Lua not installed")
    def test_write_reports(self):
        rows = [r(t, k, s, str(k * s if t == "T1" else s)) for t in ("T1", "T2")
                for k in (1, 2, 3) for s in range(1, 11)]
        with tempfile.TemporaryDirectory() as d:
            paths = write_reports(rows, "m", Path(d))
            self.assertEqual(sorted(p.name for p in paths),
                             ["histogram.csv", "histogram.md", "pass_rates.md", "same_tables.md", "sizing.md"])
            text = (Path(d) / "same_tables.md").read_text()
            self.assertIn("T2 vs T1", text)
            self.assertIn("little power", text)
            self.assertIn("secondary", text)
            self.assertIn("excluded", text)


if __name__ == "__main__":
    unittest.main()
