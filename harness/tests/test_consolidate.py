"""Tests for harness/consolidate.py (phase 6A). Offline: no LLM call, no generated program, no T7 adapter."""

import hashlib
from pathlib import Path
import re
import tempfile
import unittest

from harness import analyze as AN
from harness import consolidate as C
from harness.consolidate_report import report


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class TestSameDiagnostics(unittest.TestCase):
    def test_ks_cannot_reject_at_three_or_fewer(self):
        for n, m in ((3, 3), (2, 3), (1, 3), (1, 1)):
            d = C.same_diagnostics(list(range(n)), [100 + i for i in range(m)])
            self.assertGreater(d["ks_critical"], 1.0, (n, m))
            self.assertFalse(d["ks_can_reject"])
        self.assertLess(C.same_diagnostics(list(range(9)), list(range(9)))["ks_critical"], 1.0)

    def test_singleton_lower_median_has_zero_eps_and_cliff_decides(self):
        d = C.same_diagnostics([10], [20, 21, 22])
        self.assertEqual((d["eps"], d["abs_cliffs"], d["same"]), (0.0, 1.0, False))
        self.assertTrue(d["route"].startswith("Cliff"))

    def test_matches_frozen_same(self):
        cases = [([1284, 1759, 1202], [1626, 1256, 1343]), ([1649, 1448], [1700, 1762, 1536]), ([5, 6, 7], [5, 6, 7]),
                 (list(range(20)), list(range(10, 30)))]
        for xs, ys in cases:
            self.assertEqual(C.same_diagnostics(xs, ys)["same"], AN.same(xs, ys))

    def test_identical_samples_are_same_by_median(self):
        d = C.same_diagnostics([1, 2, 3], [1, 2, 3])
        self.assertEqual((d["same"], d["route"]), (True, "medians within eps -> same"))


class TestConsolidatedRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen = {p: sha(p) for p in sorted(C.FROZEN.rglob("*")) if p.is_file()}
        cls.ext_before = {p: sha(p) for p in sorted(C.EXT.glob("*.csv"))}
        cls.res = C.build()
        cls.again = C.build()

    def test_design_denominators(self):
        internal = self.res["internal"]
        self.assertEqual(len(internal), 18)
        self.assertEqual(sum(r["steps"] for r in internal), 540)
        self.assertEqual(sum(r["requests"] for r in internal), 1080)
        for r in internal:
            self.assertEqual((r["trials_attempted"], r["steps"], r["requests"]), (3, 30, 60))
            self.assertLessEqual(r["steps_executable"], r["steps_generated"])
            self.assertLessEqual(r["steps_semantic_pass"], r["steps_executable"])

    def test_loc_samples_match_analyze(self):
        for r in self.res["internal"]:
            totals = AN.trial_totals(self.res["runs"], r["domain"], r["treatment"])
            self.assertEqual(r["complete_loc_trajectories"], len(totals))
            self.assertEqual(r["loc_total_2_10_values"], ";".join(f"{v:g}" for v in totals))

    def test_verdicts_match_frozen_tables(self):
        text = (C.FROZEN / "gpt-oss-20b" / "same_tables.md").read_text()
        for p in [p for p in self.res["pairs"] if p["loc_sample"] == "trial_total"]:
            sec = text.split(f"## {p['domain']}")[1].split("### secondary")[0]
            line = next(ln for ln in sec.splitlines() if ln.startswith(f"| {p['reuse']} vs {p['base']} |"))
            self.assertIn(p["frozen_verdict"], line, p)
            self.assertFalse(p["certification_informative"])

    def test_phase_5a_findings_rechecked(self):
        tt = [p for p in self.res["pairs"] if p["loc_sample"] == "trial_total"]
        self.assertEqual(sum(p["reuse_loc_lower"] for p in tt), 8)
        self.assertEqual(sum(p["reuse_semantic_lower"] for p in tt), 8)
        self.assertEqual(sum(r["loc_trend_spearman"] > 0 for r in self.res["internal"]), 17)
        self.assertEqual(self.res["dup"], (17, 10))

    def test_best_ranks_match_frozen(self):
        text = (C.FROZEN / "gpt-oss-20b" / "same_tables.md").read_text()
        for b in self.res["best_ranks"]:
            line = next(ln for ln in text.splitlines() if ln.startswith(f"| {b['domain']} | T"))
            cells = [c.strip() for c in line.strip("|").split("|")]
            self.assertEqual(set(cells[1].split(", ")), set(b["loc_best"]))
            self.assertEqual(set(cells[3].split(", ")), set(b["semantic_best"]))

    def test_external_denominators_and_exclusions(self):
        ext = self.res["external"]
        for r in ext:
            self.assertEqual(r["attempted"], 3)
            self.assertLessEqual(r["successful"], r["integrity_valid"])
            self.assertLessEqual(r["integrity_valid"], r["attempted"])
        self.assertFalse(any(r["skill"] == "Spot" for r in ext))
        self.assertFalse(any(r["skill"] == "Fix" and r["domain"] == "nasa" for r in ext))
        rem01 = [r for r in ext if r["request_id"].endswith("-01")]
        self.assertTrue(rem01 and not any(r["counts_as_external"] or r["in_delivery_total"] for r in rem01))
        racep = [r for r in ext if r["skill"] == "Race" and r["quality_label"].startswith("median MAE")]
        self.assertEqual(len(racep), 18)
        self.assertFalse(any(r["in_delivery_total"] for r in racep))
        for r in self.res["internal"]:
            sel = [x for x in ext if x["domain"] == r["domain"] and x["treatment"] == r["treatment"]
                   and x["in_delivery_total"]]
            self.assertEqual(r["external_attempts"], sum(x["attempted"] for x in sel))
            self.assertEqual(r["external_successes"], sum(x["successful"] for x in sel))

    def test_external_values_copied_not_recomputed(self):
        src = {(r["domain"], r["treatment"]): r for r in C.read(C.EXT / "prediction_summary.csv")
               if r["scope"] == "domain_treatment"}
        for r in [x for x in self.res["external"] if x["skill"] == "Guess"]:
            s = src[(r["domain"], r["treatment"])]
            self.assertEqual((r["successful"], r["integrity_valid"], r["quality_value"]),
                             (int(s["beats_baseline_n"]), int(s["primary_n"]), s["median_mae_ratio"]))

    def test_t7(self):
        t7 = {t["domain"]: t for t in self.res["t7"]}
        src = {r["domain"]: r for r in C.read(C.EXT / "t7_prediction.csv") if r["kind"] == "predict"}
        for d, t in t7.items():
            self.assertEqual((t["trials"], t["independent_trials"]), (1, 0))
            self.assertEqual(t["guess_mae_ratio"], float(src[d]["mae_ratio"]))
            self.assertTrue(t["race_prediction_identical"])
        self.assertAlmostEqual(t7["cars"]["guess_mae_ratio"], 0.605, places=3)
        self.assertAlmostEqual(t7["nasa"]["guess_mae_ratio"], 0.819, places=3)
        self.assertAlmostEqual(t7["apache"]["guess_mae_ratio"], 0.450, places=3)
        self.assertEqual((t7["nasa"]["loc_steps_2_10"], t7["apache"]["loc_steps_2_10"]), (1108, 1713))
        self.assertEqual((t7["cars"]["loc_steps_2_10"], t7["cars"]["library_baseline_loc"]), (585, 2255))
        self.assertIn("LLM-assisted", t7["cars"]["foundation"])
        self.assertEqual(t7["cars"]["adapter_files"], 3)

    def test_report_language_and_reproducibility(self):
        text = report(self.res)
        self.assertNotRegex(text.lower(), r"statistically significant|significantly (better|worse|lower|higher)")
        self.assertIn("no anomaly labels", text)
        self.assertIn("not independently human-written", text)
        self.assertEqual(report(self.again), text)
        self.assertEqual(C.to_csv(self.res["internal"]), C.to_csv(self.again["internal"]))
        self.assertEqual(C.to_csv(self.res["external"]), C.to_csv(self.again["external"]))

    def test_write_guard_and_outputs(self):
        with self.assertRaises(ValueError):
            C.write(self.res, C.FROZEN / "x")
        with tempfile.TemporaryDirectory() as tmp:
            out, files = C.write(self.res, Path(tmp) / "c", report)
            self.assertEqual({p.name for p in out.iterdir()},
                             {"treatment_internal.csv", "treatment_external.csv", "reuse_pair_comparison.csv",
                              "t7_comparison.csv", "statistical_validity.md", "consolidated_report.md",
                              "manifest.json"})
            self.assertTrue(re.search(r"KS cannot reject", files["statistical_validity.md"]))

    def test_nothing_frozen_changed(self):
        self.assertEqual({p: sha(p) for p in self.frozen}, self.frozen)
        self.assertEqual({p: sha(p) for p in self.ext_before}, self.ext_before)


if __name__ == "__main__":
    unittest.main()
