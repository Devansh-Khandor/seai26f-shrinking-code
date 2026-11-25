"""Tests for harness/external.py (phase 1: prediction MAE). No LLM calls, no generated code is run."""

import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from harness import answers as A
from harness import external as E
from harness import oracle as O

ROOT = "fake-v2"
MODEL = E.MODEL
CARS = O.load(E.DATA_DIR / "auto93.csv")
RID, TARGET = "CAR-03", "Mpg+"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preds(shift=0.0, data=CARS, target=TARGET, drop=()):
    return {str(r): data.value(r, target) + shift for r in O.held_out(data.n) if r not in drop}


def answer(p, mae=0.0):
    return f"ANSWER {RID} " + json.dumps({"predictions": p, "mae": mae})


def check_txt(stdout, runs=True, timed_out=False):
    return (f"runs={runs} contract=True timed_out={timed_out} output_capped=False\n"
            "files written at runtime (discarded): none\n--- stdout ---\n"
            f"{stdout}\n--- stderr ---\n\n")


def semantic(status="pass", failed=(), mask="ran"):
    runs = {"base": "ran", "shuffled": "ran"}
    if mask is not None:
        runs.update({"mask Lbs-": "ran", f"mask {TARGET}": mask})
    return json.dumps({"status": status, "runs": runs, "requests": [
        {"request_id": RID, "status": status, "passed": 0, "total": 0, "failed": list(failed), "notes": []},
        {"request_id": "CAR-04", "status": "pass", "passed": 0, "total": 0, "failed": [], "notes": []}]})


LEAK = "mask:held-out targets not used: changing only the held-out Mpg+ values moved the estimates"
MEAN = E.baseline(CARS, TARGET)[0]
SCENARIOS = {
    # (treatment, trial): (runs.csv status, check.txt text or None, semantic.json text or None)
    ("T1", 1): ("ok", check_txt(answer(preds(1.0), 1.0)), semantic()),
    ("T1", 2): ("parse_failed", None, None),
    ("T1", 3): ("ok", check_txt(answer(preds(1.0)), runs=False), semantic("execution_failure", mask=None)),
    ("T2", 1): ("ok", check_txt("== CAR-03 ==\nno answer line"), semantic("fail", ["base:answer present: no"], None)),
    ("T2", 2): ("ok", check_txt(f"ANSWER {RID} {{predictions: 1}}"), semantic("fail", mask=None)),
    ("T2", 3): ("ok", check_txt(answer(preds(1.0, drop=(5,)))), semantic("fail", mask=None)),
    ("T3", 1): ("ok", check_txt(answer(preds(0.5))), semantic("fail", [LEAK])),
    ("T3", 2): ("ok", check_txt(answer(preds(0.5))), semantic("unverified", mask="timed out")),
    ("T3", 3): ("ok", check_txt("", runs=False, timed_out=True), semantic("execution_failure", mask=None)),
    ("T4", 1): ("ok", check_txt(answer({str(r): MEAN for r in O.held_out(CARS.n)})), semantic("fail")),
    ("T4", 2): ("ok", check_txt(answer(preds(9.0)) + "\n" + answer(preds(2.0), 2.0)), semantic()),
    ("T4", 3): ("ok", check_txt("--- stderr ---\n" + answer(preds(3.0))), semantic()),
    ("T5", 1): ("ok", check_txt(answer(preds(1.0))), semantic()),
    ("T5", 2): ("ok", check_txt(answer(preds(1.0))), semantic()),
    ("T5", 3): ("ok", check_txt(answer(preds(1.0))), semantic()),
    ("T6", 1): ("ok", check_txt(answer(preds(1.0))), semantic()),
    ("T6", 2): ("ok", check_txt(answer(preds(1.0))), semantic()),
    # ("T6", 3) has no runs.csv row: not attempted.
}


def build_fixture(tmp):
    tmp = Path(tmp)
    archive, runs_csv, req_csv = tmp / "fixture.zip", tmp / "runs.csv", tmp / "requirements.csv"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(f"{ROOT}/README.txt", "fixture")
        for (t, k), (status, check, sem) in SCENARIOS.items():
            if check is not None:
                z.writestr(E.entry_name(ROOT, MODEL, "cars", t, k, 2, "check.txt"), check)
            if sem is not None:
                z.writestr(E.entry_name(ROOT, MODEL, "cars", t, k, 2, "semantic.json"), sem)
        # Unrelated entries the extractor must ignore.
        z.writestr(E.entry_name(ROOT, MODEL, "cars", "T1", 1, 3, "check.txt"), check_txt(answer(preds(99.0))))
        z.writestr(f"{ROOT}/results/runs.csv", "x")
    with open(runs_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "domain", "treatment", "trial", "skill_index", "status", "runs", "semantic"])
        for (t, k), (status, check, sem) in SCENARIOS.items():
            runs = "True" if check and "runs=True" in check else "False"
            w.writerow([MODEL, "cars", t, k, 2, status, runs, json.loads(sem)["status"] if sem else "parse_failure"])
    with open(req_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "domain", "treatment", "trial", "request_id", "status"])
        for (t, k), (status, check, sem) in SCENARIOS.items():
            w.writerow([MODEL, "cars", t, k, RID, json.loads(sem)["status"] if sem else "parse_failure"])
    return archive, runs_csv, req_csv


class TestPaths(unittest.TestCase):
    def test_parse_entry(self):
        p = E.parse_entry("gpt-oss-20b-v2-3trials/runs/gpt-oss-20b/nasa/T4/trial3/step02.check.txt")
        self.assertEqual(p, {"root": "gpt-oss-20b-v2-3trials", "model": "gpt-oss-20b", "domain": "nasa",
                             "treatment": "T4", "trial": 3, "step": 2, "kind": "check.txt"})
        self.assertEqual(E.parse_entry("r/runs/m/apache/T6/trial1/step10.semantic.json")["kind"], "semantic.json")

    def test_parse_entry_rejects_other_files(self):
        for name in ["r/README.txt", "r/runs/m/cars/T1/trial1/step02.prompt.txt",
                     "r/runs/m/cars/T1/trial1/step02/skill_02.py", "r/runs/m/cars/T7/trial1/step02.check.txt",
                     "r/runs/m/cars/T1/trial1/step02.semantic.pre-regrade.json", "r/results/runs.csv"]:
            self.assertIsNone(E.parse_entry(name), name)

    def test_entry_name_round_trip(self):
        name = E.entry_name("r", "m", "apache", "T2", 1, 2, "semantic.json")
        self.assertEqual(name, "r/runs/m/apache/T2/trial1/step02.semantic.json")
        p = E.parse_entry(name)
        self.assertEqual((p["domain"], p["treatment"], p["trial"], p["step"]), ("apache", "T2", 1, 2))

    def test_request_ids_and_frozen_targets(self):
        self.assertEqual(E.request_ids("cars", 2), ("CAR-03", "CAR-04"))
        self.assertEqual(E.request_ids("nasa", 10), ("NASA-19", "NASA-20"))
        for d, (rid, target, direction) in E.PREDICTION.items():
            self.assertEqual(rid, E.request_ids(d, E.GUESS_STEP)[0])
            self.assertEqual(A.REQUESTS[rid], ("predict", {"target": target}))
            self.assertIn((target, direction), O.load(E.DATA_DIR / E.DATASETS[d]).goals())

    def test_real_archive_layout(self):
        with zipfile.ZipFile(E.ARCHIVE) as z:
            names = z.namelist()
        self.assertEqual(E.archive_root(names, MODEL), "gpt-oss-20b-v2-3trials")
        checks = [p for p in map(E.parse_entry, names) if p and p["kind"] == "check.txt" and p["step"] == 2]
        self.assertEqual(len({(p["domain"], p["treatment"], p["trial"]) for p in checks}), len(checks))
        self.assertLessEqual(len(checks), 54)


class TestParsing(unittest.TestCase):
    def test_split_check(self):
        flags, out, err = E.split_check(check_txt("a\n--- stderr ---\nb", runs=True) + "Traceback")
        self.assertEqual(flags, {"runs": True, "contract": True, "timed_out": False, "output_capped": False})
        self.assertEqual(out, "a\n--- stderr ---\nb")
        self.assertIn("Traceback", err)

    def test_split_check_without_markers(self):
        self.assertEqual(E.split_check("runs=False")[1:], ("", ""))

    def test_answer_lines(self):
        got = A.parse_answers(f"x\nANSWER {RID} {{\"mae\": 1}}\n  ANSWER {RID} {{\"mae\": 2}}  \nANSWER CAR-04 []")
        self.assertEqual(got[RID], {"mae": 2})                      # last line wins
        self.assertNotIn("CAR-04", got)                             # a non-object is not an ANSWER line
        self.assertTrue(A.parse_answers(f"ANSWER {RID} {{bad}}")[RID].startswith("invalid JSON"))
        self.assertNotIn(RID, A.parse_answers(f"answer {RID} {{}}"))


class TestMetric(unittest.TestCase):
    def test_held_out_alignment(self):
        for d, (rid, target, _) in E.PREDICTION.items():
            data = O.load(E.DATA_DIR / E.DATASETS[d])
            _, _, held = E.baseline(data, target)
            self.assertEqual(held, [r for r in range(1, data.n + 1) if r % 5 == 0])
            self.assertEqual(held, O.held_out(data.n))
        self.assertEqual(len(O.held_out(CARS.n)), 79)

    def test_training_only_baseline(self):
        for d, (rid, target, _) in E.PREDICTION.items():
            data = O.load(E.DATA_DIR / E.DATASETS[d])
            mean, mae, held = E.baseline(data, target)
            train = [data.value(r, target) for r in range(1, data.n + 1) if r % 5 != 0]
            self.assertAlmostEqual(mean, sum(train) / len(train), places=9)
            test = [data.value(r, target) for r in held]
            self.assertAlmostEqual(mae, sum(abs(mean - y) for y in test) / len(test), places=9)
            ctx = A.Ctx(data, {}, {"target": target})
            self.assertAlmostEqual(mae, A._baseline_mae(ctx, target)[0], places=9)   # the frozen checker agrees

    def test_baseline_ignores_held_out_values(self):
        held = set(O.held_out(CARS.n))
        raw = [list(r) for r in CARS.raw]
        i = CARS.col(TARGET)
        for r in held:
            raw[r - 1][i] = "1000000"
        changed = CARS.with_raw(raw)
        self.assertEqual(E.baseline(changed, TARGET)[0], E.baseline(CARS, TARGET)[0])

    def test_model_mae_is_recomputed(self):
        p = {int(k): v for k, v in preds(2.5).items()}
        self.assertAlmostEqual(E.model_mae(CARS, TARGET, p), 2.5, places=9)
        p[5] += 79.0
        self.assertAlmostEqual(E.model_mae(CARS, TARGET, p), 2.5 + 1.0, places=9)
        ctx = A.check_request(RID, CARS, {"predictions": {str(k): v for k, v in p.items()}, "mae": 0})
        self.assertAlmostEqual(E.model_mae(CARS, TARGET, p), A._mae(ctx, A._heldout_preds(ctx), TARGET), places=9)

    def test_model_mae_requires_exact_rows(self):
        p = {int(k): v for k, v in preds(1.0).items()}
        with self.assertRaises(ValueError):
            E.model_mae(CARS, TARGET, {r: v for r, v in p.items() if r != 5})
        with self.assertRaises(ValueError):
            E.model_mae(CARS, TARGET, {**p, 1: 0.0})

    def test_mae_ratio(self):
        self.assertEqual(E.mae_ratio(1.0, 4.0), (0.25, ""))
        self.assertEqual(E.mae_ratio(1.0, 0), (None, "baseline_zero"))


class TestGates(unittest.TestCase):
    def test_leakage_gate(self):
        g = lambda **kw: E.leakage_gate(json.loads(semantic(**kw)), RID, TARGET)[0]
        self.assertEqual(g(), "pass")
        self.assertEqual(g(status="fail", failed=[LEAK]), "fail")
        self.assertEqual(g(mask="timed out"), "unverified")
        self.assertEqual(g(mask="crashed", failed=["mask:program runs: crashed"]), "unverified")
        self.assertEqual(g(failed=["mask:answer present: no valid ANSWER"]), "unverified")
        self.assertEqual(g(mask=None), "not_run")
        self.assertEqual(g(failed=["shuffled:mae is computed correctly: x"]), "pass")

    def test_single_target_mask_key(self):
        sem = {"runs": {"base": "ran", "mask": "ran"}, "requests": [{"request_id": "APACHE-03", "failed": []}]}
        self.assertEqual(E.leakage_gate(sem, "APACHE-03", "Performance-")[0], "pass")
        self.assertEqual(E.leakage_gate(sem, "NASA-03", "Effort-")[0], "not_run")

    def test_shuffled_gate(self):
        self.assertEqual(E.shuffled_gate(json.loads(semantic(failed=["shuffled:x: y"])), RID), "fail")
        self.assertEqual(E.shuffled_gate(json.loads(semantic()), RID), "pass")


class TestExtract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.archive, cls.runs, cls.reqs = build_fixture(cls.tmp.name)
        cls.before = sha(cls.archive)
        cls.rows = E.extract(cls.archive, cls.runs, cls.reqs, E.DATA_DIR, domains=("cars",))
        cls.by = {(r["treatment"], r["trial"]): r for r in cls.rows}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_attempt_kept_once(self):
        self.assertEqual(len(self.rows), 18)
        self.assertEqual(len(self.by), 18)
        self.assertEqual({r["request_id"] for r in self.rows}, {RID})

    def test_reasons(self):
        want = {("T1", 1): "", ("T1", 2): "generation_failed", ("T1", 3): "execution_failure",
                ("T2", 1): "answer_missing", ("T2", 2): "answer_invalid_json", ("T2", 3): "answer_schema_error",
                ("T3", 1): "leakage_failed", ("T3", 2): "leakage_unverified", ("T3", 3): "execution_timeout",
                ("T4", 1): "", ("T4", 2): "", ("T4", 3): "", ("T6", 3): "not_attempted"}
        for key, reason in want.items():
            self.assertEqual(self.by[key]["exclusion_reason"], reason, key)
            self.assertEqual(self.by[key]["primary"], reason == "", key)
        for r in self.rows:
            self.assertIn(r["exclusion_reason"], ("",) + E.REASONS)

    def test_values(self):
        _, base, _ = E.baseline(CARS, TARGET)
        t1 = self.by[("T1", 1)]
        self.assertAlmostEqual(t1["model_mae"], 1.0, places=9)
        self.assertAlmostEqual(t1["baseline_mae"], base, places=9)
        self.assertAlmostEqual(t1["mae_ratio"], 1.0 / base, places=9)
        self.assertTrue(t1["beats_baseline"])
        self.assertTrue(t1["reported_mae_matches"])
        self.assertEqual(t1["n_heldout"], 79)

    def test_mean_predictor_does_not_beat_baseline(self):
        r = self.by[("T4", 1)]
        self.assertAlmostEqual(r["mae_ratio"], 1.0, places=9)
        self.assertFalse(r["beats_baseline"])
        self.assertTrue(r["primary"])

    def test_last_answer_line_wins(self):
        self.assertAlmostEqual(self.by[("T4", 2)]["model_mae"], 2.0, places=9)

    def test_stderr_marker_inside_stdout(self):
        self.assertAlmostEqual(self.by[("T4", 3)]["model_mae"], 3.0, places=9)

    def test_leaked_answer_excluded_but_visible(self):
        r = self.by[("T3", 1)]
        self.assertEqual(r["answer_state"], "valid")
        self.assertEqual(r["leakage_gate"], "fail")
        self.assertAlmostEqual(r["model_mae"], 0.5, places=9)
        self.assertFalse(r["primary"])

    def test_failed_rows_have_no_metric(self):
        for key in [("T1", 2), ("T1", 3), ("T2", 1), ("T2", 2), ("T2", 3), ("T3", 3), ("T6", 3)]:
            self.assertEqual(self.by[key]["model_mae"], "", key)
            self.assertEqual(self.by[key]["mae_ratio"], "", key)

    def test_provenance(self):
        r = self.by[("T1", 1)]
        self.assertEqual(r["check_entry"], E.entry_name(ROOT, MODEL, "cars", "T1", 1, 2, "check.txt"))
        with zipfile.ZipFile(self.archive) as z:
            self.assertEqual(r["check_sha256"], hashlib.sha256(z.read(r["check_entry"])).hexdigest())
            self.assertEqual(r["check_crc32"], f"{z.getinfo(r['check_entry']).CRC:08x}")
        self.assertEqual(self.by[("T1", 2)]["check_entry"], "")

    def test_summary(self):
        s = {(x["scope"], x["treatment"]): x for x in E.summarize(self.rows)}
        cars = s[("domain", "all")]
        self.assertEqual(cars["attempted"], 18)
        self.assertEqual(cars["leakage_failed"], 1)
        self.assertEqual(cars["leakage_unverified"], 1)
        self.assertEqual(cars["generation_failed"], 2)            # parse_failed + not attempted
        self.assertEqual(cars["execution_failed"], 2)
        self.assertEqual(cars["primary_n"], 9)
        t4 = s[("domain_treatment", "T4")]
        self.assertEqual((t4["primary_n"], t4["beats_baseline_n"]), (3, 2))
        self.assertAlmostEqual(t4["delivery_rate"], 2 / 3)
        self.assertAlmostEqual(t4["conditional_beat_rate"], 2 / 3)
        t2 = s[("domain_treatment", "T2")]
        self.assertEqual((t2["primary_n"], t2["median_mae_ratio"], t2["delivery_rate"]), (0, "", 0.0))

    def test_archive_untouched(self):
        self.assertEqual(sha(self.archive), self.before)

    def test_duplicate_csv_rows_rejected(self):
        with open(self.runs, newline="") as f:
            lines = f.read().splitlines()
        dup = Path(self.tmp.name) / "dup_runs.csv"
        dup.write_text("\n".join(lines + [lines[1]]) + "\n")
        with self.assertRaises(ValueError):
            E.extract(self.archive, dup, self.reqs, E.DATA_DIR, domains=("cars",))


class TestOutputs(unittest.TestCase):
    def test_guard_refuses_frozen_dir(self):
        for p in [E.FROZEN_DIR, E.FROZEN_DIR / "gpt-oss-20b", E.FROZEN_DIR / "x" / ".." / "y"]:
            with self.assertRaises(ValueError):
                E.guard_out(p)
        self.assertEqual(E.guard_out(E.OUT_ROOT / MODEL), (E.OUT_ROOT / MODEL).resolve())

    def test_main_writes_only_to_out_dir(self):
        frozen = {p: sha(p) for p in E.FROZEN_DIR.rglob("*") if p.is_file()}
        with tempfile.TemporaryDirectory() as tmp:
            archive, runs, reqs = build_fixture(tmp)
            before = {p: sha(p) for p in Path(tmp).iterdir()}
            out = Path(tmp) / "out"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(E.main(["--archive", str(archive), "--runs", str(runs), "--requirements",
                                         str(reqs), "--out", str(out), "--domains", "cars"]), 0)
            self.assertEqual({p.name for p in out.iterdir()},
                             {"prediction_mae.csv", "prediction_summary.csv", "prediction_report.md", "manifest.json"})
            self.assertEqual({p: sha(p) for p in before}, before)
            with open(out / "prediction_mae.csv", newline="") as f:
                got = list(csv.DictReader(f))
            self.assertEqual(len(got), 18)
            self.assertEqual(list(got[0]), E.CSV_FIELDS)
            m = json.loads((out / "manifest.json").read_text())
            self.assertEqual(m["inputs"]["archive"]["sha256"], sha(archive))
            self.assertEqual(m["protocol"]["llm_calls"], 0)
            for name, info in m["outputs"].items():
                self.assertEqual(info["sha256"], sha(out / name))
            self.assertIn("post-observation", (out / "prediction_report.md").read_text())
        self.assertEqual({p: sha(p) for p in E.FROZEN_DIR.rglob("*") if p.is_file()}, frozen)

    def test_main_refuses_frozen_out(self):
        with self.assertRaises(ValueError):
            E.main(["--out", str(E.FROZEN_DIR / "gpt-oss-20b")])


class TestRealArchive(unittest.TestCase):
    """Reads the frozen archive (in memory) and cross-checks against the recorded verdicts."""

    @classmethod
    def setUpClass(cls):
        cls.before = sha(E.ARCHIVE)
        cls.rows = E.extract()

    def test_all_attempts(self):
        self.assertEqual(len(self.rows), 54)
        keys = {(r["domain"], r["treatment"], r["trial"]) for r in self.rows}
        self.assertEqual(len(keys), 54)
        for d in E.DOMAINS:
            self.assertEqual(sum(r["domain"] == d for r in self.rows), 18)

    def test_consistent_with_recorded_verdicts(self):
        for r in self.rows:
            if r["request_status"] == "pass":
                self.assertTrue(r["primary"], r)                   # a passing request is valid and leak-free
                self.assertTrue(r["beats_baseline"], r)
            if r["request_status"] in ("execution_failure", "parse_failure"):
                self.assertFalse(r["primary"], r)
                self.assertEqual(r["model_mae"], "", r)
            if r["leakage_gate"] == "fail":
                self.assertEqual(r["request_status"], "fail", r)
                self.assertFalse(r["primary"], r)

    def test_primary_values_are_finite(self):
        for r in self.rows:
            if r["primary"]:
                self.assertGreater(r["baseline_mae"], 0)
                self.assertGreaterEqual(r["mae_ratio"], 0)
                self.assertEqual(r["leakage_gate"], "pass")

    def test_archive_untouched(self):
        self.assertEqual(sha(E.ARCHIVE), self.before)


if __name__ == "__main__":
    unittest.main()
