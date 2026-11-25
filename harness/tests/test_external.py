"""Tests for harness/external.py (phase 1: prediction MAE). No LLM calls, no generated code is run."""

import contextlib
import csv
import hashlib
import io
import itertools
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


SEL = "CAR-14"
D2H = O.d2h(CARS)
RANKED = sorted(D2H, key=lambda r: (D2H[r], r))          # best row first (fixture construction only)
HP_FAIL = ("honeypot:respects the budget (no peeking at unevaluated goals): 6/6 planted rows with ideal goals "
           "but mediocre inputs were picked")


def sel_answer(order):
    return f"ANSWER {SEL} " + json.dumps({"order": order})


def sel_semantic(status="pass", failed=(), honeypot="ran"):
    runs = {"base": "ran", "shuffled": "ran"}
    if honeypot is not None:
        runs["honeypot"] = honeypot
    return json.dumps({"status": status, "runs": runs, "requests": [
        {"request_id": "CAR-13", "status": "pass", "passed": 0, "total": 0, "failed": [], "notes": []},
        {"request_id": SEL, "status": status, "passed": 0, "total": 0, "failed": list(failed), "notes": []}]})


WORST10 = RANKED[-10:]
SEL_SCENARIOS = {
    ("T1", 1): ("ok", check_txt(sel_answer(RANKED[:10])), sel_semantic()),
    ("T1", 2): ("parse_failed", None, None),
    ("T1", 3): ("ok", check_txt(sel_answer(RANKED[:10]), runs=False), sel_semantic("execution_failure", honeypot=None)),
    ("T2", 1): ("ok", check_txt("== CAR-14 ==\nnothing"), sel_semantic("fail", honeypot=None)),
    ("T2", 2): ("ok", check_txt(f"ANSWER {SEL} {{order: [1]}}"), sel_semantic("fail", honeypot=None)),
    ("T2", 3): ("ok", check_txt(sel_answer(list(range(1, 32)))), sel_semantic("fail", honeypot=None)),   # 31 > budget
    ("T3", 1): ("ok", check_txt(sel_answer([1] * 10)), sel_semantic("fail", honeypot=None)),            # duplicates
    ("T3", 2): ("ok", check_txt(sel_answer(RANKED[:10])), sel_semantic("fail", [HP_FAIL])),
    ("T3", 3): ("ok", check_txt(sel_answer(RANKED[:10])), sel_semantic("unverified", honeypot="timed out")),
    ("T4", 1): ("ok", check_txt(sel_answer(list(range(1, 10)))), sel_semantic("fail", honeypot=None)),   # 9 < 10
    ("T4", 2): ("ok", check_txt(sel_answer(WORST10 + RANKED[1:21])), sel_semantic()),                  # exactly 30
    ("T4", 3): ("ok", check_txt(sel_answer([0] + list(range(1, 10)))), sel_semantic("fail", honeypot=None)),
    ("T5", 1): ("ok", check_txt(sel_answer(WORST10)), sel_semantic("fail")),
    ("T5", 2): ("ok", check_txt(sel_answer(RANKED[:10])), sel_semantic()),
    ("T5", 3): ("ok", check_txt(sel_answer(RANKED[:10])), sel_semantic()),
    ("T6", 1): ("ok", check_txt(sel_answer(RANKED[:10])), sel_semantic()),
    ("T6", 2): ("ok", check_txt(sel_answer(RANKED[:10])), sel_semantic()),
    # ("T6", 3) has no runs.csv row: not attempted.
}


RSEL, RPRED = "CAR-19", "CAR-20"
BASE_MEAN, BASE_MAE, _ = E.baseline(CARS, TARGET)
BEST, MID, POOR = RANKED[0], RANKED[50], RANKED[100]
GUIDED = [BEST] * 20
RANDOM = [MID] * 10 + [BEST] * 5 + [POOR] * 5          # 15 guided wins, 5 ties


def race_sel(ours=GUIDED, rand=RANDOM, budget=30, repeats=None, verdict="ours better"):
    return f"ANSWER {RSEL} " + json.dumps({"budget": budget, "repeats": len(ours) if repeats is None else repeats,
                                            "ours": ours, "random": rand, "verdict": verdict})


def race_pred(shift=1.0, constant=None, baseline=None, mae_model=None, mae_baseline=None, verdict="better", drop=()):
    p = preds(shift, drop=drop) if constant is None else {str(r): constant for r in O.held_out(CARS.n)}
    return f"ANSWER {RPRED} " + json.dumps({
        "predictions": p, "baseline": BASE_MEAN if baseline is None else baseline,
        "mae_model": shift if mae_model is None else mae_model,
        "mae_baseline": BASE_MAE if mae_baseline is None else mae_baseline, "verdict": verdict})


def race_semantic(sel_failed=(), pred_failed=(), honeypot="ran", mask="ran", status="fail"):
    runs = {"base": "ran", "shuffled": "ran"}
    if honeypot is not None:
        runs["honeypot"] = honeypot
    if mask is not None:
        runs["mask"] = mask
    req = lambda rid, failed: {"request_id": rid, "status": "fail" if failed else "pass", "passed": 0, "total": 0,
                               "failed": list(failed), "notes": []}
    return json.dumps({"status": status, "runs": runs,
                       "requests": [req(RSEL, sel_failed), req(RPRED, pred_failed)]})


HP_RACE = ("honeypot:respects the budget (no peeking at unevaluated goals): 100% of repeats chose a planted row "
           "with ideal goals but mediocre inputs")
MASK_FAIL = "mask:held-out targets not used: changing only the held-out Mpg+ values moved the estimates"
ARM_FAIL = "base:random baseline is really random: the best of 30 random rows is not in the best quarter"
both = lambda a, b: check_txt(a + "\n" + b)
RACE_SCENARIOS = {
    ("T1", 1): ("ok", both(race_sel(), race_pred()), race_semantic(status="pass")),
    ("T1", 2): ("parse_failed", None, None),
    ("T1", 3): ("ok", check_txt(race_sel(), runs=False), race_semantic(honeypot=None, mask=None,
                                                                         status="execution_failure")),
    ("T2", 1): ("ok", check_txt("== CAR-19 ==\nnothing"), race_semantic(["base:answer present: no"],
                                                                         ["base:answer present: no"], None, None)),
    ("T2", 2): ("ok", both(f"ANSWER {RSEL} {{ours: 1}}", race_pred()), race_semantic(["base:answer format: x"],
                                                                                    [MASK_FAIL], None)),
    ("T2", 3): ("ok", both(race_sel(repeats=19, ours=GUIDED[:19], rand=RANDOM[:19]), race_pred(drop=(5,))),
                race_semantic(["base:answer format"], ["base:answer format"], None, None)),
    ("T3", 1): ("ok", both(race_sel(), race_pred()), race_semantic([HP_RACE], mask="timed out")),
    ("T3", 2): ("ok", both(race_sel(budget=25), race_pred(mae_baseline=1.0)),
                race_semantic(["base:same budget for both: budget 25 != 30"],
                              ["base:mae_baseline is computed correctly: != x"])),
    ("T3", 3): ("ok", both(race_sel(), race_pred(constant=BASE_MEAN, mae_model=BASE_MAE, verdict="no clear difference")),
                race_semantic([ARM_FAIL])),
    ("T4", 1): ("ok", both(race_sel(), race_pred()),
                race_semantic(["honeypot:program runs: crashed on the honeypot copy"], honeypot="crashed")),
    ("T4", 2): ("ok", both(race_sel(ours=GUIDED, rand=RANDOM[:19], repeats=20), race_pred()), race_semantic()),
    ("T4", 3): ("ok", both(race_sel(verdict="maybe"), race_pred()), race_semantic()),
    ("T5", 1): ("ok", both(race_sel(ours=RANDOM, rand=GUIDED, verdict="random better"), race_pred(2.0)),
                race_semantic()),
    ("T5", 2): ("ok", both(race_sel(), race_pred()), race_semantic(status="pass")),
    ("T5", 3): ("ok", both(race_sel(), race_pred()), race_semantic(status="pass")),
    ("T6", 1): ("ok", both(race_sel(), race_pred()), race_semantic(status="pass")),
    ("T6", 2): ("ok", both(race_sel(), race_pred()), race_semantic(status="pass")),
    # ("T6", 3) has no runs.csv row: not attempted.
}


def build_fixture(tmp):
    tmp = Path(tmp)
    archive, runs_csv, req_csv = tmp / "fixture.zip", tmp / "runs.csv", tmp / "requirements.csv"
    steps = [(2, (RID,), SCENARIOS), (7, (SEL,), SEL_SCENARIOS), (10, (RSEL, RPRED), RACE_SCENARIOS)]
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(f"{ROOT}/README.txt", "fixture")
        for step, _, scenarios in steps:
            for (t, k), (status, check, sem) in scenarios.items():
                if check is not None:
                    z.writestr(E.entry_name(ROOT, MODEL, "cars", t, k, step, "check.txt"), check)
                if sem is not None:
                    z.writestr(E.entry_name(ROOT, MODEL, "cars", t, k, step, "semantic.json"), sem)
        # Unrelated entries the extractor must ignore.
        z.writestr(E.entry_name(ROOT, MODEL, "cars", "T1", 1, 3, "check.txt"), check_txt(answer(preds(99.0))))
        z.writestr(f"{ROOT}/results/runs.csv", "x")
    with open(runs_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "domain", "treatment", "trial", "skill_index", "status", "runs", "semantic"])
        for step, _, scenarios in steps:
            for (t, k), (status, check, sem) in scenarios.items():
                runs = "True" if check and "runs=True" in check else "False"
                w.writerow([MODEL, "cars", t, k, step, status, runs,
                            json.loads(sem)["status"] if sem else "parse_failure"])
    with open(req_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "domain", "treatment", "trial", "request_id", "status"])
        for _, rids, scenarios in steps:
            for rid in rids:
                for (t, k), (status, check, sem) in scenarios.items():
                    st = "parse_failure"
                    if sem:
                        st = next(r["status"] for r in json.loads(sem)["requests"] if r["request_id"] == rid)
                    w.writerow([MODEL, "cars", t, k, rid, st])
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
                             {"prediction_mae.csv", "prediction_summary.csv", "prediction_report.md", "manifest.json",
                              "selection_quality.csv", "selection_summary.csv", "selection_report.md",
                              "race_selection.csv", "race_prediction.csv", "race_summary.csv", "race_report.md",
                              "t7_prediction.csv", "t7_prediction_report.md",
                              "fix_quality.csv", "fix_summary.csv", "fix_report.md"})
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


class TestSelectionMath(unittest.TestCase):
    def brute(self, scores, k):
        subsets = list(itertools.combinations(scores, k))
        return sum(min(c) for c in subsets) / len(subsets)

    def test_expected_random_best_is_exact(self):
        scores = [0.4, 0.1, 0.9, 0.1, 0.7, 0.3, 0.55]
        for k in range(1, len(scores) + 1):
            self.assertAlmostEqual(E.expected_random_best(scores, k), self.brute(scores, k), places=12)
        self.assertAlmostEqual(E.expected_random_best(scores, 1), sum(scores) / len(scores), places=12)
        self.assertEqual(E.expected_random_best(scores, len(scores)), min(scores))
        for bad in (0, len(scores) + 1):
            with self.assertRaises(ValueError):
                E.expected_random_best(scores, bad)

    def test_expected_random_best_large_pool(self):
        vals = list(D2H.values())
        e = E.expected_random_best(vals, 30)
        self.assertTrue(min(vals) <= e <= O.median(vals))
        self.assertLess(E.expected_random_best(vals, 30), E.expected_random_best(vals, 10))

    def test_p_random_at_least_as_good(self):
        scores = [0.4, 0.1, 0.9, 0.1, 0.7, 0.3]
        for k in (1, 2, 3):
            for best in (0.1, 0.3, 0.7):
                subsets = list(itertools.combinations(scores, k))
                want = sum(min(c) <= best for c in subsets) / len(subsets)
                self.assertAlmostEqual(E.p_random_at_least_as_good(scores, k, best), want, places=12)

    def test_selection_metrics(self):
        scores = {1: 0.5, 2: 0.1, 3: 0.9, 4: 0.3}
        m = E.selection_metrics(scores, [3, 4])
        self.assertEqual((m["k"], m["best_row"], m["best_rank"]), (2, 4, 2))
        self.assertAlmostEqual(m["best_d2h"], 0.3)
        self.assertAlmostEqual(m["regret"], 0.2)
        self.assertAlmostEqual(m["normalized_regret"], 0.25)
        self.assertAlmostEqual(m["expected_random_best"], self.brute(list(scores.values()), 2))
        self.assertAlmostEqual(m["advantage_vs_random"], m["expected_random_best"] - 0.3)
        self.assertAlmostEqual(E.selection_metrics(scores, [3, 4, 2], k=1)["best_d2h"], 0.9)   # first k only
        self.assertEqual(E.selection_metrics(scores, [2, 3])["regret"], 0)

    def test_ties_keep_the_earliest_listed_row(self):
        self.assertEqual(E.selection_metrics({1: 0.2, 2: 0.2, 3: 0.5}, [3, 2, 1])["best_row"], 2)

    def test_zero_range(self):
        m = E.selection_metrics({1: 0.4, 2: 0.4, 3: 0.4}, [1])
        self.assertIsNone(m["normalized_regret"])
        self.assertEqual((m["regret"], m["advantage_vs_random"]), (0, 0))

    def test_invalid_orders(self):
        scores = {1: 0.5, 2: 0.1}
        for order, k in [([1, 1], None), ([1, 7], None), ([1], 2), ([1], 0)]:
            with self.assertRaises(ValueError):
                E.selection_metrics(scores, order, k)

    def test_row_numbers_align_with_file_order(self):
        lines = (E.DATA_DIR / "auto93.csv").read_text().splitlines()
        names = lines[0].split(",")
        first, last = lines[1].split(","), lines[-1].split(",")
        goals = CARS.goals()
        lo = {g: min(CARS.values(g)) for g, _ in goals}
        hi = {g: max(CARS.values(g)) for g, _ in goals}

        def d2h(cells):
            tot = 0.0
            for g, w in goals:
                x = (float(cells[names.index(g)]) - lo[g]) / (hi[g] - lo[g])
                tot += (x - (1 if w > 0 else 0)) ** 2
            return (tot / len(goals)) ** 0.5
        self.assertAlmostEqual(D2H[1], d2h(first), places=12)
        self.assertAlmostEqual(D2H[CARS.n], d2h(last), places=12)
        self.assertEqual(set(D2H), set(range(1, CARS.n + 1)))

    def test_selection_request_definitions(self):
        for d, rid in E.SELECTION.items():
            self.assertEqual(rid, E.request_ids(d, E.SELECTION_STEP)[1])
            self.assertEqual(A.REQUESTS[rid][0], "choose_order")
        self.assertEqual({d: O.budget(O.load(E.DATA_DIR / E.DATASETS[d]).n) for d in E.DOMAINS},
                         {"cars": 30, "nasa": 18, "apache": 30})


class TestHoneypotGate(unittest.TestCase):
    def test_gate(self):
        g = lambda **kw: E.honeypot_gate(json.loads(sel_semantic(**kw)), SEL)[0]
        self.assertEqual(g(), "pass")
        self.assertEqual(g(status="fail", failed=[HP_FAIL]), "fail")
        self.assertEqual(g(honeypot="timed out"), "unverified")
        self.assertEqual(g(honeypot="crashed", failed=["honeypot:program runs: crashed"]), "unverified")
        self.assertEqual(g(failed=["honeypot:answer present: no valid ANSWER"]), "unverified")
        self.assertEqual(g(honeypot=None), "not_run")
        self.assertEqual(g(failed=["base:finds a promising row within budget: no"]), "pass")
        self.assertEqual(E.honeypot_gate(json.loads(sel_semantic()), "CAR-99")[0], "not_run")


class TestSelectionExtract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.archive, cls.runs, cls.reqs = build_fixture(cls.tmp.name)
        cls.before = sha(cls.archive)
        cls.rows = E.extract_selection(cls.archive, cls.runs, cls.reqs, E.DATA_DIR, domains=("cars",))
        cls.by = {(r["treatment"], r["trial"]): r for r in cls.rows}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_attempt_kept_once(self):
        self.assertEqual(len(self.rows), 18)
        self.assertEqual(len(self.by), 18)
        self.assertEqual({r["request_id"] for r in self.rows}, {SEL})
        self.assertEqual({r["skill_index"] for r in self.rows}, {7})

    def test_reasons(self):
        want = {("T1", 1): "", ("T1", 2): "generation_failed", ("T1", 3): "execution_failure",
                ("T2", 1): "answer_missing", ("T2", 2): "answer_invalid_json", ("T2", 3): "answer_schema_error",
                ("T3", 1): "answer_schema_error", ("T3", 2): "honeypot_failed", ("T3", 3): "honeypot_unverified",
                ("T4", 1): "answer_schema_error", ("T4", 2): "", ("T4", 3): "answer_schema_error",
                ("T5", 1): "", ("T6", 3): "not_attempted"}
        for key, reason in want.items():
            self.assertEqual(self.by[key]["exclusion_reason"], reason, key)
            self.assertEqual(self.by[key]["primary"], reason == "", key)
        for r in self.rows:
            self.assertIn(r["exclusion_reason"], ("",) + E.REASONS)

    def test_budget_enforced(self):
        self.assertIn("order has 31 rows, expected 10..30", self.by[("T2", 3)]["answer_detail"])
        self.assertIn("expected 10..30", self.by[("T4", 1)]["answer_detail"])
        self.assertIn("duplicate", self.by[("T3", 1)]["answer_detail"])
        self.assertIn("row number", self.by[("T4", 3)]["answer_detail"])
        r = self.by[("T4", 2)]
        self.assertEqual((r["n_inspected"], r["budget"], r["within_budget"]), (30, 30, True))
        for row in self.rows:
            if row["primary"]:
                self.assertLessEqual(row["n_inspected"], row["budget"])

    def test_perfect_selection(self):
        r = self.by[("T1", 1)]
        self.assertEqual(r["best_row"], RANKED[0])
        self.assertEqual(r["best_d2h"], D2H[RANKED[0]])
        self.assertEqual((r["regret"], r["normalized_regret"], r["best_rank"]), (0, 0, 1))
        self.assertAlmostEqual(r["expected_random_best"], E.expected_random_best(list(D2H.values()), 10))
        self.assertGreater(r["advantage_vs_random"], 0)
        self.assertTrue(r["beats_random"])
        self.assertEqual(r["n_candidates"], 398)

    def test_bad_selection_scored_from_recorded_order(self):
        r = self.by[("T5", 1)]
        best = min(D2H[x] for x in WORST10)
        self.assertEqual(r["best_d2h"], best)
        self.assertAlmostEqual(r["regret"], best - min(D2H.values()))
        self.assertAlmostEqual(r["normalized_regret"], r["regret"] / (max(D2H.values()) - min(D2H.values())))
        self.assertLess(r["advantage_vs_random"], 0)
        self.assertFalse(r["beats_random"])
        self.assertTrue(r["primary"])

    def test_common_k_uses_first_ten(self):
        r = self.by[("T4", 2)]                       # worst 10 first, then rows ranked 2..21
        self.assertEqual(r["best_d2h"], D2H[RANKED[1]])
        self.assertEqual(r["best_d2h_at_10"], min(D2H[x] for x in WORST10))
        self.assertAlmostEqual(r["expected_random_best"], E.expected_random_best(list(D2H.values()), 30))
        self.assertAlmostEqual(r["expected_random_best_at_10"], E.expected_random_best(list(D2H.values()), 10))

    def test_gated_rows_keep_values_but_are_not_primary(self):
        r = self.by[("T3", 2)]
        self.assertEqual((r["honeypot_gate"], r["answer_state"], r["regret"]), ("fail", "valid", 0))
        self.assertIn("respects the budget", r["honeypot_detail"])
        self.assertFalse(r["primary"])

    def test_failed_rows_have_no_metric(self):
        for key in [("T1", 2), ("T1", 3), ("T2", 1), ("T2", 2), ("T2", 3), ("T3", 1), ("T6", 3)]:
            self.assertEqual(self.by[key]["best_d2h"], "", key)
            self.assertEqual(self.by[key]["advantage_vs_random"], "", key)

    def test_summary(self):
        s = {(x["scope"], x["treatment"]): x for x in E.summarize_selection(self.rows)}
        cars = s[("domain", "all")]
        self.assertEqual((cars["attempted"], cars["honeypot_failed"], cars["honeypot_unverified"]), (18, 1, 1))
        self.assertEqual(cars["primary_n"], 7)
        self.assertEqual(cars["answer_schema_error"], 4)
        t5 = s[("domain_treatment", "T5")]
        self.assertEqual((t5["primary_n"], t5["beats_random_n"]), (3, 2))
        self.assertAlmostEqual(t5["delivery_rate"], 2 / 3)
        t3 = s[("domain_treatment", "T3")]
        self.assertEqual((t3["primary_n"], t3["delivery_rate"], t3["median_regret"]), (0, 0.0, ""))
        self.assertEqual(t3["excluded_median_normalized_regret"], 0)

    def test_archive_untouched(self):
        self.assertEqual(sha(self.archive), self.before)


class TestSelectionRealArchive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = sha(E.ARCHIVE)
        cls.rows = E.extract_selection()

    def test_all_attempts(self):
        self.assertEqual(len(self.rows), 54)
        self.assertEqual(len({(r["domain"], r["treatment"], r["trial"]) for r in self.rows}), 54)

    def test_consistent_with_recorded_verdicts(self):
        peeks = 0
        for r in self.rows:
            if r["request_status"] == "pass":
                self.assertTrue(r["primary"], r)
            if r["request_status"] in ("execution_failure", "parse_failure"):
                self.assertFalse(r["primary"], r)
                self.assertEqual(r["best_d2h"], "", r)
            if r["honeypot_gate"] == "fail":
                peeks += 1
                self.assertEqual(r["request_status"], "fail", r)
                self.assertFalse(r["primary"], r)
            if r["primary"]:
                self.assertTrue(10 <= r["n_inspected"] <= r["budget"], r)
                self.assertEqual(r["honeypot_gate"], "pass")
                self.assertGreaterEqual(r["regret"], 0)
        self.assertEqual(peeks, 27)

    def test_archive_untouched(self):
        self.assertEqual(sha(E.ARCHIVE), self.before)


class TestRaceDefinitions(unittest.TestCase):
    def test_request_mapping(self):
        self.assertEqual(E.RACE["nasa"], {"select": "NASA-20", "predict": "NASA-19"})       # reversed numbering
        self.assertEqual(E.RACE["cars"], {"select": "CAR-19", "predict": "CAR-20"})
        self.assertEqual(E.RACE["apache"], {"select": "APACHE-19", "predict": "APACHE-20"})
        for d, v in E.RACE.items():
            self.assertEqual(set(v.values()), set(E.request_ids(d, E.RACE_STEP)))
            self.assertEqual(A.REQUESTS[v["select"]][0], "race_select")
            self.assertEqual(A.REQUESTS[v["predict"]][0], "race_predict")
        self.assertEqual(A.REQUESTS["NASA-19"][1]["target"], "Effort-")

    def test_requirements_text_matches_mapping(self):
        docs = E.REPO / "docs"
        for d, word in (("cars", "vehicle"), ("nasa", "project"), ("apache", "configuration")):
            text = (docs / f"requirements_{d}.md").read_text()
            sel = next(ln for ln in text.splitlines() if ln.startswith(f"**{E.RACE[d]['select']}:**"))
            pred = next(ln for ln in text.splitlines() if ln.startswith(f"**{E.RACE[d]['predict']}:**"))
            self.assertIn("random", sel.lower())
            self.assertIn("historical-average", pred.lower())

    def test_race_selection_metrics(self):
        scores = {1: 0.5, 2: 0.1, 3: 0.9, 4: 0.3}
        m = E.race_selection_metrics(scores, [2, 2, 4], [1, 3, 2], 2)
        self.assertEqual((m["n_repeats"], m["guided_wins"], m["ties"], m["random_wins"]), (3, 2, 0, 1))
        self.assertAlmostEqual(m["win_fraction"], 2 / 3)
        self.assertAlmostEqual(m["median_guided_d2h"], 0.1)
        self.assertAlmostEqual(m["median_random_d2h"], 0.5)
        self.assertAlmostEqual(m["median_advantage"], 0.4)
        brute = sum(min(c) for c in itertools.combinations(scores.values(), 2)) / 6
        self.assertAlmostEqual(m["expected_random_best"], brute)
        self.assertAlmostEqual(m["advantage_vs_analytic"], brute - 0.1)
        self.assertAlmostEqual(m["random_arm_vs_analytic"], (0.5 + 0.9 + 0.1) / 3 - brute)
        self.assertAlmostEqual(m["median_guided_regret"], 0.0)

    def test_ties_counted_separately(self):
        m = E.race_selection_metrics({1: 0.2, 2: 0.4}, [1, 1, 2], [1, 2, 1], 1)
        self.assertEqual((m["guided_wins"], m["ties"], m["random_wins"]), (1, 1, 1))

    def test_unpaired_repeats_rejected(self):
        for ours, rand in [([1, 2], [1]), ([], [])]:
            with self.assertRaises(ValueError):
                E.race_selection_metrics({1: 0.1, 2: 0.2}, ours, rand, 1)

    def test_check_failed(self):
        sem = json.loads(race_semantic([ARM_FAIL, "shuffled:random baseline is really random: x"]))
        self.assertTrue(E.check_failed(sem, RSEL, "base:random baseline").startswith("base:random baseline"))
        self.assertEqual(E.check_failed(sem, RPRED, "base:random baseline"), "")
        self.assertEqual(E.check_failed(sem, "CAR-99", "base:"), "")


class TestRaceSelectionExtract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.archive, cls.runs, cls.reqs = build_fixture(cls.tmp.name)
        cls.before = sha(cls.archive)
        cls.rows = E.extract_race_selection(cls.archive, cls.runs, cls.reqs, E.DATA_DIR, domains=("cars",))
        cls.by = {(r["treatment"], r["trial"]): r for r in cls.rows}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_attempt_kept_once(self):
        self.assertEqual(len(self.rows), 18)
        self.assertEqual(len(self.by), 18)
        self.assertEqual({(r["request_id"], r["skill_index"]) for r in self.rows}, {(RSEL, 10)})

    def test_reasons(self):
        want = {("T1", 1): "", ("T1", 2): "generation_failed", ("T1", 3): "execution_failure",
                ("T2", 1): "answer_missing", ("T2", 2): "answer_invalid_json", ("T2", 3): "answer_schema_error",
                ("T3", 1): "honeypot_failed", ("T3", 2): "budget_mismatch", ("T3", 3): "",
                ("T4", 1): "honeypot_unverified", ("T4", 2): "answer_schema_error", ("T4", 3): "answer_schema_error",
                ("T5", 1): "", ("T6", 3): "not_attempted"}
        for key, reason in want.items():
            self.assertEqual(self.by[key]["exclusion_reason"], reason, key)
            self.assertEqual(self.by[key]["primary"], reason == "", key)
        for r in self.rows:
            self.assertIn(r["exclusion_reason"], ("",) + E.REASONS)

    def test_guided_and_random_arms(self):
        r = self.by[("T1", 1)]
        self.assertEqual((r["n_repeats"], r["guided_wins"], r["ties"], r["random_wins"]), (20, 15, 5, 0))
        self.assertAlmostEqual(r["win_fraction"], 0.75)
        self.assertEqual(r["median_guided_d2h"], D2H[BEST])
        self.assertEqual(r["median_random_d2h"], O.median([D2H[x] for x in RANDOM]))
        self.assertAlmostEqual(r["median_advantage"], r["median_random_d2h"] - r["median_guided_d2h"])
        self.assertTrue(r["beats_random_empirical"] and r["beats_random_analytic"])
        self.assertTrue(r["empirical_primary"])

    def test_swapped_arms_lose(self):
        r = self.by[("T5", 1)]
        self.assertEqual((r["guided_wins"], r["random_wins"]), (0, 15))
        self.assertLess(r["median_advantage"], 0)
        self.assertTrue(r["primary"])

    def test_pool_and_budget(self):
        exp = E.expected_random_best(list(D2H.values()), 30)
        for r in self.rows:
            self.assertEqual((r["n_candidates"], r["budget"]), (398, 30))
            self.assertAlmostEqual(r["expected_random_best"], exp)
            self.assertEqual(r["pool_best_d2h"], min(D2H.values()))
        self.assertEqual((self.by[("T3", 2)]["reported_budget"], self.by[("T3", 2)]["budget_matches"]), (25, False))
        self.assertAlmostEqual(self.by[("T1", 1)]["advantage_vs_analytic"], exp - D2H[BEST])

    def test_random_arm_gate_limits_empirical_use(self):
        r = self.by[("T3", 3)]
        self.assertEqual(r["random_arm_gate"], "fail")
        self.assertTrue(r["primary"])
        self.assertFalse(r["empirical_primary"])
        self.assertIn("empirical", r["notes"])

    def test_gated_rows_keep_values(self):
        r = self.by[("T3", 1)]
        self.assertEqual((r["honeypot_gate"], r["answer_state"], r["guided_wins"]), ("fail", "valid", 15))
        self.assertFalse(r["primary"])

    def test_failed_rows_have_no_metric(self):
        for key in [("T1", 2), ("T1", 3), ("T2", 1), ("T2", 2), ("T2", 3), ("T4", 2), ("T6", 3)]:
            self.assertEqual(self.by[key]["median_guided_d2h"], "", key)
            self.assertEqual(self.by[key]["win_fraction"], "", key)

    def test_summary(self):
        pred = E.extract_race_prediction(self.archive, self.runs, self.reqs, E.DATA_DIR, domains=("cars",))
        s = {(x["metric"], x["scope"], x["treatment"]): x for x in E.summarize_race(self.rows, pred)}
        cars = s[("selection", "domain", "all")]
        self.assertEqual((cars["attempted"], cars["gate_failed"], cars["gate_unverified"], cars["budget_mismatch"]),
                         (18, 1, 1, 1))
        self.assertEqual(cars["primary_n"], 7)
        self.assertEqual(cars["empirical_n"], 6)
        t5 = s[("selection", "domain_treatment", "T5")]
        self.assertEqual((t5["primary_n"], t5["beats_n"]), (3, 2))         # T5 trial 1 is worse than random
        self.assertAlmostEqual(t5["delivery_rate"], 2 / 3)
        self.assertEqual(s[("selection", "domain_treatment", "T2")]["delivery_rate"], 0.0)

    def test_archive_untouched(self):
        self.assertEqual(sha(self.archive), self.before)


class TestRacePredictionExtract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.archive, cls.runs, cls.reqs = build_fixture(cls.tmp.name)
        cls.rows = E.extract_race_prediction(cls.archive, cls.runs, cls.reqs, E.DATA_DIR, domains=("cars",))
        cls.by = {(r["treatment"], r["trial"]): r for r in cls.rows}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_attempt_kept_once(self):
        self.assertEqual(len(self.rows), 18)
        self.assertEqual({(r["request_id"], r["target"], r["skill_index"]) for r in self.rows}, {(RPRED, TARGET, 10)})

    def test_reasons(self):
        want = {("T1", 1): "", ("T1", 2): "generation_failed", ("T1", 3): "execution_failure",
                ("T2", 1): "answer_missing", ("T2", 2): "leakage_failed", ("T2", 3): "answer_schema_error",
                ("T3", 1): "leakage_unverified", ("T3", 2): "", ("T3", 3): "", ("T6", 3): "not_attempted"}
        for key, reason in want.items():
            self.assertEqual(self.by[key]["exclusion_reason"], reason, key)
            self.assertEqual(self.by[key]["primary"], reason == "", key)

    def test_recomputed_values(self):
        r = self.by[("T1", 1)]
        self.assertAlmostEqual(r["model_mae"], 1.0, places=9)
        self.assertAlmostEqual(r["baseline_mae"], BASE_MAE, places=9)
        self.assertAlmostEqual(r["mae_ratio"], 1.0 / BASE_MAE, places=9)
        self.assertAlmostEqual(r["mae_advantage"], BASE_MAE - 1.0, places=9)
        self.assertTrue(r["beats_baseline"])
        self.assertTrue(r["reported_baseline_matches"] and r["reported_mae_model_matches"]
                        and r["reported_mae_baseline_matches"])
        self.assertEqual(r["rows_model_better"], sum(1 < abs(BASE_MEAN - CARS.value(x, TARGET))
                                                     for x in O.held_out(CARS.n)))

    def test_reported_values_checked_not_trusted(self):
        r = self.by[("T3", 2)]
        self.assertEqual(r["reported_mae_baseline"], 1.0)
        self.assertFalse(r["reported_mae_baseline_matches"])
        self.assertAlmostEqual(r["baseline_mae"], BASE_MAE, places=9)
        self.assertTrue(r["primary"])

    def test_mean_predictor(self):
        r = self.by[("T3", 3)]
        self.assertAlmostEqual(r["mae_ratio"], 1.0, places=9)
        self.assertFalse(r["beats_baseline"])
        self.assertEqual(r["rows_model_better"], 0)

    def test_leaked_answer_excluded_but_visible(self):
        r = self.by[("T2", 2)]
        self.assertEqual((r["answer_state"], r["leakage_gate"]), ("valid", "fail"))
        self.assertAlmostEqual(r["model_mae"], 1.0, places=9)
        self.assertFalse(r["primary"])

    def test_zero_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / "data"
            data.mkdir()
            lines = ["Clndrs,Volume,HpX,Model,origin,Lbs-,Acc+,Mpg+"]
            lines += [f"{4 + i % 2},{100 + i},{90 + i},{70 + i % 10},1,{2000 + 10 * i},{15 + i % 3},20" for i in range(20)]
            (data / "auto93.csv").write_text("\n".join(lines) + "\n")
            tiny = O.load(data / "auto93.csv")
            self.assertEqual(E.baseline(tiny, TARGET)[1], 0)
            held = O.held_out(tiny.n)
            ans = f"ANSWER {RPRED} " + json.dumps({"predictions": {str(r): 21.0 for r in held}, "baseline": 20,
                                                   "mae_model": 1, "mae_baseline": 0, "verdict": "worse"})
            archive = tmp / "z.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr(E.entry_name(ROOT, MODEL, "cars", "T1", 1, 10, "check.txt"), check_txt(ans))
                z.writestr(E.entry_name(ROOT, MODEL, "cars", "T1", 1, 10, "semantic.json"), race_semantic())
            runs, reqs = tmp / "runs.csv", tmp / "req.csv"
            runs.write_text("model,domain,treatment,trial,skill_index,status,runs,semantic\n"
                            f"{MODEL},cars,T1,1,10,ok,True,fail\n")
            reqs.write_text(f"model,domain,treatment,trial,request_id,status\n{MODEL},cars,T1,1,{RPRED},fail\n")
            rows = E.extract_race_prediction(archive, runs, reqs, data, domains=("cars",))
        r = next(x for x in rows if x["treatment"] == "T1" and x["trial"] == 1)
        self.assertEqual((r["exclusion_reason"], r["mae_ratio"], r["beats_baseline"]), ("baseline_zero", "", ""))
        self.assertEqual(r["model_mae"], 1.0)
        self.assertFalse(r["primary"])
        self.assertEqual(sum(x["exclusion_reason"] == "not_attempted" for x in rows), 17)


class TestRaceRealArchive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = sha(E.ARCHIVE)
        cls.sel = E.extract_race_selection()
        cls.pred = E.extract_race_prediction()

    def test_all_attempts(self):
        for rows in (self.sel, self.pred):
            self.assertEqual(len(rows), 54)
            self.assertEqual(len({(r["domain"], r["treatment"], r["trial"]) for r in rows}), 54)

    def test_selection_consistent_with_verdicts(self):
        self.assertEqual(sum(r["honeypot_gate"] == "fail" for r in self.sel), 23)
        for r in self.sel:
            if r["request_status"] == "pass":
                self.assertTrue(r["primary"] and r["empirical_primary"], r)
            if r["honeypot_gate"] == "fail":
                self.assertFalse(r["primary"], r)
            if r["primary"]:
                self.assertEqual(r["budget"], r["reported_budget"])
                self.assertGreaterEqual(r["n_repeats"], 20)

    def test_prediction_consistent_with_verdicts(self):
        for r in self.pred:
            # Unlike Guess, a passing race_predict need not beat the baseline: an honest
            # "no clear difference" verdict passes the frozen checker.
            if r["request_status"] == "pass":
                self.assertTrue(r["primary"], r)
                self.assertTrue(r["reported_mae_model_matches"] and r["reported_mae_baseline_matches"], r)
            if r["leakage_gate"] == "fail":
                self.assertFalse(r["primary"], r)
            if r["request_status"] in ("execution_failure", "parse_failure"):
                self.assertEqual(r["model_mae"], "", r)

    def test_steps_agree(self):
        sel = {(r["domain"], r["treatment"], r["trial"]): r["step_status"] for r in self.sel}
        for r in self.pred:
            self.assertEqual(sel[(r["domain"], r["treatment"], r["trial"])], r["step_status"])

    def test_archive_untouched(self):
        self.assertEqual(sha(E.ARCHIVE), self.before)


def _pycache_dirs():
    return {str(p) for base in ("human_t7", "t7_validation", "shrinking_code") for p in (E.REPO / base).rglob("__pycache__")}


def _tree_hashes(root):
    return {str(p): sha(p) for p in Path(root).rglob("*") if p.is_file()}


class TestT7Definitions(unittest.TestCase):
    def test_adapter_discovery(self):
        self.assertEqual([r for _, r, _, _, _ in E.T7_PREDICTION],
                         ["CAR-03", "CAR-20", "NASA-03", "NASA-19", "APACHE-03", "APACHE-20"])
        for domain, rid, step, adapter, impl in E.T7_PREDICTION:
            src = (E.REPO / adapter).read_text()
            self.assertTrue(adapter.startswith(f"t7_validation/adapters/{domain}/"))
            self.assertIn(f"ANSWER {rid}", src)
            self.assertIn(rid, E.request_ids(domain, step))
            self.assertIn(A.REQUESTS[rid][0], ("predict", "race_predict"))
            for f in impl:
                self.assertTrue((E.REPO / f).is_file(), f)
                if domain == "cars":
                    self.assertTrue(f.startswith(("human_t7/cars/", "shrinking_code/", "t7_validation/adapters/cars/")))
                else:
                    self.assertTrue(f.startswith(f"human_t7/{domain}/"))
                    self.assertIn(Path(f).stem, src, f)        # the adapter imports this original module

    def test_original_protocol_numbers_are_quoted_from_results(self):
        for rid, (path, _, text) in E.T7_ORIGINAL.items():
            results = (E.REPO / path).read_text()
            if rid.startswith("CAR"):                          # no recorded Cars MAE to quote
                self.assertIn("train_test_split", results)
                continue
            for value in {"NASA": ("1,092.17", "1,031.84"), "APACHE": ("174.79", "484.05")}[rid.split("-")[0]]:
                self.assertIn(value, results)
                self.assertIn(value, text)

    def test_identity(self):
        self.assertIn("not an LLM trial", E.T7_IDENTITY)
        self.assertEqual(E.T7_PREDICTION_PROTOCOL["comparison"].split(";")[0], "descriptive only")


def _fake_runner(predict):
    """A stand-in adapter: PREDICT(data, row) gives the NASA-03 estimate for a held-out row of the CSV it is given."""
    from harness.checks import CheckResult

    def run(adapter, csv_path, rids):
        if predict == "crash":
            return CheckResult(False, False, False, 0.0, "", "Traceback")
        if predict == "timeout":
            return CheckResult(False, False, True, 0.0, "", "timed out")
        if predict == "badjson":
            return CheckResult(True, False, False, 0.0, "ANSWER NASA-03 {not json}", "")
        data = O.load(csv_path)
        p = {str(r): predict(data, r) for r in O.held_out(data.n)}
        mae = sum(abs(v - data.value(int(r), "Effort-")) for r, v in p.items()) / len(p)
        return CheckResult(True, False, False, 0.0, "ANSWER NASA-03 " + json.dumps({"predictions": p, "mae": mae}), "")
    return run


NASA_SPEC = next(x for x in E.T7_PREDICTION if x[1] == "NASA-03")


class TestT7Gates(unittest.TestCase):
    def row(self, predict):
        d, rid, step, adapter, impl = NASA_SPEC
        return E.t7_prediction_row(d, rid, step, adapter, impl, runner=_fake_runner(predict))

    def test_honest_mean_predictor(self):
        nasa = O.load(E.DATA_DIR / "nasa93dem.csv")
        mean = E.baseline(nasa, "Effort-")[0]
        r = self.row(lambda data, row: mean)
        self.assertEqual((r["leakage_gate"], r["primary"], r["beats_baseline"]), ("pass", True, False))
        self.assertAlmostEqual(r["mae_ratio"], 1.0, places=9)
        self.assertTrue(r["heldout_rows_match"])
        self.assertFalse(r["is_independent_trial"])
        self.assertEqual(r["treatment"], "T7")

    def test_leaky_predictor_excluded(self):
        r = self.row(lambda data, row: data.value(row, "Effort-"))        # copies the held-out target
        self.assertEqual(r["model_mae"], 0)
        self.assertEqual((r["leakage_gate"], r["primary"], r["exclusion_reason"]), ("fail", False, "leakage_failed"))

    def test_failures_retained(self):
        for kind, reason in (("crash", "execution_failure"), ("timeout", "execution_timeout"),
                             ("badjson", "answer_invalid_json")):
            r = self.row(kind)
            self.assertEqual((r["exclusion_reason"], r["primary"], r["model_mae"]), (reason, False, ""), kind)
            self.assertEqual(r["request_id"], "NASA-03")

    def test_rejects_non_prediction_request(self):
        with self.assertRaises(ValueError):
            E.t7_prediction_row("nasa", "NASA-20", 10, NASA_SPEC[3], NASA_SPEC[4], runner=_fake_runner("crash"))


class TestT7Prediction(unittest.TestCase):
    """Runs the real NASA and Apache adapters (offline, deterministic, no LLM)."""

    @classmethod
    def setUpClass(cls):
        cls.pycache = _pycache_dirs()
        cls.frozen = _tree_hashes(E.FROZEN_DIR)
        cls.t7_sources = {f: sha(E.REPO / f) for _, _, _, a, i in E.T7_PREDICTION for f in (a, *i)}
        cls.rows = E.extract_t7_prediction()
        cls.again = E.extract_t7_prediction()
        cls.by = {r["request_id"]: r for r in cls.rows}

    def test_one_record_per_request(self):
        self.assertEqual(len(self.rows), 6)
        self.assertEqual(set(self.by), {"CAR-03", "CAR-20", "NASA-03", "NASA-19", "APACHE-03", "APACHE-20"})
        for r in self.rows:
            self.assertEqual(r["treatment"], "T7")
            self.assertFalse(r["is_independent_trial"])
            self.assertEqual(r["protocol"], E.T7_PROTOCOL)

    def test_heldout_alignment_and_baseline(self):
        for r in self.rows:
            data = O.load(E.DATA_DIR / E.DATASETS[r["domain"]])
            mean, base, held = E.baseline(data, r["target"])
            self.assertTrue(r["heldout_rows_match"])
            self.assertEqual((r["n_heldout"], r["n_training"]), (len(held), data.n - len(held)))
            self.assertEqual((r["baseline_mean"], r["baseline_mae"]), (mean, base))
            self.assertEqual(r["target"], A.REQUESTS[r["request_id"]][1]["target"])
        self.assertEqual(self.by["NASA-03"]["n_heldout"], 18)
        self.assertEqual(self.by["APACHE-03"]["n_heldout"], 38)
        self.assertEqual((self.by["CAR-03"]["n_heldout"], self.by["CAR-03"]["n_training"]), (79, 319))

    def test_mae_recomputed_from_adapter_output(self):
        for domain, rid, step, adapter, impl in E.T7_PREDICTION:
            csv_path = E.DATA_DIR / E.DATASETS[domain]
            out = E.run_adapter(adapter, csv_path, E.request_ids(domain, step)).stdout
            preds = A.parse_answers(out)[rid]["predictions"]
            data = O.load(csv_path)
            target = A.REQUESTS[rid][1]["target"]
            mae = sum(abs(float(v) - data.value(int(k), target)) for k, v in preds.items()) / len(preds)
            self.assertAlmostEqual(self.by[rid]["model_mae"], mae, places=9)
            self.assertAlmostEqual(self.by[rid]["mae_ratio"], mae / self.by[rid]["baseline_mae"], places=12)
            self.assertTrue(self.by[rid]["reported_mae_matches"])

    def test_gates(self):
        for r in self.rows:
            self.assertEqual(r["answer_state"], "valid")
            self.assertIn(r["leakage_gate"], ("pass", "fail", "unverified"))
            self.assertEqual(r["primary"], r["leakage_gate"] == "pass")
            self.assertIn("mask", r["rerun_outcomes"])
            self.assertIn("shuffled", r["rerun_outcomes"])

    def test_cars_rows(self):
        a, b = self.by["CAR-03"], self.by["CAR-20"]
        for r, rid in ((a, "CAR-03"), (b, "CAR-20")):
            self.assertEqual(r["evaluated_requests"], rid)
            self.assertIn("LLM assistance", r["foundation"])
            self.assertIn(r["shuffled_gate"], ("pass", "fail"))
            self.assertNotIn("honeypot", r["rerun_outcomes"])
        self.assertEqual((a["model_mae"], a["baseline_mae"]), (b["model_mae"], b["baseline_mae"]))
        for rid in ("NASA-03", "APACHE-20"):
            self.assertIn("standalone", self.by[rid]["foundation"])

    def test_reproducible(self):
        self.assertEqual(self.rows, self.again)

    def test_evidence_isolated(self):
        self.assertFalse((E.REPO / "t7_validation" / "results").exists())
        self.assertEqual(_pycache_dirs(), self.pycache)
        self.assertEqual({f: sha(E.REPO / f) for f in self.t7_sources}, self.t7_sources)

    def test_frozen_results_untouched(self):
        self.assertEqual(_tree_hashes(E.FROZEN_DIR), self.frozen)


CARS_SPECS = {x[1]: x for x in E.T7_PREDICTION if x[0] == "cars"}


def _adapter_preds(rid, csv_path):
    d, _, step, adapter, _ = CARS_SPECS[rid]
    out = E.run_adapter(adapter, csv_path, (rid,))
    assert out.runs, out.stderr
    return {int(k): float(v) for k, v in A.parse_answers(out.stdout)[rid]["predictions"].items()}


class TestCarsT7Adapters(unittest.TestCase):
    """Cars T7 prediction adapters (phase 4C): orchestration of the shared library only."""

    @classmethod
    def setUpClass(cls):
        cls.protected = {str(p): sha(p) for base in ("human_t7", "shrinking_code") for p in (E.REPO / base).rglob("*.py")}
        cls.csv = E.DATA_DIR / "auto93.csv"
        cls.p03 = _adapter_preds("CAR-03", cls.csv)
        cls.p20 = _adapter_preds("CAR-20", cls.csv)

    def test_discovery(self):
        self.assertEqual(set(CARS_SPECS), {"CAR-03", "CAR-20"})
        self.assertEqual(E.T7_EVALUATED, {"CAR-03": ("CAR-03",), "CAR-20": ("CAR-20",)})
        for rid, (_, _, _, adapter, impl) in CARS_SPECS.items():
            self.assertTrue((E.REPO / adapter).is_file())
            self.assertIn(f"ANSWER {rid}", (E.REPO / adapter).read_text())
            self.assertIn("shrinking_code/prediction.py", impl)
        self.assertIn("LLM assistance", E.T7_FOUNDATION["cars"])

    def test_reuses_the_original_calls(self):
        guess = (E.REPO / "t7_validation/adapters/cars/guess.py").read_text()
        race = (E.REPO / "t7_validation/adapters/cars/race.py").read_text()
        self.assertIn("KNNRegressor.fit(training, k=K)", guess)
        self.assertIn("prediction_race(training, heldout, TARGET)", race)
        self.assertIn("KNNRegressor.fit(training, k=5)", (E.REPO / "human_t7/cars/guess.py").read_text())
        self.assertIn("prediction_race(training, test, target)", (E.REPO / "human_t7/cars/race.py").read_text())
        for src in (guess, race, (E.REPO / "t7_validation/adapters/cars/common.py").read_text()):
            for forbidden in ("class ", "sqrt", "sorted(", "distance"):
                self.assertNotIn(forbidden, src)                # no replacement model or distance code

    def test_predictions_equal_direct_library_calls(self):
        from shrinking_code.data import Row, Table
        from shrinking_code.prediction import KNNRegressor
        table = Table.load(self.csv)
        training = Table(table.columns, tuple(r for r in table.rows if r.number % 5))
        model = KNNRegressor.fit(training, k=5)
        for r in table.rows:
            if r.number % 5 == 0:
                hidden = Row(r.number, tuple(None if c.role == "y" else v for c, v in zip(table.columns, r.cells)))
                self.assertEqual(self.p03[r.number], model.predict(hidden, "Mpg+").value)

    def test_heldout_alignment(self):
        self.assertEqual(sorted(self.p03), O.held_out(CARS.n))
        self.assertEqual(sorted(self.p20), O.held_out(CARS.n))

    def test_car03_and_car20_predict_the_same(self):
        self.assertEqual(self.p03, self.p20)

    def test_masked_targets_do_not_move_predictions(self):
        from harness import semantic as S
        masked, _ = S.masked(CARS, "Mpg+")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "auto93.csv"
            path.write_text(masked.text())
            for rid, base in (("CAR-03", self.p03), ("CAR-20", self.p20)):
                self.assertEqual(_adapter_preds(rid, path), base, rid)

    def test_shuffled_copy_follows_rows(self):
        from harness import semantic as S
        shuffled, _ = S.shuffled(CARS)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "auto93.csv"
            path.write_text(shuffled.text())
            got = _adapter_preds("CAR-03", path)
        self.assertEqual(sorted(got), O.held_out(CARS.n))            # still the rows divisible by 5 of that file

    def test_leaky_cars_program_is_excluded(self):
        from harness.checks import CheckResult

        def leaky(adapter, csv_path, rids):
            data = O.load(csv_path)
            p = {str(r): data.value(r, "Mpg+") for r in O.held_out(data.n)}
            return CheckResult(True, False, False, 0.0, "ANSWER CAR-03 " + json.dumps({"predictions": p, "mae": 0}), "")
        d, rid, step, adapter, impl = CARS_SPECS["CAR-03"]
        r = E.t7_prediction_row(d, rid, step, adapter, impl, runner=leaky)
        self.assertEqual((r["leakage_gate"], r["exclusion_reason"], r["primary"]), ("fail", "leakage_failed", False))
        self.assertEqual(r["evaluated_requests"], "CAR-03")

    def test_original_code_unchanged(self):
        now = {str(p): sha(p) for base in ("human_t7", "shrinking_code") for p in (E.REPO / base).rglob("*.py")}
        self.assertEqual(now, self.protected)


CARS_FIX_CSV = """Clndrs,Volume,Model,origin,Lbs-,Acc+,Mpg+
4,100,70,1,2000,16,30
4,110,70,1,2100,15,28
8,300,72,1,4000,11,12
8,310,72,1,4100,10,11
6,200,75,2,3000,14,20
6,210,76,2,3100,13,19
4,105,80,3,2050,17,33
8,305,80,1,3900,12,14
"""
APACHE_FIX_CSV = """base,a,b,c,Performance-
1,0,0,0,100
1,1,0,0,200
1,0,1,0,300
1,1,1,0,400
1,0,0,1,500
1,1,0,1,600
1,0,1,1,700
1,1,1,1,800
"""


def _data(text):
    return O.parse(text)


class TestFixMetrics(unittest.TestCase):
    def setUp(self):
        self.cars = _data(CARS_FIX_CSV)
        self.ap = _data(APACHE_FIX_CSV)
        self.s = O.d2h(self.cars)

    def test_alternatives_alignment_and_improvement(self):
        ans = {"row": 4, "alternatives": [{"row": 3, "changes": {"Volume": [310, 300]}},
                                          {"row": 8, "changes": {"Model": [72, 80]}}]}
        m, fab, ok = E.fix_metrics(self.cars, "fix_alternatives", ans)
        best = min((3, 8), key=lambda a: self.s[a])
        self.assertEqual(fab, 0)
        self.assertEqual((m["reference_row"], m["candidate_row"]), (4, best))
        self.assertEqual(m["reference_score"], self.s[4])
        self.assertAlmostEqual(m["observed_improvement"], self.s[4] - self.s[best])
        near = O.XSpace(self.cars).nearest(4, 0.25)
        self.assertEqual(m["n_alternatives_similar"], sum(a in near for a in (3, 8)))
        self.assertEqual(m["n_alternatives_better"], sum(self.s[a] < self.s[4] for a in (3, 8)))
        self.assertEqual(ok, m["reference_poor"] and m["n_alternatives_better_and_similar"] > 0)

    def test_alternative_that_is_worse_is_not_success(self):
        ans = {"row": 1, "alternatives": [{"row": 4, "changes": {"Clndrs": [4, 8]}}]}   # row 1 is good, row 4 bad
        m, fab, ok = E.fix_metrics(self.cars, "fix_alternatives", ans)
        self.assertLess(m["observed_improvement"], 0)
        self.assertFalse(m["reference_poor"])
        self.assertFalse(ok)

    def test_fabricated_values_counted(self):
        ans = {"row": 4, "alternatives": [{"row": 3, "changes": {"Volume": [999, 300]}}]}
        self.assertEqual(E.fix_metrics(self.cars, "fix_alternatives", ans)[1], 1)

    def test_targets_direction_and_minimum(self):
        th = {"Lbs-": 2500, "Acc+": 15, "Mpg+": 25}           # rows 1, 2, 7 meet all three
        ans = {"row": 4, "targets": th, "alternative": 2, "changed": ["Clndrs", "Volume", "Model"]}
        m, fab, ok = E.fix_metrics(self.cars, "fix_targets", ans)
        goals = self.cars.goals()
        meets = lambda x: sum(w * self.cars.value(x, g) >= w * th[g] for g, w in goals)
        self.assertEqual((m["targets_met_reference"], m["targets_met_candidate"]), (meets(4), meets(2)))
        cands = [x for x in range(1, 9) if x != 4 and meets(x) > meets(4)]
        diff = lambda x: sum(self.cars.value(4, k) != self.cars.value(x, k) for k in self.cars.inputs())
        self.assertEqual(m["min_changes"], min(diff(x) for x in cands))
        self.assertEqual(m["excess_changes"], diff(2) - m["min_changes"])
        self.assertEqual(ok, m["excess_changes"] == 0)
        self.assertTrue(m["changes_listed_correctly"])

    def test_targets_out_of_range_fail(self):
        ans = {"row": 4, "targets": {"Lbs-": 1, "Acc+": 15, "Mpg+": 25}, "alternative": 2, "changed": []}
        m, _, ok = E.fix_metrics(self.cars, "fix_targets", ans)
        self.assertFalse(m["target_valid"])
        self.assertFalse(ok)

    def test_observed_improvement_and_best_possible(self):
        ans = {"row": 8, "alternative": 4, "changes": {"c": [1, 0]}, "performance": {"before": 800, "after": 400}}
        m, fab, ok = E.fix_metrics(self.ap, "fix_observed", ans)
        self.assertEqual((m["observed_improvement"], m["n_changes"], fab), (400, 1, 0))
        self.assertEqual(m["best_possible_improvement"], 800 - 200)    # rows 1-2 settings from row 8: best is 200
        self.assertAlmostEqual(m["improvement_captured"], 400 / 600)
        self.assertTrue(m["reported_before_matches"] and m["reported_after_matches"])
        self.assertTrue(ok)

    def test_zero_improvement_and_zero_denominator(self):
        ans = {"row": 1, "alternative": 2, "changes": {"a": [0, 1]}, "performance": {"before": 100, "after": 90}}
        m, _, ok = E.fix_metrics(self.ap, "fix_observed", ans)
        self.assertEqual(m["best_possible_improvement"], -100)          # nothing near row 1 is faster
        self.assertEqual(m["improvement_captured"], "")
        self.assertFalse(ok)                                              # worse, and row 1 is not poor
        self.assertFalse(m["reported_after_matches"])                     # the program's 90 is not trusted
        same = _data("base,a,Performance-\n1,0,500\n1,1,500\n1,0,100\n")
        m, _, ok = E.fix_metrics(same, "fix_observed", {"row": 1, "alternative": 2, "changes": {"a": [0, 1]},
                                                         "performance": {"before": 500, "after": 500}})
        self.assertEqual(m["observed_improvement"], 0)
        self.assertFalse(ok)

    def test_target_gap_and_minimum(self):
        ans = {"row": 8, "target": 150, "alternative": 1, "changed": ["a", "b", "c"]}
        m, _, ok = E.fix_metrics(self.ap, "fix_target", ans)
        self.assertEqual((m["gap_reference"], m["gap_candidate"], m["gap_reduction"]), (650, 50, 600))
        self.assertEqual(m["min_changes"], 1)                            # row 4 (400) is closer and 1 setting away
        self.assertEqual(m["excess_changes"], 2)
        self.assertFalse(ok)
        m, _, ok = E.fix_metrics(self.ap, "fix_target", dict(ans, alternative=4, changed=["c"]))
        self.assertTrue(ok)

    def test_target_worse_than_row_is_invalid(self):
        m, _, ok = E.fix_metrics(self.ap, "fix_target", {"row": 8, "target": 900, "alternative": 7, "changed": ["a"]})
        self.assertFalse(m["target_valid"])
        self.assertFalse(ok)

    def test_nasa_kinds_rejected(self):
        with self.assertRaises(ValueError):
            E.fix_metrics(self.cars, "fix_predicted", {})


class TestFixExtract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        cars = O.load(E.DATA_DIR / "auto93.csv")
        s = O.d2h(cars)
        worst = max(s, key=s.get)
        near = O.XSpace(cars).nearest(worst, 0.25)
        alt = min((r for r in near if s[r] < s[worst]), key=s.get)
        col = next(k for k in cars.inputs() if cars.value(worst, k) != cars.value(alt, k))
        good = json.dumps({"row": worst, "alternatives": [{"row": alt, "changes": {col: [cars.value(worst, col),
                                                                                       cars.value(alt, col)]}}]})
        fake = json.dumps({"row": worst, "alternatives": [{"row": alt, "changes": {col: [-1, cars.value(alt, col)]}}]})
        scen = {("T1", 1): ("ok", check_txt(f"ANSWER CAR-15 {good}")), ("T1", 2): ("ok", check_txt(f"ANSWER CAR-15 {fake}")),
                ("T1", 3): ("ok", check_txt(f"ANSWER CAR-15 {good}", runs=False)), ("T2", 1): ("parse_failed", None),
                ("T2", 2): ("ok", check_txt("ANSWER CAR-15 {\"row\": 1}")), ("T2", 3): ("ok", check_txt("nothing"))}
        cls.archive, cls.runs, cls.reqs = tmp / "z.zip", tmp / "runs.csv", tmp / "req.csv"
        sem = json.dumps({"status": "pass", "runs": {"base": "ran", "shuffled": "ran"},
                          "requests": [{"request_id": r, "status": "pass", "failed": [], "notes": []}
                                       for r in ("CAR-15", "CAR-16")]})
        with zipfile.ZipFile(cls.archive, "w") as z:
            for (t, k), (st, chk) in scen.items():
                if chk:
                    z.writestr(E.entry_name(ROOT, MODEL, "cars", t, k, 8, "check.txt"), chk)
                    z.writestr(E.entry_name(ROOT, MODEL, "cars", t, k, 8, "semantic.json"), sem)
        cls.runs.write_text("model,domain,treatment,trial,skill_index,status,runs,semantic\n" + "".join(
            f"{MODEL},cars,{t},{k},8,{st},True,pass\n" for (t, k), (st, _) in scen.items()))
        cls.reqs.write_text("model,domain,treatment,trial,request_id,status\n" + "".join(
            f"{MODEL},cars,{t},{k},{r},pass\n" for (t, k) in scen for r in ("CAR-15", "CAR-16")))
        cls.before = sha(cls.archive)
        cls.rows = E.extract_fix(cls.archive, cls.runs, cls.reqs, E.DATA_DIR, domains=("cars",))
        cls.by = {(r["treatment"], r["trial"], r["request_id"]): r for r in cls.rows}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_all_attempts_kept(self):
        self.assertEqual(len(self.rows), 36)                             # 2 requests x 6 treatments x 3 trials
        self.assertEqual(len(self.by), 36)

    def test_outcomes(self):
        want = {("T1", 1): "", ("T1", 2): "fabricated_values", ("T1", 3): "execution_failure",
                ("T2", 1): "generation_failed", ("T2", 2): "answer_schema_error", ("T2", 3): "answer_missing",
                ("T3", 1): "not_attempted"}
        for (t, k), reason in want.items():
            r = self.by[(t, k, "CAR-15")]
            self.assertEqual(r["exclusion_reason"], reason, (t, k))
            self.assertEqual(r["primary"], reason == "", (t, k))
            if reason:
                self.assertFalse(r["success"], (t, k))
        self.assertTrue(self.by[("T1", 1, "CAR-15")]["success"])
        self.assertEqual(self.by[("T1", 2, "CAR-15")]["fabricated_values"], 1)

    def test_summary_counts_only_successes(self):
        s = {(x["scope"], x["treatment"], x["request_id"]): x for x in E.summarize_fix(self.rows)}
        t1 = s[("domain_treatment", "T1", "CAR-15")]
        self.assertEqual((t1["attempted"], t1["executable"], t1["schema_valid"], t1["integrity_valid"],
                          t1["successful"]), (3, 2, 2, 1, 1))
        self.assertAlmostEqual(t1["delivery_rate"], 1 / 3)
        t2 = s[("domain_treatment", "T2", "CAR-15")]
        self.assertEqual((t2["integrity_valid"], t2["successful"], t2["conditional_success_rate"]), (0, 0, ""))

    def test_archive_untouched(self):
        self.assertEqual(sha(self.archive), self.before)


class TestFixRealArchive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = sha(E.ARCHIVE)
        cls.rows = E.extract_fix()
        cls.again = E.extract_fix()

    def test_all_attempts(self):
        self.assertEqual(len(self.rows), 108)
        self.assertEqual(len({(r["domain"], r["treatment"], r["trial"], r["request_id"]) for r in self.rows}), 108)

    def test_nasa_not_measurable(self):
        nasa = [r for r in self.rows if r["domain"] == "nasa"]
        self.assertEqual(len(nasa), 36)
        for r in nasa:
            self.assertFalse(r["measurable"] or r["primary"] or r["success"])
            self.assertIn("not independently measurable", r["notes"])
            self.assertEqual(r["observed_improvement"], "")

    def test_agrees_with_frozen_checker(self):
        z = zipfile.ZipFile(E.ARCHIVE)
        for r in self.rows:
            if not r["primary"]:
                continue
            data = O.load(E.DATA_DIR / E.DATASETS[r["domain"]])
            ans = A.parse_answers(E.split_check(z.read(r["check_entry"]).decode())[1])[r["request_id"]]
            items = {n.split(":", 1)[1]: ok for n, ok, _ in A.check_request(r["request_id"], data, ans).items}
            if r["kind"] in ("fix_targets", "fix_target") and "fewest changes" in items:
                self.assertEqual(items["fewest changes"], r["excess_changes"] == 0, r["request_id"])
            if r["kind"] == "fix_observed":
                self.assertEqual(items["alternative really is better"], r["observed_improvement"] > 0)
            if r["success"]:
                self.assertEqual(r["fabricated_values"], 0)

    def test_success_implies_valid(self):
        for r in self.rows:
            if r["success"]:
                self.assertTrue(r["primary"] and r["integrity_valid"] and r["measurable"])

    def test_reproducible_and_untouched(self):
        self.assertEqual(self.rows, self.again)
        self.assertEqual(sha(E.ARCHIVE), self.before)


if __name__ == "__main__":
    unittest.main()
