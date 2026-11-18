"""Prudence tests for the semantic checker.

1. A known-good, independently written solution (reference_solution.py) passes all 60 requests.
2. Plausible-looking but wrong answers fail, for every kind of request.
3. The controlled reruns catch hard-coding, leakage, peeking at the budget, and reading the future.
4. Execution failures, checker errors and timeouts are kept apart from wrong answers.
"""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import unittest
from unittest import mock

from harness import answers as A
from harness import oracle as O
from harness.checks import CheckResult, contract_ok
from harness.failures import categorize, crash_category
from harness.requirements import DOMAIN_FILES, REPO_ROOT, load_domain
from harness.semantic import evaluate, honeypot
from harness.tests import reference_solution as ref

DOMAINS = {d: load_domain(d) for d in DOMAIN_FILES}
DATA = {d: O.load(dom.csv_path) for d, dom in DOMAINS.items()}


def ref_stdout(path, skill):
    buf = io.StringIO()
    with redirect_stdout(buf):
        ref.main(str(path), skill)
    return buf.getvalue()


_CACHE = {}


def ref_answers(domain, skill):
    if (domain, skill) not in _CACHE:
        _CACHE[(domain, skill)] = A.parse_answers(ref_stdout(DOMAINS[domain].csv_path, skill))
    return copy.deepcopy(_CACHE[(domain, skill)])


def answer(rid):
    d = {"CAR": "cars", "NASA": "nasa", "APACHE": "apache"}[rid.split("-")[0]]
    skill = (int(rid.split("-")[1]) + 1) // 2
    return d, ref_answers(d, skill)[rid]


def failures(rid, ans, domain=None):
    d = domain or {"CAR": "cars", "NASA": "nasa", "APACHE": "apache"}[rid.split("-")[0]]
    c = A.check_request(rid, DATA[d], ans)
    return [f"{n}: {x}" for n, ok, x in c.items if not ok]


def result(stdout, runs=True, timed_out=False, stderr=""):
    return CheckResult(runs, runs and contract_ok(stdout, []), timed_out, 1.0, stdout, stderr)


def ref_rerun(skill):
    return lambda path: result(ref_stdout(path, skill))


class TestCoverage(unittest.TestCase):
    def test_all_60_requests_have_a_kind_and_text(self):
        ids = [r.id for dom in DOMAINS.values() for s in dom.skills for r in s.requests]
        self.assertEqual(len(ids), 60)
        self.assertEqual(sorted(ids), sorted(A.REQUESTS))
        for dom in DOMAINS.values():
            for s in dom.skills:
                for r in s.requests:
                    text = A.spec_text(r.id, DATA[dom.name])
                    self.assertIn(f"ANSWER {r.id}", text)
                    self.assertNotIn("{", text.split("JSON is ")[0])

    def test_frozen_requirements_untouched(self):
        for d, (req, _) in DOMAIN_FILES.items():
            text = (REPO_ROOT / req).read_text(encoding="utf-8")
            self.assertIn("Frozen requirements baseline", text)
            self.assertNotIn("ANSWER", text)


class TestKnownGood(unittest.TestCase):
    """The reference solution uses other aggregates/distances/tests than the oracle and must still pass."""

    def test_reference_passes_every_request(self):
        for d, dom in DOMAINS.items():
            for s in dom.skills:
                got = ref_answers(d, s.index)
                for r in s.requests:
                    with self.subTest(request=r.id):
                        self.assertIn(r.id, got)
                        self.assertEqual(failures(r.id, got[r.id]), [])


class TestWrongAnswersFail(unittest.TestCase):
    """Each mutation keeps the format valid but makes the answer wrong."""

    def expect(self, rid, mutate, *needles):
        d, a = answer(rid)
        mutate(a, DATA[d])
        fails = failures(rid, a)
        self.assertTrue(fails, f"{rid}: mutated answer passed")
        for n in needles:
            self.assertTrue(any(n in f for f in fails), f"{rid}: expected a failure about {n!r}, got {fails}")

    # Remember
    def test_summary_best_worst_swapped(self):
        def m(a, data):
            a["Mpg+"]["best"], a["Mpg+"]["worst"] = a["Mpg+"]["worst"], a["Mpg+"]["best"]
        self.expect("CAR-01", m, "Mpg+ best", "Mpg+ worst")

    def test_summary_wrong_mean(self):
        self.expect("NASA-01", lambda a, d: a["Effort-"].update(mean=a["Effort-"]["mean"] * 1.1), "Effort- mean")

    def test_top_is_the_worst_rows(self):
        def m(a, data):
            s = O.d2h(data)
            a["top"] = sorted(s, key=s.get, reverse=True)[:10]
        self.expect("CAR-02", m, "rows are good overall")

    def test_top_by_single_goal(self):
        def m(a, data):
            a["top"] = sorted(range(1, data.n + 1), key=lambda r: -data.value(r, "Kloc+"))[:10]
        self.expect("NASA-02", m)

    def test_settings_shares_invented(self):
        self.expect("APACHE-02", lambda a, d: a["settings"][0].update(top_share=a["settings"][0]["top_share"] - 0.3),
                    "shares")

    # Guess
    def test_reported_mae_inconsistent(self):
        # Devansh's example: plausible numbers, error reported as 0
        self.expect("CAR-03", lambda a, d: a.update(mae=0), "mae is computed correctly")

    def test_constant_prediction_no_better_than_mean(self):
        def m(a, data):
            a["predictions"] = {k: 999.0 for k in a["predictions"]}
            a["mae"] = O.mean([abs(999 - data.value(int(k), "Effort-")) for k in a["predictions"]])
        self.expect("NASA-03", m, "beats the historical mean")

    def test_missing_heldout_row(self):
        self.expect("APACHE-03", lambda a, d: a["predictions"].pop(next(iter(a["predictions"]))), "answer format")

    def test_intervals_that_miss(self):
        def m(a, data):
            for v in a["predictions"].values():
                v.update(estimate=v["estimate"] + 5000, low=v["estimate"] + 4999, high=v["estimate"] + 5001)
        self.expect("CAR-04", m, "ranges cover the truth")

    def test_pair_differs_in_two_settings(self):
        def m(a, data):
            p = a["pairs"][0]
            inputs = data.inputs()
            for b in range(1, data.n + 1):
                if sum(data.value(p["a"], c) != data.value(b, c) for c in inputs) == 2:
                    p["b"] = b
                    break
        self.expect("APACHE-04", m, "differs only in")

    # Certify
    def test_clear_difference_called_same(self):
        def m(a, data):
            for c in a["comparisons"]:
                if {c["a"], c["b"]} == {"4", "8"}:
                    c["verdict"] = "same"
        self.expect("CAR-05", m, "4.0 vs 8.0 verdict")

    def test_group_count_wrong(self):
        self.expect("NASA-06", lambda a, d: a["groups"]["h"].update(n=a["groups"]["h"]["n"] + 3), "cplx=h n")

    def test_year_group_count_wrong(self):
        self.expect("CAR-06", lambda a, d: a["groups"][0].update(n=a["groups"][0]["n"] + 1), " n")

    # Flow
    def test_desirable_inverted(self):
        self.expect("CAR-07", lambda a, d: [e.update(desirable=not e["desirable"]) for e in a["assessments"]],
                    "desirable arrivals are better")

    def test_predictions_against_the_truth(self):
        def m(a, data):
            for e in a["assessments"]:
                e["predicted_effort"] = -data.value(e["row"], "Effort-")
        self.expect("NASA-07", m, "effort predictions track")

    def test_alert_on_a_good_arrival(self):
        def m(a, data):
            best = min(a["assessments"], key=lambda e: data.value(e["row"], "Performance-"))
            best["alert"] = True
        self.expect("APACHE-07", m, "alert is justified")

    def test_summary_uses_rows_not_yet_arrived(self):
        def m(a, data):
            for g, _ in data.goals():
                vs = data.values(g)
                a["checkpoints"][0]["summary"][g].update(n=len(vs), mean=O.mean(vs), min=min(vs), max=max(vs))
        self.expect("CAR-08", m, "after 0:")

    # Blame
    def test_poor_rows_are_the_best(self):
        def m(a, data):
            a["poor"] = sorted(range(1, data.n + 1), key=lambda r: -data.value(r, "Mpg+"))[:20]
        self.expect("CAR-09", m, "poor rows are poor")

    def test_constant_column_ranked_first(self):
        self.expect("NASA-09", lambda a, d: a.update(ranking=["prec"] + [f for f in a["ranking"] if f != "prec"]),
                    "top-ranked column")

    def test_typical_value_invented(self):
        self.expect("CAR-10", lambda a, d: a["differences"][0].update(bad=12345.0), "bad value")

    def test_combo_share_invented(self):
        def m(a, data):
            c = a["combinations"][0]
            c["bad_share"] = c["bad_share"] - 0.3 if c["bad_share"] > 0.5 else c["bad_share"] + 0.3
        self.expect("APACHE-10", m, "shares")

    # Justify
    def test_neighbors_are_far_away(self):
        def m(a, data):
            x = O.XSpace(data)
            far = sorted(range(1, data.n + 1), key=lambda r: -x.dist(a["recommended"], r))[:4]
            a["neighbors"] = [{"row": r, "values": {g: data.value(r, g) for g, _ in data.goals()}} for r in far]
        self.expect("CAR-11", m, "is similar")

    def test_cited_values_invented(self):
        self.expect("APACHE-11", lambda a, d: a["neighbors"][0]["values"].update({"Performance-": 1.0}), "values are real")

    def test_project_not_held_out(self):
        self.expect("NASA-11", lambda a, d: a.update(project=a["project"] + 1), "project is held out")

    def test_tradeoffs_hidden(self):
        self.expect("CAR-12", lambda a, d: a.update(tradeoffs=[]), "tradeoffs are correct")

    # Choose
    def test_best_not_evaluated(self):
        def m(a, data):
            a["best"] = next(r for r in range(1, data.n + 1) if r not in a["evaluated"])
        self.expect("CAR-13", m, "best was evaluated")

    def test_over_budget(self):
        self.expect("APACHE-13", lambda a, d: a.update(evaluated=list(range(1, 41)), best=1), "budget")

    def test_rule_count_wrong(self):
        self.expect("NASA-13", lambda a, d: a.update(matching=a["matching"] + 2), "matching count")

    def test_order_starts_with_the_worst(self):
        def m(a, data):
            s = O.d2h(data)
            a["order"] = sorted(s, key=s.get, reverse=True)[:30]
        self.expect("CAR-14", m, "order favors promising rows")

    # Fix
    def test_alternative_is_worse(self):
        def m(a, data):
            s = O.zsum(data)
            worse = max(s, key=s.get)
            a["alternatives"] = [{"row": worse, "changes": {c: [data.value(a["row"], c), data.value(worse, c)]
                                                            for c in data.inputs()
                                                            if data.value(a["row"], c) != data.value(worse, c)}}]
        self.expect("CAR-15", m, "is better")

    def test_change_list_incomplete(self):
        def m(a, data):
            if len(a["changes"]) > 1:
                a["changes"].pop(next(iter(a["changes"])))
            else:
                a["changes"]["handle"] = [data.value(a["row"], "handle"), data.value(a["row"], "handle")]
        self.expect("APACHE-15", m)

    def test_changes_size_column(self):
        self.expect("NASA-15", lambda a, d: a.update(changes={"Kloc+": [d.value(a["project"], "Kloc+"), 1.0]}),
                    "answer format")

    def test_prediction_gets_worse(self):
        self.expect("NASA-16", lambda a, d: a["predicted_defects"].update(after=a["predicted_defects"]["before"] * 2),
                    "prediction improves")

    def test_not_the_fewest_changes(self):
        def m(a, data):
            th = {g: num for g, num in a["targets"].items()}
            meets = lambda x: sum((data.value(x, g) >= th[g]) if g.endswith("+") else (data.value(x, g) <= th[g])
                                  for g, _ in data.goals())
            diff = lambda x: [c for c in data.inputs() if data.value(a["row"], c) != data.value(x, c)]
            cands = [x for x in range(1, data.n + 1) if meets(x) > meets(a["row"])]
            far = max(cands, key=lambda x: len(diff(x)))
            a["alternative"], a["changed"] = far, diff(far)
        self.expect("CAR-16", m, "fewest changes")

    def test_changed_settings_wrong(self):
        self.expect("APACHE-16", lambda a, d: a.update(changed=["base"]), "changed settings are correct")

    # Spot
    def test_flags_the_most_typical_rows(self):
        def m(a, data):
            k = O.XSpace(data).knn_scores()
            fl = sorted(k, key=k.get)[:len(a["flagged"])]
            a["flagged"], a["scores"] = fl, {str(r): 1.0 for r in fl}
        self.expect("CAR-17", m, "flagged rows are unusual")

    def test_actual_value_invented(self):
        self.expect("CAR-18", lambda a, d: a["flagged"][0].update(actual=-1.0), "actual is real")

    # Race
    def test_random_arm_sabotaged(self):
        def m(a, data):
            s = O.d2h(data)
            worst = max(s, key=s.get)
            a["random"] = [worst] * len(a["random"])
        self.expect("CAR-19", m, "random baseline is really random")

    def test_baseline_not_the_mean(self):
        self.expect("NASA-19", lambda a, d: a.update(baseline=a["baseline"] * 0.5), "baseline is the historical mean")

    def test_claims_the_wrong_winner(self):
        def m(a, data):
            a["verdict"] = "worse" if a["verdict"] == "better" else "better"
        self.expect("APACHE-20", m, "verdict follows")


class TestControlledRuns(unittest.TestCase):
    """End to end through semantic.evaluate, with the reference playing the generated program."""

    def run_eval(self, domain, skill, base_stdout=None, rerun=None):
        dom = DOMAINS[domain]
        rids = [r.id for r in dom.skills[skill - 1].requests]
        base = result(base_stdout if base_stdout is not None else ref_stdout(dom.csv_path, skill))
        return evaluate(rids, dom.csv_path, base, rerun or ref_rerun(skill))

    def test_honest_program_passes_every_rerun(self):
        for domain, skill in (("cars", 2), ("cars", 4), ("nasa", 7), ("apache", 10)):
            with self.subTest(domain=domain, skill=skill):
                sem = self.run_eval(domain, skill)
                self.assertEqual(sem.status, "pass", [r.failed for r in sem.requests])
                self.assertTrue(all(v == "ran" for v in sem.runs.values()))

    def test_hardcoded_output_fails_on_shuffled_rows(self):
        fixed = ref_stdout(DOMAINS["cars"].csv_path, 1)
        sem = self.run_eval("cars", 1, fixed, lambda path: result(fixed))
        self.assertEqual(sem.status, "fail")
        self.assertTrue(any(f.startswith("shuffled:") for r in sem.requests for f in r.failed))

    def test_leaking_held_out_targets_is_caught(self):
        def leaky(path):
            data = O.load(path)
            preds = {str(r): data.value(r, "Mpg+") + 0.5 for r in O.held_out(data.n)}
            out = ref_stdout(path, 2).splitlines()
            out = [ln if not ln.startswith("ANSWER CAR-03") else
                   "ANSWER CAR-03 " + json.dumps({"predictions": preds,
                                                  "mae": O.mean([0.5] * len(preds))}) for ln in out]
            return "\n".join(out) + "\n"
        sem = self.run_eval("cars", 2, leaky(DOMAINS["cars"].csv_path), lambda p: result(leaky(p)))
        car03 = sem.requests[0]
        self.assertEqual(car03.status, "fail")
        self.assertTrue(any("held-out targets not used" in f for f in car03.failed), car03.failed)

    def test_other_goal_as_input_is_not_leakage(self):
        # Predicting Mpg+ from a held-out car's Lbs- is allowed: only Mpg+ is unknown.
        # The mask rerun for CAR-03 must therefore leave Lbs- alone (and vice versa).
        def fit(path):
            data = O.load(path)
            ho = set(O.held_out(data.n))
            tr = [r for r in range(1, data.n + 1) if r not in ho]
            xs, ys = [data.value(r, "Lbs-") for r in tr], [data.value(r, "Mpg+") for r in tr]
            mx, my = O.mean(xs), O.mean(ys)
            b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
            preds = {str(r): my + b * (data.value(r, "Lbs-") - mx) for r in sorted(ho)}
            mae = O.mean([abs(p - data.value(int(r), "Mpg+")) for r, p in preds.items()])
            out = ref_stdout(path, 2).splitlines()
            out = [ln if not ln.startswith("ANSWER CAR-03") else
                   "ANSWER CAR-03 " + json.dumps({"predictions": preds, "mae": mae}) for ln in out]
            return "\n".join(out) + "\n"
        sem = self.run_eval("cars", 2, fit(DOMAINS["cars"].csv_path), lambda p: result(fit(p)))
        self.assertEqual(sem.status, "pass", [r.failed for r in sem.requests])
        self.assertEqual(set(sem.runs), {"base", "shuffled", "mask Lbs-", "mask Mpg+"})

    def test_peeking_at_goals_is_caught(self):
        def peek(path):
            data = O.load(path)
            s = O.d2h(data)
            best = sorted(s, key=s.get)[:O.budget(data.n)]
            out = ref_stdout(path, 7).splitlines()
            out = [ln if not ln.startswith("ANSWER CAR-13") else
                   "ANSWER CAR-13 " + json.dumps({"evaluated": best, "best": best[0]}) for ln in out]
            return "\n".join(out) + "\n"
        sem = self.run_eval("cars", 7, peek(DOMAINS["cars"].csv_path), lambda p: result(peek(p)))
        car13 = sem.requests[0]
        self.assertEqual(car13.status, "fail")
        self.assertTrue(any("respects the budget" in f for f in car13.failed), car13.failed)

    def test_honeypots_are_planted_with_ideal_goals(self):
        data, extra = honeypot(DATA["cars"])
        self.assertEqual(data.n, DATA["cars"].n + 6)
        for r in extra["honeypots"]:
            self.assertEqual(data.value(r, "Mpg+"), max(DATA["cars"].values("Mpg+")))
            self.assertEqual(data.value(r, "Lbs-"), min(DATA["cars"].values("Lbs-")))

    def test_reading_future_arrivals_is_caught(self):
        def future_reader(path):
            data = O.load(path)
            s = O.zsum(data)   # ranks each arrival against the WHOLE file, future rows included
            cut = sorted(s.values())[len(s) // 4]
            rows = O.arrivals(data.n)
            ans = {"assessments": [{"row": r, "desirable": s[r] <= cut} for r in rows]}
            out = ref_stdout(path, 4).splitlines()
            out = [ln if not ln.startswith("ANSWER CAR-07") else "ANSWER CAR-07 " + json.dumps(ans) for ln in out]
            return "\n".join(out) + "\n"
        sem = self.run_eval("cars", 4, future_reader(DOMAINS["cars"].csv_path), lambda p: result(future_reader(p)))
        car07 = sem.requests[0]
        self.assertEqual(car07.status, "fail")
        self.assertTrue(any("no use of future arrivals" in f for f in car07.failed), car07.failed)

    def test_crash_on_controlled_copy_fails(self):
        sem = self.run_eval("cars", 1, None, lambda p: result("", runs=False, stderr="Traceback\nKeyError: 3"))
        self.assertEqual(sem.status, "fail")
        self.assertTrue(all(any("program runs" in f for f in r.failed) for r in sem.requests))

    def test_timeout_on_controlled_copy_is_unverified_not_fail(self):
        sem = self.run_eval("cars", 1, None, lambda p: result("", runs=False, timed_out=True))
        self.assertEqual(sem.status, "unverified")
        self.assertTrue(all(r.status == "unverified" for r in sem.requests))

    def test_execution_failure_is_kept(self):
        dom = DOMAINS["cars"]
        crashed = CheckResult(False, False, False, 1.0, "", "Traceback\nZeroDivisionError: x")
        sem = evaluate(["CAR-05", "CAR-06"], dom.csv_path, crashed, lambda p: self.fail("no rerun"))
        self.assertEqual(sem.status, "execution_failure")
        self.assertEqual([r.status for r in sem.requests], ["execution_failure"] * 2)

    def test_checker_bug_is_a_checker_error(self):
        boom = mock.Mock(side_effect=RuntimeError("checker bug"))
        with mock.patch.dict(A.KINDS, {"groups": (A.KINDS["groups"][0], boom, ())}):
            sem = self.run_eval("cars", 3)
        self.assertEqual(sem.status, "checker_error")
        self.assertIn("checker bug", sem.requests[0].notes[0])

    def test_missing_answer_line_fails(self):
        stdout = "\n".join(ln for ln in ref_stdout(DOMAINS["nasa"].csv_path, 5).splitlines()
                           if not ln.startswith("ANSWER NASA-09"))
        sem = self.run_eval("nasa", 5, stdout)
        self.assertEqual(sem.requests[0].status, "fail")
        self.assertEqual(sem.requests[1].status, "pass")

    def test_deterministic(self):
        a = self.run_eval("apache", 4)
        b = self.run_eval("apache", 4)
        self.assertEqual([(r.status, r.passed, r.total, r.failed) for r in a.requests],
                         [(r.status, r.passed, r.total, r.failed) for r in b.requests])

    def test_format_contract_pass_but_semantic_fail(self):
        # passes the original checker: header, 3 lines, digits; but the numbers are wrong
        bogus = ("== CAR-03 ==\nPrediction results\nPredicted MPG: 999\nActual MPG: 25\nError: 0\n"
                 "ANSWER CAR-03 " + json.dumps({"predictions": {str(r): 999 for r in O.held_out(398)}, "mae": 0}) +
                 "\n== CAR-04 ==\nweight 1\nweight 2\nweight 3\n")
        self.assertTrue(contract_ok(bogus, ["CAR-03", "CAR-04"]))
        sem = self.run_eval("cars", 2, bogus, lambda p: result(bogus))
        self.assertEqual(sem.status, "fail")


class TestEvidence(unittest.TestCase):
    def test_evidence_file_records_every_check(self):
        import tempfile
        dom = DOMAINS["apache"]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "step03.semantic.json"
            evaluate(["APACHE-05", "APACHE-06"], dom.csv_path, result(ref_stdout(dom.csv_path, 3)),
                     ref_rerun(3), evidence=path)
            rec = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(rec["status"], "pass")
        self.assertEqual([r["request_id"] for r in rec["requests"]], ["APACHE-05", "APACHE-06"])


class TestFailureCategories(unittest.TestCase):
    def crashed(self, stderr):
        return CheckResult(False, False, False, 1.0, "", stderr)

    def test_generation_and_system_statuses(self):
        self.assertEqual(categorize("parse_failed")[0], "generation:extraction")
        self.assertEqual(categorize("output_truncated")[0], "generation:truncated")
        self.assertEqual(categorize("context_overflow")[0], "system:context_overflow")

    def test_syntax_error(self):
        err = 'File "skill_01.py", line 3\n    def (\n        ^\nSyntaxError: invalid syntax'
        self.assertEqual(categorize("ok", self.crashed(err), None, "plain")[0], "code:syntax")

    def test_missing_allowed_module_is_the_machines_fault(self):
        err = "Traceback\nModuleNotFoundError: No module named 'sklearn'"
        self.assertEqual(crash_category(err, "sklearn")[0], "system:dependency")
        self.assertEqual(crash_category(err, "plain")[0], "code:forbidden_import")

    def test_ezr_api_misuse(self):
        err = "Traceback\nImportError: cannot import name 'Data' from 'ezr' (vendor/python/ezr.py)"
        self.assertEqual(crash_category(err, "ezr")[0], "code:api_misuse")
        err2 = 'Traceback\n  File "x/vendor/python/ezr.py", line 9, in add\nAttributeError: \'list\' object has no attribute \'at\''
        self.assertEqual(crash_category(err2, "ezr")[0], "code:api_misuse")

    def test_plain_runtime_error(self):
        self.assertEqual(crash_category("Traceback\nZeroDivisionError: division by zero", "plain"),
                         ("code:runtime", "ZeroDivisionError: division by zero"))

    def test_timeout(self):
        r = CheckResult(False, False, True, 120.0, "", "")
        self.assertEqual(categorize("timeout", r)[0], "runtime:timeout")

    def test_semantic_categories(self):
        ok = CheckResult(True, True, False, 1.0, "x", "")
        dom = DOMAINS["cars"]
        sem = evaluate(["CAR-01", "CAR-02"], dom.csv_path, result("no answers\n"), lambda p: self.fail("no rerun"))
        self.assertEqual(categorize("ok", ok, sem)[0], "code:no_answer")
        good = evaluate(["CAR-01", "CAR-02"], dom.csv_path, result(ref_stdout(dom.csv_path, 1)), ref_rerun(1))
        self.assertEqual(categorize("ok", ok, good), ("", ""))


if __name__ == "__main__":
    unittest.main()
