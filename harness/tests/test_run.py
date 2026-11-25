from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest import mock

from harness.backends import BackendError, Completion, FakeBackend
from harness.results import load_rows
from harness.run import RunConfig, parse_list, run
from harness.workspace import step_dir, trial_dir


def good_reply(prompt):
    ids = sorted(set(re.findall(r"\*\*(CAR-\d\d):\*\*", prompt)))
    skill = int(re.search(r"skill (\d+) of 10", prompt).group(1))
    lines = "".join(f"print('== {i} ==')\nprint('a 1')\nprint('b 2')\nprint('c 3')\n" for i in ids)
    return f"FILE: skill_{skill:02d}.py\n```python\nx = {skill}\n{lines}```\n"


class TestRun(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.out = []

    def tearDown(self):
        self._tmp.cleanup()

    def cfg(self, backend, **kw):
        base = dict(backend=backend, domains=["cars"], treatments=["T1"], trials=[1],
                    skills=[1, 2, 3], runs_root=self.root / "runs",
                    results_path=self.root / "runs.csv")
        base.update(kw)
        return RunConfig(**base)

    def go(self, backend, **kw):
        run(self.cfg(backend, **kw), out=self.out.append)
        return load_rows(self.root / "runs.csv")

    def trial(self, model="fake"):
        return trial_dir(self.root / "runs", model, "cars", "T1", 1)

    def test_happy_path(self):
        rows = self.go(FakeBackend(good_reply))
        self.assertEqual([r["status"] for r in rows], ["ok"] * 3)
        self.assertEqual([r["runs"] for r in rows], ["True"] * 3)
        self.assertEqual([r["contract"] for r in rows], ["True"] * 3)
        for i in (1, 2, 3):
            self.assertTrue(step_dir(self.trial(), i).is_dir())
        self.assertTrue(all(int(r["new_loc"]) > 0 for r in rows))
        self.assertIn("cars T1 trial 1 skill 3/10 ✓ runs ✓ contract", self.out[-1])

    def test_resume_no_new_calls(self):
        fake = FakeBackend(good_reply)
        self.go(fake)
        self.go(fake)
        self.assertEqual(len(fake.calls), 3)

    def test_resume_rebuilds_missing_step(self):
        fake = FakeBackend(good_reply)
        self.go(fake)
        shutil.rmtree(step_dir(self.trial(), 3))
        self.go(fake)
        self.assertEqual(len(fake.calls), 4)

    def test_overflow(self):
        fake = FakeBackend(good_reply, context_tokens=4200)
        rows = self.go(fake)
        self.assertEqual([r["status"] for r in rows], ["context_overflow"] * 3)
        self.assertEqual(rows[0]["new_loc"], "")
        self.assertEqual(rows[0]["runs"], "False")
        self.assertEqual(len(fake.calls), 0)
        self.assertTrue(step_dir(self.trial(), 3).is_dir())

    def test_backend_error_stops_trial(self):
        def flaky(prompt):
            if "skill 2 of 10" in prompt:
                raise BackendError("server exploded")
            return good_reply(prompt)

        rows = self.go(FakeBackend(flaky))
        self.assertEqual([(r["skill_index"], r["status"]) for r in rows],
                         [("1", "ok"), ("2", "backend_error")])
        self.assertFalse(step_dir(self.trial(), 2).exists())
        good = FakeBackend(good_reply)
        self.go(good)
        self.assertEqual(len(good.calls), 2)
        latest = {r["skill_index"]: r["status"] for r in load_rows(self.root / "runs.csv")}
        self.assertEqual(latest, {"1": "ok", "2": "ok", "3": "ok"})

    def test_parse_failed_continues(self):
        def lazy(prompt):
            return "no code" if "skill 2 of 10" in prompt else good_reply(prompt)

        rows = self.go(FakeBackend(lazy))
        self.assertEqual([r["status"] for r in rows], ["ok", "parse_failed", "ok"])
        self.assertEqual(rows[1]["new_loc"], "")

    def test_prior_code_flows_into_next_prompt(self):
        fake = FakeBackend(good_reply)
        self.go(fake)
        self.assertIn("FILE: skill_01.py", fake.calls[1])
        self.assertIn("FILE: skill_02.py", fake.calls[2])

    def test_dry_run(self):
        fake = FakeBackend(good_reply)
        self.go(fake, dry_run=True, treatments=["T1", "T5"])
        self.assertEqual(len(fake.calls), 0)
        self.assertTrue((self.root / "runs" / "_dryrun" / "cars_T1.txt").exists())
        self.assertTrue(any("cars T5 skill1 est" in line for line in self.out))
        self.assertFalse((self.root / "runs.csv").exists())

    def test_runtime_files_not_counted_as_code(self):
        def writer(prompt):
            reply = good_reply(prompt)
            extra = ("open('helper_out.py', 'w').write('x = 1\\n' * 200)\n"
                     "open('state.json', 'w').write('{}' * 5000)\n")
            return reply.replace("```python\n", "```python\n" + extra, 1)

        fake = FakeBackend(writer)
        rows = self.go(fake)
        self.assertEqual([r["status"] for r in rows], ["ok"] * 3)
        self.assertLess(int(rows[0]["new_loc"]), 20)
        self.assertNotIn("FILE: state.json", fake.calls[1])
        self.assertNotIn("FILE: helper_out.py", fake.calls[1])
        self.assertFalse((step_dir(self.trial(), 1) / "state.json").exists())

    def test_preflight_requires_lua_for_ezr(self):
        with mock.patch("harness.run.lua_executable", return_value=None):
            with self.assertRaises(SystemExit):
                self.go(FakeBackend(good_reply), treatments=["T5"])

    def test_preflight_requires_sklearn(self):
        with mock.patch("harness.run.python_has_sklearn", return_value=False):
            with self.assertRaises(SystemExit):
                self.go(FakeBackend(good_reply), treatments=["T3"])

    def test_unwritable_path_is_parse_failed(self):
        def bad(prompt):
            if "skill 2 of 10" in prompt:
                return "FILE: skill_01.py/inner.py\n```python\nx = 1\n```"
            return good_reply(prompt)

        rows = self.go(FakeBackend(bad))
        self.assertEqual([r["status"] for r in rows], ["ok", "parse_failed", "ok"])

    def test_length_cutoff_is_output_truncated(self):
        def cut(prompt):
            text = good_reply(prompt)
            return Completion(text, 10, 99, "length") if "skill 2 of 10" in prompt else text

        rows = self.go(FakeBackend(cut))
        self.assertEqual([r["status"] for r in rows], ["ok", "output_truncated", "ok"])
        self.assertEqual(rows[1]["finish_reason"], "length")
        self.assertEqual(rows[1]["new_loc"], "")

    def test_vendored_ezr_in_reply_dropped(self):
        def lua_reply(prompt):
            skill = int(re.search(r"skill (\d+) of 10", prompt).group(1))
            return (f"FILE: ezr.lua\n```lua\nlocal x = 1\n```\n"
                    f"FILE: skill_{skill:02d}.lua\n```lua\nprint(1)\n```\n")

        with mock.patch("harness.run.lua_executable", return_value="lua"):
            self.go(FakeBackend(lua_reply), treatments=["T5"], skills=[1])
        t5 = trial_dir(self.root / "runs", "fake", "cars", "T5", 1)
        self.assertFalse((step_dir(t5, 1) / "ezr.lua").exists())
        self.assertTrue((step_dir(t5, 1) / "skill_01.lua").exists())

    def test_trial_outer_loop(self):
        fake = FakeBackend(good_reply)
        self.go(fake, treatments=["T1", "T2"], trials=[1, 2], skills=[1])
        order = [(r["trial"], r["treatment"]) for r in load_rows(self.root / "runs.csv")]
        self.assertEqual(order, [("1", "T1"), ("1", "T2"), ("2", "T1"), ("2", "T2")])

    def test_cli_think_flag(self):
        from harness.run import build_parser
        a = build_parser().parse_args(["--backend", "ollama", "--model", "gpt-oss:20b", "--think", "low"])
        self.assertEqual(a.think, "low")

    def test_ctrl_c_exits_cleanly(self):
        from harness import run as run_mod
        with mock.patch.object(run_mod, "run", side_effect=KeyboardInterrupt), \
             mock.patch("builtins.print") as fake_print:
            with self.assertRaises(SystemExit) as cm:
                run_mod.main(["--backend", "fake", "--model", "fake"])
        self.assertEqual(cm.exception.code, 130)
        self.assertIn("rerun the same command", str(fake_print.call_args))

    def test_parse_list(self):
        self.assertEqual(parse_list("T1-T6", "T"), ["T1", "T2", "T3", "T4", "T5", "T6"])
        self.assertEqual(parse_list("T1,T3", "T"), ["T1", "T3"])
        self.assertEqual(parse_list("1-10"), [str(i) for i in range(1, 11)])
        self.assertEqual(parse_list("cars,nasa"), ["cars", "nasa"])


if __name__ == "__main__":
    unittest.main()
