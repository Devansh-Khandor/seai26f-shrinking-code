"""Offline end-to-end smoke test of protocol v2 (no real LLM):
generation -> saved source -> sandboxed run -> semantic checks -> LOC -> failure category
-> runs.csv / requirements.csv -> analyze reports -> LLM diagnosis of failures -> tags."""

from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from harness.analyze import write_reports
from harness.backends import FakeBackend
from harness.diagnose import diagnose
from harness.results import load_rows
from harness.run import RunConfig, requirements_path, run
from harness.tag import tag_runs

REF = (Path(__file__).parent / "reference_solution.py").read_text(encoding="utf-8")


def reference_reply(prompt):
    """A 'model' that writes the known-good solution, importing ezr when it is offered."""
    skill = int(re.search(r"skill (\d+) of 10", prompt).group(1))
    ezr = "import ezr  # offered by T5/T6\n" if "FILE: ezr.py" in prompt else ""
    files = [("reference_solution.py", REF),
             (f"skill_{skill:02d}.py", f"import sys\n{ezr}from reference_solution import main\nmain(sys.argv[1], {skill})\n")]
    return "\n".join(f"FILE: {name}\n```python\n{text.rstrip()}\n```" for name, text in files)


def silent_reply(prompt):
    """A 'model' whose program passes the old format contract but prints no checkable answers."""
    skill = int(re.search(r"skill (\d+) of 10", prompt).group(1))
    ids = sorted(set(re.findall(r"\*\*([A-Z]+-\d\d):\*\* ", prompt)))
    body = "".join(f"print('== {i} ==')\nprint('a 1')\nprint('b 2')\nprint('c 3')\n" for i in ids)
    return f"FILE: skill_{skill:02d}.py\n```python\n{body}```"


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.results = self.root / "results" / "runs.csv"

    def tearDown(self):
        self._tmp.cleanup()

    def go(self, reply, treatments, skills):
        cfg = RunConfig(FakeBackend(reply), ["cars"], treatments, [1], skills, self.root / "runs", self.results)
        run(cfg, out=lambda s: None)
        return load_rows(self.results)

    def test_known_good_program_passes_semantic_in_t1_and_python_ezr_t5(self):
        rows = self.go(reference_reply, ["T1", "T5"], [1, 2, 3])
        self.assertEqual(len(rows), 6)
        for r in rows:
            self.assertEqual((r["status"], r["runs"], r["contract"], r["semantic"], r["failure_category"]),
                             ("ok", "True", "True", "pass", ""), r)
        req = load_rows(requirements_path(self.results))
        self.assertEqual(len(req), 12)
        self.assertTrue(all(r["status"] == "pass" and int(r["checks_total"]) > 0 for r in req))
        self.assertTrue(all(Path(self.root.parent, r["evidence"]).name.endswith(".semantic.json") for r in req))

        written = write_reports(rows, "fake", self.root / "report", req)
        names = sorted(p.name for p in written)
        self.assertIn("requirements.md", names)
        pr = (self.root / "report" / "pass_rates.md").read_text(encoding="utf-8")
        self.assertIn("| cars | T5 | 3 | 100% | 100% | 100% |", pr)

    def test_contract_pass_without_answers_is_a_semantic_failure_and_gets_diagnosed(self):
        rows = self.go(silent_reply, ["T1"], [1])
        r = rows[0]
        self.assertEqual((r["contract"], r["semantic"], r["failure_category"]), ("True", "fail", "code:no_answer"))
        pr = write_reports(rows, "fake", self.root / "report", load_rows(requirements_path(self.results)))
        text = (self.root / "report" / "pass_rates.md").read_text(encoding="utf-8")
        self.assertIn("code:no_answer 1", text)
        self.assertTrue(pr)

        judge = FakeBackend(lambda p: '{"cause": "code", "category": "code:no_answer", '
                                      '"explanation": "prints no ANSWER lines"}')
        diagnose(judge, self.results, self.root / "runs", out=lambda s: None)
        diag = load_rows(self.results.parent / "diagnoses.csv")
        self.assertEqual([(d["confirmed_category"], d["suspected_cause"], d["basis"]) for d in diag],
                         [("code:no_answer", "code", "suspected (LLM)")])
        self.assertIn("generated code", judge.calls[0].lower())
        diagnose(judge, self.results, self.root / "runs", out=lambda s: None)   # resumable: nothing new
        self.assertEqual(len(judge.calls), 1)

    def test_parse_failure_is_recorded_for_both_requests(self):
        rows = self.go(lambda p: "I cannot help with that.", ["T2"], [1])
        self.assertEqual((rows[0]["status"], rows[0]["semantic"], rows[0]["failure_category"]),
                         ("parse_failed", "parse_failure", "generation:extraction"))
        req = load_rows(requirements_path(self.results))
        self.assertEqual([r["status"] for r in req], ["parse_failure", "parse_failure"])

    @unittest.skipIf(shutil.which("git") is None, "git required")
    def test_finished_trial_tags_with_v2_prefix(self):
        repo = self.root / "repo"
        repo.mkdir()
        for args in (["init", "-q"], ["config", "user.name", "T"], ["config", "user.email", "t@example.com"]):
            subprocess.run(["git", "-C", str(repo), *args], check=True)
        self.go(reference_reply, ["T1"], list(range(1, 11)))
        new = tag_runs("fake", self.root / "runs", self.results, repo=repo, out=lambda s: None, prefix="v2-")
        self.assertEqual(new, [("cars", "T1", 1)])
        tags = subprocess.run(["git", "-C", str(repo), "tag", "-l"], capture_output=True, text=True).stdout.split()
        self.assertEqual(len(tags), 10)
        self.assertTrue(all(t.startswith("v2-t1-cars-r1-step") for t in tags))


if __name__ == "__main__":
    unittest.main()
